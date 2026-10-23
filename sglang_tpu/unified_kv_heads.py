# SPDX-License-Identifier: Apache-2.0
"""Merged sliding-window KV heads of the unified KV pool (see unified_kv_pool).

A leaf module: the attention backends import it when they load, which happens
before out-of-tree plugins may swap the KV pool base classes, so it must not
import any KV pool module.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def merge_swa_kv_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    target_kv_heads: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``k``/``v`` ``[T, Hkv, D]`` into ``target_kv_heads`` KV heads
    ``[T, target_kv_heads, g * D]`` (where ``g = Hkv // target_kv_heads``)
    and block-diagonalize ``q`` ``[T, Hq, D]`` -> ``[T, Hq, g * D]``.

    Uses a one-hot ``torch.eye`` multiply so ``torch_tpu`` avoids strided
    slice concatenations.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_kv_heads <= target_kv_heads:
        return q, k, v
    group_factor = num_kv_heads // target_kv_heads
    merged_dim = group_factor * head_dim
    eye = torch.eye(group_factor, dtype=q.dtype, device=q.device).view(
        1, 1, group_factor, 1, group_factor, 1
    )
    q_merged = (
        q.reshape(
            num_tokens,
            target_kv_heads,
            group_factor,
            num_q_heads // num_kv_heads,
            1,
            head_dim,
        )
        * eye
    ).reshape(num_tokens, num_q_heads, merged_dim)
    return (
        q_merged,
        k.reshape(num_tokens, target_kv_heads, merged_dim),
        v.reshape(num_tokens, target_kv_heads, merged_dim),
    )


def split_swa_output(
    out: torch.Tensor, num_kv_heads: int, target_kv_heads: int = 1
) -> torch.Tensor:
    """Unpack block-diagonal SWA attention output ``[T, Hq, g * D]`` -> ``[T, Hq, D]``."""
    if num_kv_heads <= target_kv_heads:
        return out
    group_factor = num_kv_heads // target_kv_heads
    num_tokens, num_q_heads, merged_dim = out.shape
    head_dim = merged_dim // group_factor
    eye = torch.eye(group_factor, dtype=out.dtype, device=out.device).view(
        1, 1, group_factor, 1, group_factor, 1
    )
    return (
        (
            out.reshape(
                num_tokens,
                target_kv_heads,
                group_factor,
                num_q_heads // num_kv_heads,
                group_factor,
                head_dim,
            )
            * eye
        )
        .sum(dim=4)
        .reshape(num_tokens, num_q_heads, head_dim)
    )


def split_kv_into_chunks(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_splits: int
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Split ``(q, k, v)`` along the KV-head axis into ``num_splits`` equal chunks."""
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if (
        num_splits < 1
        or num_kv_heads < num_splits
        or num_kv_heads % num_splits != 0
        or num_q_heads % num_splits != 0
    ):
        raise ValueError(
            f"split_kv_into_chunks requires num_kv_heads ({num_kv_heads}) and "
            f"num_q_heads ({num_q_heads}) divisible by num_splits ({num_splits})"
        )
    q_step = num_q_heads // num_splits
    kv_step = num_kv_heads // num_splits
    return [
        (
            q[:, s * q_step : (s + 1) * q_step, :],
            k[:, s * kv_step : (s + 1) * kv_step, :],
            v[:, s * kv_step : (s + 1) * kv_step, :],
        )
        for s in range(num_splits)
    ]


def concat_split_kv_output(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Recombine split-page attention outputs along the query-head axis."""
    return outputs[0] if len(outputs) == 1 else torch.cat(list(outputs), dim=1)


def split_swa_kv_into_halves(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Split SWA ``(q, k, v)`` along the KV-head axis into two equal half-size halves
    (used for Gemma 4 at TP2/TP4 where the SWA page is 2x the Global/Full page size)."""
    first, second = split_kv_into_chunks(q, k, v, 2)
    return first, second


def concat_swa_halves_output(
    out_first: torch.Tensor, out_second: torch.Tensor
) -> torch.Tensor:
    """Recombine the two half-page SWA attention outputs along the query-head axis."""
    return concat_split_kv_output((out_first, out_second))
