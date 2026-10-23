# SPDX-License-Identifier: Apache-2.0
"""Logical-to-Physical Unified KV cache pool for hybrid Full + SWA models on TPU.

Shares ``G = num_full_layers`` physical 5D Pallas KV buffers across Full-Attention
(1 physical page per logical page) and Sliding-Window Attention
(``M_swa = ceil(num_swa_layers / num_full_layers)`` groups) with zero external
fragmentation. Physical page 0 is reserved as the dummy-write sink for padded tokens.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang_tpu.kv_pool import (
    GB,
    TpuFusedMHATokenToKVPool,
    TpuSeqAlongLaneKVPool,
    _cdiv,
    _seq_along_lane_head_dim,
    _torch_dtype_packing,
)
from sglang_tpu.unified_kv_heads import (
    concat_split_kv_output,
    concat_swa_halves_output,
    merge_swa_kv_heads,
    split_kv_into_chunks,
    split_swa_kv_into_halves,
    split_swa_output,
)

logger = logging.getLogger(__name__)


def num_swa_groups(num_full_layers: int, num_swa_layers: int) -> int:
    """Return number of SWA groups ``M_swa = ceil(num_swa_layers / num_full_layers)``."""
    if num_full_layers <= 0 or num_swa_layers <= 0:
        raise ValueError(
            f"Unified KV pool requires positive full ({num_full_layers}) and "
            f"SWA ({num_swa_layers}) layer counts."
        )
    return _cdiv(num_swa_layers, num_full_layers)


def _fused_page_shape(
    page_size: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    store_dtype: torch.dtype,
    *,
    seq_along_lane: bool = False,
) -> Tuple[int, int, int, int]:
    """Return per-page 4D shape for ``SEQ_ACROSS_LANE`` (standard fused) or ``SEQ_ALONG_LANE``."""
    if v_head_dim > head_dim:
        raise NotImplementedError(
            "TpuUnifiedSWAKVPool requires v_head_dim <= head_dim "
            f"(got v_head_dim={v_head_dim} > head_dim={head_dim})."
        )
    kv_packing = _torch_dtype_packing(store_dtype)
    if seq_along_lane:
        try:
            hd_lanes = _seq_along_lane_head_dim(
                head_dim=head_dim, kv_packing=kv_packing
            )
        except ValueError:
            align = TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT
            hd_lanes = _cdiv(head_dim, align) * align
        return (head_num * 2, hd_lanes // kv_packing, kv_packing, page_size)
    align = TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT
    padded_head_dim = _cdiv(head_dim, align) * align
    kv_heads_packed = _cdiv(head_num * 2, kv_packing)
    return (page_size, kv_heads_packed, kv_packing, padded_head_dim)


class UnifiedPageLayout(NamedTuple):
    """Resolved physical page layout after ``max(head_dim)`` merging and ``gcd(num_heads)`` splitting."""

    page_shape: Tuple[int, int, int, int]
    base_head_num: int
    base_head_dim: int
    base_v_head_dim: int
    full_merge_factor: int
    swa_merge_factor: int
    full_target_kv_heads: int
    swa_target_kv_heads: int
    full_page_splits: int
    swa_page_splits: int

    @property
    def full_kv_heads_merged(self) -> bool:
        return self.full_merge_factor > 1

    @property
    def swa_kv_heads_merged(self) -> bool:
        return self.swa_merge_factor > 1

    def needs_seq_along_lane(self, store_dtype: torch.dtype) -> bool:
        """Return True if ``SEQ_ACROSS_LANE`` pads ``(base_head_num, base_head_dim)``
        (e.g. odd ``base_head_num`` in FP8 after head merging or ``gcd`` splitting)."""
        kv_packing = _torch_dtype_packing(store_dtype)
        return (
            (2 * self.base_head_num) % kv_packing != 0
            or self.base_head_dim % TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT != 0
        )


def resolve_unified_page_layout(
    *,
    page_size: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    swa_head_num: int,
    swa_head_dim: int,
    swa_v_head_dim: int,
    store_dtype: torch.dtype,
    seq_along_lane: bool = False,
) -> UnifiedPageLayout:
    """Resolve Full/SWA geometry into a single physical page shape via two steps:
    1. Equalize ``head_dim`` to ``D_base = max(head_dim, swa_head_dim)`` by merging KV heads.
    2. Equalize effective ``num_heads`` to ``H_base = gcd(H'_full, H'_swa)`` by splitting pages.
    """
    full_shape = _fused_page_shape(
        page_size,
        head_num,
        head_dim,
        v_head_dim,
        store_dtype,
        seq_along_lane=seq_along_lane,
    )
    swa_shape = _fused_page_shape(
        page_size,
        swa_head_num,
        swa_head_dim,
        swa_v_head_dim,
        store_dtype,
        seq_along_lane=seq_along_lane,
    )
    if full_shape == swa_shape and head_num == swa_head_num:
        base_dim = max(head_dim, swa_head_dim)
        return UnifiedPageLayout(
            page_shape=full_shape,
            base_head_num=head_num,
            base_head_dim=base_dim,
            base_v_head_dim=max(v_head_dim, swa_v_head_dim),
            full_merge_factor=1,
            swa_merge_factor=1,
            full_target_kv_heads=head_num,
            swa_target_kv_heads=swa_head_num,
            full_page_splits=1,
            swa_page_splits=1,
        )

    base_dim = max(head_dim, swa_head_dim)
    if base_dim % head_dim == 0 and base_dim % swa_head_dim == 0:
        g_full = base_dim // head_dim
        g_swa = base_dim // swa_head_dim
        full_ok = g_full == 1 or (v_head_dim == head_dim and head_num % g_full == 0)
        swa_ok = g_swa == 1 or (
            swa_v_head_dim == swa_head_dim and swa_head_num % g_swa == 0
        )
        if full_ok and swa_ok:
            h_full_eff = head_num // g_full
            h_swa_eff = swa_head_num // g_swa
            v_full_eff = v_head_dim * g_full
            v_swa_eff = swa_v_head_dim * g_swa
            base_heads = math.gcd(h_full_eff, h_swa_eff)
            base_full_shape = _fused_page_shape(
                page_size,
                base_heads,
                base_dim,
                v_full_eff,
                store_dtype,
                seq_along_lane=seq_along_lane,
            )
            base_swa_shape = _fused_page_shape(
                page_size,
                base_heads,
                base_dim,
                v_swa_eff,
                store_dtype,
                seq_along_lane=seq_along_lane,
            )
            if base_full_shape == base_swa_shape:
                return UnifiedPageLayout(
                    page_shape=base_full_shape,
                    base_head_num=base_heads,
                    base_head_dim=base_dim,
                    base_v_head_dim=max(v_full_eff, v_swa_eff),
                    full_merge_factor=g_full,
                    swa_merge_factor=g_swa,
                    full_target_kv_heads=h_full_eff,
                    swa_target_kv_heads=h_swa_eff,
                    full_page_splits=h_full_eff // base_heads,
                    swa_page_splits=h_swa_eff // base_heads,
                )
    raise ValueError(
        "TpuUnifiedSWAKVPool requires Full and SWA layers to share an identical "
        f"per-page fused shape (directly, after merging KV heads to max(head_dim), "
        f"or after splitting by gcd(num_heads)), "
        f"but got full={full_shape} (heads={head_num}, dim={head_dim}) vs "
        f"swa={swa_shape} (heads={swa_head_num}, dim={swa_head_dim})."
    )


def resolve_unified_page_shape(
    *,
    page_size: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    swa_head_num: int,
    swa_head_dim: int,
    swa_v_head_dim: int,
    store_dtype: torch.dtype,
    seq_along_lane: bool = False,
) -> Tuple[Tuple[int, int, int, int], bool]:
    """Validate Full/SWA page compatibility and return ``(page_shape, swa_kv_heads_merged)``."""
    layout = resolve_unified_page_layout(
        page_size=page_size,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        swa_head_num=swa_head_num,
        swa_head_dim=swa_head_dim,
        swa_v_head_dim=swa_v_head_dim,
        store_dtype=store_dtype,
        seq_along_lane=seq_along_lane,
    )
    return layout.page_shape, layout.swa_kv_heads_merged


class _SharedBufferView(TpuFusedMHATokenToKVPool):
    """``TpuFusedMHATokenToKVPool`` that either owns the shared buffers or views them."""

    PACKS_HEADS = False

    def __init__(
        self,
        *,
        shared_buffers: Optional[List[torch.Tensor]] = None,
        layer_groups: Optional[List[int]] = None,
        group_pages: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._shared_buffers = None if shared_buffers is None else list(shared_buffers)
        if self._shared_buffers is not None:
            kwargs["layer_num"] = len(self._shared_buffers)
        self.layer_groups = (
            list(layer_groups)
            if layer_groups is not None
            else [0] * kwargs["layer_num"]
        )
        self.group_pages = group_pages
        self._has_group_offsets = group_pages is not None and any(
            g > 0 for g in self.layer_groups
        )
        super().__init__(**kwargs)

    def _create_buffers(self):
        if self._shared_buffers is None:
            super()._create_buffers()
            return
        self.kv_buffer = self._shared_buffers
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [int(np.prod(x.shape[1:])) * x.dtype.itemsize for x in self.kv_buffer],
            device=self.device,
        )

    def _finalize_allocation_log(self, size: int) -> None:
        if self._shared_buffers is None:
            super()._finalize_allocation_log(size)

    def _raiden_backup(self):
        return None if self._has_group_offsets else super()._raiden_backup()

    def _map_group_token_indices(
        self, group: int, indices_cpu: torch.Tensor
    ) -> torch.Tensor:
        if group == 0 or self.group_pages is None:
            return indices_cpu
        phys_pages = self.group_pages[group - 1, indices_cpu // self.page_size].to(
            torch.long
        )
        return phys_pages * self.page_size + (indices_cpu % self.page_size)

    @contextlib.contextmanager
    def _group_buffers(self, buffers: List[torch.Tensor]):
        orig_buffers, orig_num = self.kv_buffer, self.layer_num
        self.kv_buffer, self.layer_num = buffers, len(buffers)
        try:
            yield
        finally:
            self.kv_buffer, self.layer_num = orig_buffers, orig_num

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        if not self._has_group_offsets:
            return super().get_cpu_copy(indices, mamba_indices, req_pool_index)
        indices_cpu = indices.to(device="cpu", dtype=torch.long)
        out: list = [None] * self.layer_num
        for g in range(max(self.layer_groups) + 1):
            lids = [i for i, grp in enumerate(self.layer_groups) if grp == g]
            with self._group_buffers([self.kv_buffer[i] for i in lids]):
                grp_out = super().get_cpu_copy(
                    self._map_group_token_indices(g, indices_cpu),
                    mamba_indices,
                    req_pool_index,
                )
            for i, lid in enumerate(lids):
                out[lid] = grp_out[i]
        return out

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        if not self._has_group_offsets:
            return super().load_cpu_copy(
                kv_cache_cpu, indices, mamba_indices, req_pool_index
            )
        indices_cpu = indices.to(device="cpu", dtype=torch.long)
        for g in range(max(self.layer_groups) + 1):
            lids = [i for i, grp in enumerate(self.layer_groups) if grp == g]
            with self._group_buffers([self.kv_buffer[i] for i in lids]):
                super().load_cpu_copy(
                    [kv_cache_cpu[i] for i in lids],
                    self._map_group_token_indices(g, indices_cpu),
                    mamba_indices,
                    req_pool_index,
                )


class _SeqAlongLaneSharedBufferView(_SharedBufferView, TpuSeqAlongLaneKVPool):
    """``TpuSeqAlongLaneKVPool`` that either owns the shared buffers or views them."""


class TpuUnifiedSWAKVPool(SWAKVPool):
    """Unified Hybrid Full/SWA KV pool backed by ``G = num_full_layers`` shared 5D buffers."""

    def __init__(
        self,
        size: int,
        size_swa: Optional[int] = None,
        *,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        device: str,
        full_kv_pool_class: Optional[type] = None,
        swa_kv_pool_class: Optional[type] = None,
        full_kv_pool_kwargs: Optional[dict] = None,
        swa_kv_pool_kwargs: Optional[dict] = None,
        start_layer: int = 0,
        enable_memory_saver: bool = False,
        **kwargs,
    ):
        del swa_kv_pool_class
        del full_kv_pool_kwargs, swa_kv_pool_kwargs
        self.seq_along_lane = bool(
            kwargs.pop("seq_along_lane", False)
            or full_kv_pool_class is TpuSeqAlongLaneKVPool
        )
        self.size = size
        self.size_swa = size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = start_layer
        self.page_size = page_size
        self.layer_transfer_counter = None
        self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = None
        self.index_head_dim = None
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        v_head_dim = kwargs.get("v_head_dim") or head_dim
        swa_head_num = kwargs.get("swa_head_num") or head_num
        swa_head_dim = kwargs.get("swa_head_dim") or head_dim
        swa_v_head_dim = kwargs.get("swa_v_head_dim") or swa_head_dim
        layout = resolve_unified_page_layout(
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            store_dtype=dtype,
            seq_along_lane=self.seq_along_lane,
        )
        self.full_raw_head_num = head_num
        self.full_raw_head_dim = head_dim
        self.swa_raw_head_num = swa_head_num
        self.swa_raw_head_dim = swa_head_dim
        self.base_head_num = layout.base_head_num
        self.base_head_dim = layout.base_head_dim
        self.base_v_head_dim = layout.base_v_head_dim
        self.full_kv_heads_merged = layout.full_kv_heads_merged
        self.swa_kv_heads_merged = layout.swa_kv_heads_merged
        self.full_target_kv_heads = layout.full_target_kv_heads
        self.swa_target_kv_heads = layout.swa_target_kv_heads
        self.full_page_splits = layout.full_page_splits
        self.swa_page_splits = layout.swa_page_splits
        self.needs_seq_along_lane = layout.needs_seq_along_lane(dtype)

        G = self.full_layer_nums * self.full_page_splits
        self.num_shared_buffers = G
        effective_swa_layers = self.swa_layer_nums * self.swa_page_splits
        self.num_swa_groups = num_swa_groups(G, effective_swa_layers)
        num_blocks = size // page_size + 1
        self.swa_group_pages = torch.zeros(
            (max(0, self.num_swa_groups - 1), num_blocks), dtype=torch.int32
        )

        common = dict(
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        view_cls = (
            _SeqAlongLaneSharedBufferView if self.seq_along_lane else _SharedBufferView
        )
        self.full_kv_pool = view_cls(
            head_num=layout.base_head_num,
            head_dim=layout.base_head_dim,
            v_head_dim=layout.base_v_head_dim,
            layer_num=G,
            allocation_label="Full",
            **common,
        )
        bufs = self.full_kv_pool.kv_buffer
        self.swa_kv_pool = view_cls(
            shared_buffers=[bufs[j % G] for j in range(effective_swa_layers)],
            head_num=layout.base_head_num,
            head_dim=layout.base_head_dim,
            v_head_dim=layout.base_v_head_dim,
            allocation_label="SWA",
            layer_groups=[j // G for j in range(effective_swa_layers)],
            group_pages=self.swa_group_pages if self.num_swa_groups > 1 else None,
            **common,
        )

        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        self._swa_layer_group: Dict[int, int] = {}
        self._swa_layer_split_groups: Dict[int, List[int]] = {}
        for i, gid in enumerate(full_attention_layer_ids):
            self.layers_mapping[gid] = (i * self.full_page_splits, False)
        for j, gid in enumerate(swa_attention_layer_ids):
            base_idx = j * self.swa_page_splits
            self.layers_mapping[gid] = (base_idx, True)
            self._swa_layer_group[gid] = base_idx // G
            self._swa_layer_split_groups[gid] = [
                (base_idx + s) // G for s in range(self.swa_page_splits)
            ]

        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            "TpuUnifiedSWAKVPool mem usage: %.2f GB, unified size: %d tokens "
            "(shared buffers=%d, swa_groups=%d, swa_heads_merged=%s)",
            self.mem_usage,
            self.size,
            G,
            self.num_swa_groups,
            self.swa_kv_heads_merged,
        )

    def get_fused_kv_buffer(self, layer_id: int, split_idx: int = 0) -> torch.Tensor:
        idx, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_fused_kv_buffer(idx + split_idx)

    def swa_page_group(self, layer_id: int) -> int:
        return self._swa_layer_group.get(layer_id, 0)

    def swa_page_split_groups(self, layer_id: int) -> List[int]:
        """Return all SWA group indices used by ``layer_id`` (length 1 normally, or 2 when split)."""
        return self._swa_layer_split_groups.get(
            layer_id, [self.swa_page_group(layer_id)]
        )

    def get_state(self) -> list[torch.Tensor]:
        return self.full_kv_pool.get_state()

    def set_state(self, state: list[torch.Tensor]) -> None:
        G = self.num_shared_buffers
        effective_swa_layers = self.swa_layer_nums * self.swa_page_splits
        self.full_kv_pool.set_state(state)
        self.swa_kv_pool.set_state(
            [state[j % G] for j in range(effective_swa_layers)]
        )

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return self.full_kv_pool.get_kv_size_bytes()

    def get_state_buf_infos(self):
        return [], [], []


