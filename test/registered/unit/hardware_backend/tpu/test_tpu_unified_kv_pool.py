# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Logical-to-Physical TPU Unified SWA/Full KV cache pool."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.unified_kv_pool import (
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
    concat_split_kv_output,
    merge_swa_kv_heads,
    num_swa_groups,
    resolve_unified_page_layout,
    split_kv_into_chunks,
    split_swa_output,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuUnifiedSWAKVPoolStorage(unittest.TestCase):
    """Tests for PR 1: shared physical buffers, group page mapping, and head merging."""

    def test_equal_full_and_swa_layers_share_buffers(self):
        pool = TpuUnifiedSWAKVPool(
            size=64,
            size_swa=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=4,
            head_dim=128,
            swa_attention_layer_ids=[0, 2],
            full_attention_layer_ids=[1, 3],
            device="cpu",
        )
        self.assertEqual((pool.size, pool.size_swa), (64, 64))
        self.assertEqual((pool.num_swa_groups, pool.num_shared_buffers), (1, 2))
        self.assertFalse(pool.swa_kv_heads_merged)
        self.assertEqual(len(pool.get_state()), 2)
        for full_lid, swa_lid in ((1, 0), (3, 2)):
            self.assertEqual(
                pool.get_fused_kv_buffer(full_lid).data_ptr(),
                pool.get_fused_kv_buffer(swa_lid).data_ptr(),
            )
            self.assertEqual(pool.swa_page_group(swa_lid), 0)
        self.assertEqual(
            pool.get_kv_size_bytes(), pool.full_kv_pool.get_kv_size_bytes()
        )
        self.assertEqual(pool.get_state_buf_infos(), ([], [], []))

    def test_gemma4_ratio_and_head_merging_bf16_and_fp8(self):
        full_ids = [5, 11]
        swa_ids = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                pool = TpuUnifiedSWAKVPool(
                    size=80,
                    size_swa=40,
                    page_size=4,
                    dtype=dtype,
                    head_num=1,
                    head_dim=512,
                    swa_head_num=2,
                    swa_head_dim=256,
                    swa_attention_layer_ids=swa_ids,
                    full_attention_layer_ids=full_ids,
                    device="cpu",
                )
                self.assertTrue(pool.swa_kv_heads_merged)
                self.assertEqual((pool.num_swa_groups, pool.num_shared_buffers), (5, 2))
                self.assertEqual(pool.full_kv_pool.num_blocks, 21)
                self.assertEqual(pool.swa_group_pages.shape, (4, 21))
                for j, lid in enumerate(swa_ids):
                    self.assertEqual(pool.swa_page_group(lid), j // 2)
                    self.assertEqual(
                        pool.get_fused_kv_buffer(lid).data_ptr(),
                        pool.get_state()[j % 2].data_ptr(),
                    )
                new_state = [torch.zeros_like(buf) for buf in pool.get_state()]
                pool.set_state(new_state)
                for j, lid in enumerate(swa_ids):
                    self.assertEqual(
                        pool.get_fused_kv_buffer(lid).data_ptr(),
                        new_state[j % 2].data_ptr(),
                    )

    def test_incompatible_page_shapes_rejected(self):
        for swa_dim in (128, 256):
            with self.assertRaises(ValueError):
                TpuUnifiedSWAKVPool(
                    size=32,
                    page_size=4,
                    dtype=torch.bfloat16,
                    head_num=1,
                    head_dim=512,
                    swa_head_num=3,
                    swa_head_dim=swa_dim,
                    swa_attention_layer_ids=[0, 1],
                    full_attention_layer_ids=[2],
                    device="cpu",
                )

    def test_gcd_num_heads_and_max_head_dim_page_splits(self):
        # 1) Full (4, 128) vs SWA (1, 128): base = (1, 128), full_page_splits=4, swa_page_splits=1
        pool_full_split = TpuUnifiedSWAKVPool(
            size=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=4,
            head_dim=128,
            swa_head_num=1,
            swa_head_dim=128,
            swa_attention_layer_ids=list(range(2, 12)),
            full_attention_layer_ids=[0, 1],
            device="cpu",
        )
        self.assertEqual(
            (
                pool_full_split.base_head_num,
                pool_full_split.base_head_dim,
                pool_full_split.full_page_splits,
                pool_full_split.swa_page_splits,
                pool_full_split.num_shared_buffers,
                pool_full_split.num_swa_groups,
            ),
            (1, 128, 4, 1, 8, 2),
        )
        for s in range(4):
            self.assertEqual(
                pool_full_split.get_fused_kv_buffer(1, split_idx=s).data_ptr(),
                pool_full_split.get_state()[4 + s].data_ptr(),
            )

        # 2) Full (1, 512) vs SWA (4, 256): merge SWA 2x -> (2, 512), then split 2x -> base (1, 512)
        pool_merge_and_split = TpuUnifiedSWAKVPool(
            size=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=512,
            swa_head_num=4,
            swa_head_dim=256,
            swa_attention_layer_ids=[0, 1, 2],
            full_attention_layer_ids=[3, 4],
            device="cpu",
        )
        self.assertTrue(pool_merge_and_split.swa_kv_heads_merged)
        self.assertEqual(
            (
                pool_merge_and_split.base_head_num,
                pool_merge_and_split.base_head_dim,
                pool_merge_and_split.swa_target_kv_heads,
                pool_merge_and_split.full_page_splits,
                pool_merge_and_split.swa_page_splits,
                pool_merge_and_split.num_shared_buffers,
                pool_merge_and_split.num_swa_groups,
            ),
            (1, 512, 2, 1, 2, 2, 3),
        )
        self.assertEqual(pool_merge_and_split.swa_page_split_groups(1), [1, 1])

        # 3) Symmetric Full head merging: Full (2, 256) vs SWA (1, 512) -> Full merges 2->1 head
        for dtype, expect_needs_along in (
            (torch.bfloat16, False),
            (torch.float8_e4m3fn, True),
        ):
            pool_full_merge = TpuUnifiedSWAKVPool(
                size=32,
                page_size=4,
                dtype=dtype,
                head_num=2,
                head_dim=256,
                swa_head_num=1,
                swa_head_dim=512,
                swa_attention_layer_ids=[0, 1],
                full_attention_layer_ids=[2],
                device="cpu",
            )
            self.assertTrue(pool_full_merge.full_kv_heads_merged)
            self.assertFalse(pool_full_merge.swa_kv_heads_merged)
            self.assertEqual(
                (
                    pool_full_merge.base_head_num,
                    pool_full_merge.base_head_dim,
                    pool_full_merge.full_target_kv_heads,
                    pool_full_merge.full_page_splits,
                    pool_full_merge.swa_page_splits,
                    pool_full_merge.needs_seq_along_lane,
                ),
                (1, 512, 1, 1, 1, expect_needs_along),
            )

        # Verify math equivalence when composing merge_swa_kv_heads + split_kv_into_chunks
        torch.manual_seed(7)
        T, Hq, Hkv, D = 5, 8, 4, 32
        q = torch.randn(T, Hq, D)
        k = torch.randn(T, Hkv, D)
        v = torch.randn(T, Hkv, D)
        q_m, k_m, v_m = merge_swa_kv_heads(q, k, v, target_kv_heads=2)
        chunks = split_kv_into_chunks(q_m, k_m, v_m, 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][1].shape, (T, 1, 2 * D))
        recombined = concat_split_kv_output([c[0] for c in chunks])
        torch.testing.assert_close(
            split_swa_output(recombined, Hkv, target_kv_heads=2), q
        )

    def test_merge_swa_kv_heads_and_split_output_exact(self):
        torch.manual_seed(42)
        T, Hq, Hkv, D = 7, 8, 2, 64
        q = torch.randn(T, Hq, D)
        k = torch.randn(T, Hkv, D)
        v = torch.randn(T, Hkv, D)
        scale = D**-0.5

        qg = q.view(T, Hkv, Hq // Hkv, D)
        scores = torch.einsum("thgd,shd->thgs", qg, k) * scale
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        probs = torch.softmax(scores + mask[:, None, None, :], dim=-1)
        out_ref = torch.einsum("thgs,shd->thgd", probs, v).reshape(T, Hq, D)

        q_m, k_m, v_m = merge_swa_kv_heads(q, k, v)
        self.assertEqual(
            (q_m.shape, k_m.shape, v_m.shape), ((T, Hq, 128), (T, 1, 128), (T, 1, 128))
        )
        scores_m = torch.einsum("tqd,skd->tqks", q_m, k_m).squeeze(2) * scale
        probs_m = torch.softmax(scores_m + mask[:, None, :], dim=-1)
        out_m = torch.einsum("tqs,skd->tqd", probs_m, v_m)
        torch.testing.assert_close(
            split_swa_output(out_m, Hkv), out_ref, rtol=1e-5, atol=1e-5
        )
        # Also verify head_num > 1 (e.g. Full head_num=2, dim=512 vs SWA swa_head_num=4, dim=256)
        pool_multi = TpuUnifiedSWAKVPool(
            size=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=512,
            swa_head_num=4,
            swa_head_dim=256,
            swa_attention_layer_ids=[0, 1],
            full_attention_layer_ids=[2],
            device="cpu",
        )
        self.assertTrue(pool_multi.swa_kv_heads_merged)
        self.assertEqual(pool_multi.swa_kv_pool.head_num, 2)
        self.assertEqual(pool_multi.swa_kv_pool.head_dim, 512)

    def test_non_divisible_swa_groups_and_cpu_copy_restore(self):
        self.assertEqual(num_swa_groups(8, 20), 3)
        page_size = 4
        pool = TpuUnifiedSWAKVPool(
            size=48,
            size_swa=48,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            swa_attention_layer_ids=list(range(8, 28)),
            full_attention_layer_ids=list(range(8)),
            device="cpu",
        )
        self.assertEqual((pool.num_shared_buffers, pool.num_swa_groups), (8, 3))
        # SWA scattered pages [3, 9, 11] -> [6, 2, 10] across M_swa = 3 groups.
        pool.swa_group_pages[:, 3] = torch.tensor([9, 11], dtype=torch.int32)
        for j in range(20):
            p = 3 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 3])
            pool.get_state()[j % 8][p].fill_(float(10 + j))
        swa_copy = pool.swa_kv_pool.get_cpu_copy(
            torch.arange(3 * page_size, 4 * page_size)
        )
        pool.swa_group_pages[:, 6] = torch.tensor([2, 10], dtype=torch.int32)
        pool.swa_kv_pool.load_cpu_copy(
            swa_copy, torch.arange(6 * page_size, 7 * page_size)
        )
        for j in range(20):
            p = 6 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 6])
            self.assertTrue(torch.all(pool.get_state()[j % 8][p] == float(10 + j)))

        # Full page 2 -> 5 across all 8 shared buffers directly.
        for i in range(8):
            pool.get_state()[i][2].fill_(float(100 + i))
        full_copy = pool.full_kv_pool.get_cpu_copy(
            torch.arange(2 * page_size, 3 * page_size)
        )
        pool.full_kv_pool.load_cpu_copy(
            full_copy, torch.arange(5 * page_size, 6 * page_size)
        )
        for i in range(8):
            self.assertTrue(torch.all(pool.get_state()[i][5] == float(100 + i)))

        # Verify SEQ_ALONG_LANE saves 50% HBM vs SEQ_ACROSS_LANE at head_num=1 in fp8
        with patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
            return_value="TPU v6e",
        ):
            pool_across = TpuUnifiedSWAKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                head_num=1,
                head_dim=128,
                swa_attention_layer_ids=[1, 2],
                full_attention_layer_ids=[0],
                device="cpu",
                seq_along_lane=False,
            )
            pool_along = TpuUnifiedSWAKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                head_num=1,
                head_dim=128,
                swa_attention_layer_ids=[1, 2],
                full_attention_layer_ids=[0],
                device="cpu",
                seq_along_lane=True,
            )
        self.assertEqual(
            pool_along.full_kv_pool.kv_buffer[0].numel() * 2,
            pool_across.full_kv_pool.kv_buffer[0].numel(),
        )


class TestTpuUnifiedSWAKVPoolAllocator(unittest.TestCase):
    """Tests for PR 2: zero-fragmentation Logical-to-Physical unified allocator."""

    def _make_pool_and_allocator(
        self,
        total_pages=24,
        page_size=4,
        full_ids=(4, 9),
        swa_ids=(0, 1, 2, 3, 5, 6, 7, 8),
    ):
        size = total_pages * page_size
        pool = TpuUnifiedSWAKVPool(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=512,
            swa_head_num=2,
            swa_head_dim=256,
            swa_attention_layer_ids=list(swa_ids),
            full_attention_layer_ids=list(full_ids),
            device="cpu",
        )
        alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        return pool, alloc

    def test_zero_fragmentation_and_companion_pages(self):
        # 2 Full + 8 SWA -> G=2, M_swa=4; 24 physical pages (96 tokens).
        pool, alloc = self._make_pool_and_allocator(total_pages=24, page_size=4)
        self.assertEqual(
            (alloc.full_available_size(), alloc.swa_available_size(), alloc.max_extend_tokens()),
            (96, 24, 16),
        )
        fa, sa = alloc.full_attn_allocator, alloc.swa_attn_allocator
        # Allocate 2 Full pages (2 phys IDs) + 3 SWA pages (12 phys IDs) -> 10 free.
        full_a = fa.alloc(8)
        swa_a = sa.alloc(12)
        self.assertEqual(alloc._free_page_count(), 10)
        swa_primary = (swa_a[::4] // 4).tolist()
        companions = pool.swa_group_pages[:, swa_primary].reshape(-1).tolist()
        all_claimed = (full_a[::4] // 4).tolist() + swa_primary + companions
        self.assertEqual(len(set(all_claimed)), 14)

        # Fragment free list by alternating Full allocations/frees, then allocate 2 SWA pages (8 IDs).
        f1, f2, f3, f4 = (
            fa.alloc(4),
            fa.alloc(4),
            fa.alloc(4),
            fa.alloc(4),
        )
        fa.free(f1)
        fa.free(f3)
        self.assertEqual(alloc._free_page_count(), 8)
        swa_b = sa.alloc(8)
        self.assertIsNotNone(swa_b)
        self.assertEqual(alloc._free_page_count(), 0)

        for t, fn in (
            (full_a, fa.free),
            (f2, fa.free),
            (f4, fa.free),
            (swa_a, sa.free),
            (swa_b, sa.free),
        ):
            fn(t)
        self.assertEqual(alloc._free_page_count(), 24)

    def test_non_divisible_swa_groups_alloc_and_free(self):
        # 8 Full + 20 SWA -> G=8, M_swa=ceil(20/8)=3; 20 physical pages (80 tokens).
        pool, alloc = self._make_pool_and_allocator(
            total_pages=20,
            page_size=4,
            full_ids=range(8),
            swa_ids=range(8, 28),
        )
        self.assertEqual(
            (alloc.full_available_size(), alloc.swa_available_size(), alloc.max_extend_tokens()),
            (80, 24, 20),
        )
        full_tok = alloc.full_attn_allocator.alloc(12)  # 3 * 1 = 3 phys IDs
        swa_tok = alloc.swa_attn_allocator.alloc(8)  # 2 * 3 = 6 phys IDs
        self.assertEqual(alloc._free_page_count(), 11)
        f_p = (full_tok[::4] // 4).tolist()
        s_p = (swa_tok[::4] // 4).tolist()
        claimed = (
            f_p
            + s_p
            + pool.swa_group_pages[:, s_p].reshape(-1).tolist()
        )
        self.assertEqual(len(set(claimed)), 9)
        alloc.full_attn_allocator.free(full_tok)
        alloc.swa_attn_allocator.free(swa_tok)
        self.assertEqual(alloc._free_page_count(), 20)

    def test_alloc_extend_decode_and_eviction(self):
        _, alloc = self._make_pool_and_allocator(total_pages=20, page_size=4)
        # 1+4=5 phys IDs per joint page -> 4 joint pages max (16 tokens).
        out = alloc.alloc_extend(
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([8]),
            seq_lens_cpu=torch.tensor([8]),
            last_loc=torch.tensor([-1]),
            extend_num_tokens=8,
        )
        self.assertIsNotNone(out)
        self.assertEqual(alloc._free_page_count(), 10)
        dec = alloc.alloc_decode(
            seq_lens=torch.tensor([9]),
            seq_lens_cpu=torch.tensor([9]),
            last_loc=out[-1:],
        )
        self.assertIsNotNone(dec)
        self.assertEqual(alloc._free_page_count(), 5)

        # Request 2 more joint pages (10 phys IDs): shortfall = 5 phys IDs.
        self.assertEqual(
            alloc.reclaim_plan(8, 8, full_evictable_tokens=8, swa_evictable_tokens=8),
            (8, 4),
        )
        self.assertIsNone(
            alloc.reclaim_plan(8, 8, full_evictable_tokens=4, swa_evictable_tokens=0)
        )

        class _TreeCache:
            def supports_prefix_sharing(self):
                return True

            def full_evictable_size(self):
                return 8

            def swa_evictable_size(self):
                return 8

            def evict(self, params: EvictParams):
                if params.swa_num_tokens > 0:
                    alloc.free_swa(out[: params.swa_num_tokens])
                if params.num_tokens > 0:
                    alloc.free_full(out[: params.num_tokens])

        self.assertIsNone(alloc.reclaim_for_prealloc(_TreeCache(), 8, 8))
        self.assertGreaterEqual(alloc._free_page_count(), 10)
        self.assertTrue(alloc.prealloc_fits_assumes_reclaim)
        self.assertTrue(alloc.prealloc_ceiling_fits(8, 8))
        self.assertTrue(alloc.prealloc_fits(8, 8))


if __name__ == "__main__":
    unittest.main()



