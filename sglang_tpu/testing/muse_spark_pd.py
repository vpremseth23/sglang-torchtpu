"""Muse Spark DFlash P/D gates shared by the BF16 and MXINT8 KV variants."""

import requests

from sglang.test.tpu import MUSE_SPARK, run_model_config
from sglang.test.tpu.utils.kv_offload_probe import kv_offload_runner
from sglang_tpu.testing.pd_multihost_base import PDMultiHostTestCase


class MuseSparkDflashPDTestCase(PDMultiHostTestCase):
    """Check the serving contract, actual speculation, and a host KV round trip."""

    kv_cache_dtype = "bfloat16"

    @classmethod
    def check_server_configuration(cls):
        expected_common = {
            "nnodes": 2,
            "tp_size": 16,
            "ep_size": 1,
            "page_size": 128,
            "kv_cache_dtype": cls.kv_cache_dtype,
            "speculative_algorithm": "DFLASH",
            "speculative_draft_kv_cache_dtype": "bfloat16",
            "disaggregation_transfer_backend": "raiden",
            "disable_radix_cache": False,
        }
        for role, url in (("prefill", cls.prefill_url), ("decode", cls.decode_url)):
            response = requests.get(url + "/server_info", timeout=30)
            response.raise_for_status()
            expected = {**expected_common, "disaggregation_mode": role}
            if role == "prefill":
                expected.update(
                    enable_hierarchical_cache=True,
                    hicache_io_backend="raiden",
                    hicache_mem_layout="page_first_fused",
                    hicache_write_policy="write_through",
                    hicache_ratio=3,
                    max_total_tokens=8192,
                )
            info = response.json()
            for key, value in expected.items():
                if info.get(key) != value:
                    raise AssertionError(
                        f"{role} {key}: expected {value!r}, got {info.get(key)!r}"
                    )

    def run_feature_gates(self):
        if not self.is_test_leader:
            self.wait_for_worker_shutdown()
            return
        test_name = "muse-spark-pd-multihost-dflash-" + self.kv_cache_dtype
        run_model_config(
            {
                "model": MUSE_SPARK["served_model_name"],
                "test_name": test_name,
                "variant": "PD-TP16+EP1-DFLASH-" + self.kv_cache_dtype,
                "gates": [
                    {
                        "dataset": "gsm8k",
                        "baseline": 0.98,
                        "atol": 0.10,
                        "num_examples": 200,
                        "num_threads": 8,
                        "max_tokens": 4096,
                        "temperature": 0.0,
                        "reasoning_effort": "low",
                        # A value above 1 proves accepted draft tokens, beyond
                        # merely launching the experimental spec worker.
                        "spec_accept_length_threshold": 1.0,
                    },
                    {
                        "dataset": "kv_offload",
                        "label": "kv_offload_dflash",
                        "runner": kv_offload_runner(
                            test_name=test_name,
                            device_pool_tokens=8192,
                            page_size=128,
                            probe_tokens=2048,
                            hicache_ratio=3,
                            spec_decoding=True,
                        ),
                    },
                ],
            },
            base_url=self.base_url,
        )
