"""Muse Spark two-host-per-engine DFlash P/D, prefix cache and mxint8 KV offload."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.testing import muse_spark_pd

register_tpu_ci(
    est_time=10800,
    suite="tpu-integration",
    nightly=True,
    labels=[
        "multihost",
        "muse-spark-open-816b-a42b",
        "pd",
        "spec",
        "dflash",
        "prefix-cache",
        "kv-offload",
        "mxint8",
    ],
)


class TestMuseSparkPDDflashMxint8(muse_spark_pd.MuseSparkDflashPDTestCase):
    """Require actual speculative decoding and a complete host-cache reload."""

    config_module = "recipe/configs/muse_spark/pd_multihost_prefix_cache_kv_offload_dflash_mxint8.py"
    kv_cache_dtype = "mxint8"
    tag = "muse-spark-pd-dflash_mxint8"

    def test_prefix_cache_kv_offload_dflash(self):
        self.run_feature_gates()


if __name__ == "__main__":
    unittest.main()
