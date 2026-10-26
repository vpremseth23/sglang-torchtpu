"""Muse Spark NVFP4 TPU quality on all 1,730 MMMU-Pro standard-10 questions."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu import MUSE_SPARK, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase
from sglang_tpu.testing import muse_spark_mmmu

register_tpu_ci(
    # Manual unsharded reference. The nightly partitions the same 1,730 rows
    # into four CI-sized shards and checks their combined coverage and score.
    est_time=28800,
    suite="tpu-integration",
    nightly=False,
    labels=["singlehost", "muse-spark-open-816b-a42b", "accuracy", "multimodal"],
)

CONFIG_MODULE = "recipe/configs/muse_spark/accuracy_mmmupro.py"

GATE = {
    "dataset": "mmmu_pro",
    "label": "mmmu_pro",
    "baseline": muse_spark_mmmu.AGGREGATE["baseline"],
    "atol": muse_spark_mmmu.AGGREGATE["atol"],
    "runner": make_runner(MUSE_SPARK, "mmmu_pro"),
}


def build_model_config():
    return make_model_config(
        MUSE_SPARK,
        test_name="muse-spark-mmmupro",
        series_id=muse_spark_mmmu.AGGREGATE["id"],
        gate=GATE,
    )


class TestMuseSparkMMMUPro(AccuracyTestCase, CustomTestCase):
    """Score images and text through the multimodal recipe's server."""

    config_module = CONFIG_MODULE
    model_path = "/weights/Muse-Spark-Open-816B-A42B-NVFP4"
    tag = "muse-spark-mmmupro-e2e"
    health_timeout = 5400

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


if __name__ == "__main__":
    unittest.main()
