"""Full Muse MMMU-Pro, split into four disjoint CI-sized benchmark shards."""

from functools import partial

from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu import MUSE_SPARK, make_model_config
from sglang.test.tpu.evals.mmmu_full import run_shard
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase

SHARDS = 4
AGGREGATE = {
    "id": "accuracy-muse-spark-mmmupro",
    "display_name": MUSE_SPARK["display_name"],
    "model": MUSE_SPARK["served_model_name"],
    # Initial reference, pending full TPU calibration.
    "baseline": 0.75,
    "atol": 0.10,
}


def build_model_config(shard):
    if not 0 <= shard < SHARDS:
        raise ValueError(f"Invalid Muse MMMU-Pro shard {shard}")
    config = make_model_config(
        MUSE_SPARK,
        test_name=f"muse-spark-mmmupro-shard{shard}",
        series_id=f"accuracy-muse-spark-mmmupro-shard{shard}",
        gate={
            "dataset": "mmmu_pro",
            "label": f"mmmu_pro_shard{shard}",
            "baseline": AGGREGATE["baseline"],
            "atol": AGGREGATE["atol"],
            "runner": partial(
                run_shard,
                shard=shard,
                shards=SHARDS,
                model=MUSE_SPARK["served_model_name"],
                aggregate=AGGREGATE,
                **MUSE_SPARK["sampling"],
            ),
        },
    )
    config["display_name"] += f" MMMU-Pro shard {shard}/{SHARDS}"
    return config


class MuseSparkMMMUShardTestCase(AccuracyTestCase, CustomTestCase):
    config_module = "recipe/configs/muse_spark/accuracy_mmmupro.py"
    model_path = "/weights/Muse-Spark-Open-816B-A42B-NVFP4"
    health_timeout = 5400
    shard = None

    @classmethod
    def build_model_config(cls):
        return build_model_config(cls.shard)

    def test_accuracy(self):
        self.run_evaluations()
