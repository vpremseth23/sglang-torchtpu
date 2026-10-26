"""Muse MMMU-Pro shard 0/4, preserving the complete standard-10 dataset."""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.testing import muse_spark_mmmu

register_tpu_ci(
    est_time=9000,
    suite="nightly-tpu-accuracy-muse-spark-mmmupro-shard0",
    nightly=True,
    labels=["singlehost", "muse-spark-open-816b-a42b", "accuracy", "multimodal"],
)


def build_model_config():
    return muse_spark_mmmu.build_model_config(0)


class TestMuseSparkMMMUShard0(muse_spark_mmmu.MuseSparkMMMUShardTestCase):
    shard = 0
    tag = "muse-spark-mmmupro-shard0"


if __name__ == "__main__":
    # One pass gives an interpretable archive for a shard quality failure.
    with envs.SGLANG_TEST_MAX_RETRY.override(0):
        unittest.main()
