"""Muse Spark full MMMU-Pro evaluation with the vision tower enabled."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.accuracy import CONFIG as ACCURACY_CONFIG

CONFIG = deepcopy(ACCURACY_CONFIG)
CONFIG["server_args"].update(enable_multimodal=True)
CONFIG["runtime_env"].update(
    # The tokenizer resolves the model before scheduler init_backend registers
    # TPU architectures. Importing this processor registers its native model too.
    SGLANG_EXTERNAL_MM_MODEL_ARCH="MuseSparkForConditionalGeneration",
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE="sglang_tpu.multimodal",
)
