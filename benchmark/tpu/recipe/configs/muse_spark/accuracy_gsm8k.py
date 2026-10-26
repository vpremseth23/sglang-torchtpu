"""Muse Spark NVFP4 full GSM8K eval with a 4K output budget."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.accuracy import CONFIG as ACCURACY_CONFIG

CONFIG = deepcopy(ACCURACY_CONFIG)
CONFIG["server_args"].update(context_length=8192, max_total_tokens=65536)
