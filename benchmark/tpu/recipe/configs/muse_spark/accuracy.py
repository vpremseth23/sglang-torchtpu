"""Muse Spark NVFP4 backbone accuracy, without speculative or cache features."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

CONCURRENCY = 8
MODEL_PATH = "/weights/Muse-Spark-Open-816B-A42B-NVFP4"

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    model_path=MODEL_PATH,
    served_model_name="meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    disable_radix_cache=True,
    context_length=69632,
    max_total_tokens=557056,
    max_running_requests=CONCURRENCY,
    torch_compile_max_bs=CONCURRENCY,
    tpu_compile_bs=[1, 2, 4, CONCURRENCY],
    tpu_compile_tokens=[4096],
)
