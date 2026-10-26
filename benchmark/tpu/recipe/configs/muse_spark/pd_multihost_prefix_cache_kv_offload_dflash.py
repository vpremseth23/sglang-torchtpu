"""Muse Spark DFlash P/D: two hosts per engine, with prefill prefix offload.

Every pod launches its slice-local role. The test starts the router only on the
prefill leader, so the launcher must retain the two independent TPU topologies.
The bounded device pool makes the cache gate force an actual host reload.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

HOSTS_PER_ENGINE = 2
DEVICE_POOL_TOKENS = 8192
HICACHE_RATIO = 3


def role_config(role: str, *, kv_cache_dtype: str = "bfloat16") -> dict:
    """Build matching target/draft handoff layouts for independent TPU slices."""
    if role not in ("prefill", "decode"):
        raise ValueError(f"Expected prefill or decode, got {role!r}")
    config = deepcopy(SERVER_CONFIG)
    config["server_args"].update(
        model_path=os.environ.get("MUSE_MODEL")
        or "/weights/Muse-Spark-Open-816B-A42B-NVFP4",
        served_model_name="meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
        host="0.0.0.0",
        port=int(os.environ.get(f"{role.upper()}_PORT", "30000")),
        nnodes=HOSTS_PER_ENGINE,
        tp_size=8 * HOSTS_PER_ENGINE,
        ep_size=1,
        page_size=128,
        kv_cache_dtype=kv_cache_dtype,
        context_length=8192,
        max_total_tokens=DEVICE_POOL_TOKENS,
        max_running_requests=8,
        torch_compile_max_bs=8,
        tpu_compile_bs=[1, 2, 4, 8],
        tpu_compile_tokens=[512, 2048, 4096],
        mem_fraction_static=0.85,
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=os.environ.get("MUSE_DRAFT_MODEL")
        or "/weights/Muse-Spark-Open-DFlash2",
        speculative_num_draft_tokens=8,
        speculative_draft_kv_cache_dtype="bfloat16",
        disaggregation_mode=role,
        disaggregation_transfer_backend="raiden",
        enable_cache_report=True,
        enable_metrics=True,
        skip_server_warmup=True,
        sampling_backend="pytorch",
    )
    if role == "prefill":
        config["server_args"].update(
            disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
            enable_hierarchical_cache=True,
            hicache_io_backend="raiden",
            hicache_mem_layout="page_first_fused",
            hicache_write_policy="write_through",
            hicache_ratio=HICACHE_RATIO,
        )
    config["runtime_env"].update(
        FORCE_SINGLE_HOST=None,
        MODEL=None,
        HOST=None,
        PORT=None,
        SGLANG_RAIDEN_RESHARD="0",
        SGLANG_RAIDEN_KV_PORT="31000",
        SGLANG_SPEC_TP_SYNC="rng",
        SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
        SGLANG_SIMULATE_ACC_LEN=None,
    )
    return config


CONFIG = role_config(os.environ.get("CI_PD_ROLE", "prefill"))
