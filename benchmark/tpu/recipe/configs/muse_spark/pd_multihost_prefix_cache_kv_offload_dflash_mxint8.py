"""Muse Spark multihost DFlash P/D with MXINT8 target KV and BF16 draft KV.

Requires a compatible MXINT8 backend extension. There is no dtype fallback:
running this gate on a BF16-only image must fail during server startup.
"""

import os

from benchmark.tpu.recipe.configs.muse_spark.pd_multihost_prefix_cache_kv_offload_dflash import (
    role_config,
)

CONFIG = role_config(os.environ.get("CI_PD_ROLE", "prefill"), kv_cache_dtype="mxint8")
