"""
Helper functions for GPU allocation in Snakemake workflows.

Configuration via Snakemake config:
    gpu_allocation:
      enable_dynamic: true
      cluster_cmd: "module load admin && ClusterStateJSON"

Or via environment variable:
    export DISABLE_DYNAMIC_GPU=1
"""

import yaml
from pathlib import Path
from gpu_allocator import get_optimal_cpu as _get_optimal_cpu
from snakemake.logging import logger

# Load config once at module level
def _load_gpu_config():
    for cand in [
        Path(__file__).resolve().parent.parent.parent / "config" / "gpu_config.yaml",
        Path(__file__).resolve().parent.parent / "config" / "gpu_config.yaml",
        Path.cwd() / "config" / "gpu_config.yaml",
    ]:
        if cand.exists():
            with open(cand) as f:
                cfg = yaml.safe_load(f)
                return cfg.get("gpu_requirements", {}), cfg.get("gpu_allocation", {})
    logger.warning("gpu_helpers: gpu_config.yaml not found")
    return {}, {}

GPU_REQUIREMENTS, GPU_ALLOCATION_CONFIG = _load_gpu_config()


def _is_enabled(val):
    """Convert various representations to boolean."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("yes", "y", "true", "1")
    return val in (1, "1", True)


def get_gpu_for_rule_name(rule_name, snakemake_config=None, **override_kwargs):
    """Get GPU allocation for a named rule using config-based requirements."""
    requirements = GPU_REQUIREMENTS.get(rule_name, GPU_REQUIREMENTS.get("default", {}))

    # Build allocation config, merging from multiple sources
    alloc_config = dict(GPU_ALLOCATION_CONFIG)
    if snakemake_config and "gpu_allocation" in snakemake_config:
        alloc_config.update(snakemake_config["gpu_allocation"])

    enable_dynamic = _is_enabled(alloc_config.get("enable_dynamic", True))

    if not enable_dynamic:
        fallback = override_kwargs.get("fallback_gpu") or requirements.get("fallback_gpu") or "gpu:1"
        return fallback

    # Merge configs: allocator config, requirements, then overrides
    params = {**alloc_config, **requirements, **override_kwargs}
    params["enable_dynamic"] = True
    if "fallback_gpu" not in params:
        params["fallback_gpu"] = "__GENERIC_GPU1__"

    logger.debug(f"gpu_helpers: rule={rule_name}, params={params}")
    return _get_optimal_cpu(**params)


def get_dynamic_gpu(**kwargs):
    """Alias for direct GPU allocation without rule name."""
    return _get_optimal_cpu(**kwargs)


__all__ = ["get_gpu_for_rule_name", "get_dynamic_gpu"]
