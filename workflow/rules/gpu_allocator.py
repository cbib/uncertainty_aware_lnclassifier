#!/usr/bin/env python3
"""
Dynamic GPU allocation for Snakemake rules based on VRAM requirements.
Queries cluster state and selects appropriate GPU based on availability.

Can be disabled via:
- Environment variable: DISABLE_DYNAMIC_GPU=1
- Config parameter: enable_dynamic_gpu=False
"""

import json
import logging
import os
import subprocess
from typing import Dict, List, Optional

try:
    from snakemake.logging import logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)


class GPUAllocator:
    """Manages dynamic GPU allocation based on cluster availability and VRAM requirements."""

    # GPU name to VRAM mapping (in GB)
    GPU_VRAM_MAP = {
        "nvidia_h100_nvl": 94,
        "nvidia_h100_nvl_1g.12gb": 12,
        "nvidia_h100_nvl_1g.24gb": 24,
        "nvidia_h100_nvl_4g.47gb": 47,
    }

    def __init__(
        self,
        cluster_cmd: str = "module load admin && ClusterStateJSON",
        enable_dynamic: bool = True,
    ):
        self.cluster_cmd = cluster_cmd
        self.cluster_state = None
        # Check environment variable to disable dynamic allocation
        env_disable = os.environ.get("DISABLE_DYNAMIC_GPU", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.enable_dynamic = enable_dynamic and not env_disable

        if not self.enable_dynamic:
            logger.debug("Dynamic GPU allocation is disabled. Using static fallback.")

    def get_cluster_state(self) -> Dict:
        """Query cluster state via ClusterStateJSON command."""
        if not self.enable_dynamic:
            return None

        try:
            result = subprocess.run(
                ["bash", "-c", self.cluster_cmd],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            self.cluster_state = json.loads(result.stdout)
            return self.cluster_state
        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            logger.warning(f"Could not query cluster state: {e}")
            logger.debug("Falling back to static GPU allocation.")
            self.enable_dynamic = False  # Disable for future calls
            return None

    def get_available_gpus(self) -> List[Dict]:
        """
        Get list of available GPUs with their stats.

        Returns:
            List of dicts with keys: name, total, used, available, vram_gb
        """
        if self.cluster_state is None:
            self.get_cluster_state()

        if not self.cluster_state or "gpu" not in self.cluster_state:
            return []

        gpus = []
        for gpu in self.cluster_state["gpu"]:
            gpu_info = {
                "name": gpu["name"],
                "total": int(gpu["total"]),
                "used": int(gpu["used"]),
                "available": int(gpu["total"]) - int(gpu["used"]),
                "vram_gb": self.GPU_VRAM_MAP.get(gpu["name"], 0),
            }
            gpus.append(gpu_info)

        return gpus

    def select_gpu(
        self,
        min_vram_gb: int = 0,
        max_vram_gb: int = 1000,
        prefer_smallest: bool = True,
        require_available: bool = True,
    ) -> Optional[str]:
        """
        Select appropriate GPU based on VRAM requirements.

        Args:
            min_vram_gb: Minimum VRAM required (GB)
            max_vram_gb: Maximum VRAM to allocate (GB) - helps avoid over-allocation
            prefer_smallest: If True, prefer smallest GPU that meets requirements
            require_available: If True, only select GPUs with available slots

        Returns:
            GPU name string (e.g., "nvidia_h100_nvl_1g.24gb") or None
        """
        if not self.enable_dynamic:
            return None  # Will trigger fallback in get_gres_string

        available_gpus = self.get_available_gpus()

        if not available_gpus:
            logger.warning("No GPU information available from cluster")
            return None

        # Filter GPUs by requirements
        suitable_gpus = [
            gpu
            for gpu in available_gpus
            if min_vram_gb <= gpu["vram_gb"] <= max_vram_gb
            and (not require_available or gpu["available"] > 0)
        ]

        if not suitable_gpus:
            # Fallback: relax availability constraint
            if require_available:
                logger.warning(
                    f"No available GPUs with {min_vram_gb}-{max_vram_gb}GB VRAM"
                )
                suitable_gpus = [
                    gpu
                    for gpu in available_gpus
                    if min_vram_gb <= gpu["vram_gb"] <= max_vram_gb
                ]

            if not suitable_gpus:
                logger.warning(
                    f"No GPUs found matching VRAM requirements {min_vram_gb}-{max_vram_gb}GB"
                )
                return None

        # Sort by VRAM preference first
        suitable_gpus.sort(key=lambda x: x["vram_gb"], reverse=(not prefer_smallest))
        # Then by availability (stable sort keeps VRAM preference)
        suitable_gpus.sort(key=lambda x: x["available"], reverse=True)

        selected = suitable_gpus[0]
        logger.debug(
            f"Selected GPU: {selected['name']} ({selected['vram_gb']}GB, {selected['available']}/{selected['total']} available)"
        )

        return selected["name"]

    def get_gres_string(
        self,
        min_vram_gb: int = 0,
        max_vram_gb: int = 1000,
        num_gpus: int = 1,
        prefer_smallest: bool = True,
        fallback_gpu: Optional[str] = None,
    ) -> str:
        """
        Generate SLURM gres string for GPU allocation.

        Args:
            min_vram_gb: Minimum VRAM required
            max_vram_gb: Maximum VRAM to allocate
            num_gpus: Number of GPUs to request
            prefer_smallest: Prefer smallest suitable GPU
            fallback_gpu: GPU to use if dynamic selection fails

        Returns:
            SLURM gres string (e.g., "gpu:nvidia_h100_nvl_1g.24gb:1")
        """
        gpu_name = self.select_gpu(
            min_vram_gb=min_vram_gb,
            max_vram_gb=max_vram_gb,
            prefer_smallest=prefer_smallest,
        )

        if gpu_name is None:
            if fallback_gpu:
                # Special sentinel for generic gpu:1
                if fallback_gpu == "__GENERIC_GPU1__":
                    logger.warning("Using generic fallback GPU: gpu:1")
                    return f"gpu:{num_gpus}"
                logger.warning(f"Using fallback GPU: {fallback_gpu}")
                gpu_name = fallback_gpu
            else:
                gpu_name = "nvidia_h100_nvl_1g.24gb"
                logger.warning(f"Using default fallback GPU: {gpu_name}")

        return f"gpu:{gpu_name}:{num_gpus}"


def get_optimal_cpu(
    min_vram_gb: int = 0,
    max_vram_gb: int = 1000,
    num_gpus: int = 1,
    prefer_smallest: bool = True,
    fallback_gpu: Optional[str] = None,
    enable_dynamic: bool = True,
    cluster_cmd: str = "module load admin && ClusterStateJSON",
) -> str:
    """
    Convenience function for use in Snakemake rules.

    Usage in Snakefile:
        from workflow.scripts.gpu_allocator import get_optimal_cpu

        rule my_gpu_rule:
            resources:
                gres=lambda wildcards: get_optimal_cpu(min_vram_gb=20, max_vram_gb=50)

    To disable dynamic allocation, set environment variable:
        export DISABLE_DYNAMIC_GPU=1

    Or pass enable_dynamic=False:
        gres=lambda wildcards: get_optimal_cpu(..., enable_dynamic=False)

    Args:
        min_vram_gb: Minimum VRAM required
        max_vram_gb: Maximum VRAM to allocate
        num_gpus: Number of GPUs
        prefer_smallest: Prefer smallest suitable GPU
        fallback_gpu: Fallback GPU name if selection fails
        enable_dynamic: Enable dynamic allocation (can be disabled via DISABLE_DYNAMIC_GPU env var)
        cluster_cmd: Command to query cluster state (customize for your cluster)

    Returns:
        SLURM gres string
    """
    allocator = GPUAllocator(cluster_cmd=cluster_cmd, enable_dynamic=enable_dynamic)
    return allocator.get_gres_string(
        min_vram_gb=min_vram_gb,
        max_vram_gb=max_vram_gb,
        num_gpus=num_gpus,
        prefer_smallest=prefer_smallest,
        fallback_gpu=fallback_gpu,
    )


if __name__ == "__main__":
    # Test/debug mode
    allocator = GPUAllocator()

    print("=== Cluster GPU Status ===")
    gpus = allocator.get_available_gpus()
    for gpu in gpus:
        print(
            f"  {gpu['name']}: {gpu['available']}/{gpu['total']} available ({gpu['vram_gb']}GB)"
        )

    print("\n=== Test Allocations ===")

    test_cases = [
        {"min_vram_gb": 10, "max_vram_gb": 15, "desc": "Light workload (10-15GB)"},
        {"min_vram_gb": 20, "max_vram_gb": 30, "desc": "Medium workload (20-30GB)"},
        {"min_vram_gb": 40, "max_vram_gb": 50, "desc": "Heavy workload (40-50GB)"},
        {"min_vram_gb": 80, "max_vram_gb": 100, "desc": "Full GPU (80-100GB)"},
    ]

    for test in test_cases:
        print(f"\n{test['desc']}:")
        gres = allocator.get_gres_string(
            min_vram_gb=test["min_vram_gb"], max_vram_gb=test["max_vram_gb"]
        )
        print(f"  gres={gres}")
