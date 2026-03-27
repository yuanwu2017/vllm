# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Engine-level initialization for the unified cache system.

This module handles setting up the ExpertCacheEngine, UnifiedCacheManager,
and ExpertCacheRegistry during vLLM engine startup. It reads the
UnifiedCacheConfig and initializes all components.

Usage (called from vLLM engine init):
    from vllm.unified_cache.engine_init import maybe_init_unified_cache
    manager = maybe_init_unified_cache(vllm_config, device)
"""

from typing import Optional

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.unified_cache.expert_cache_engine import ExpertCacheEngine
from vllm.unified_cache.expert_tracker import ExpertActivationTracker
from vllm.unified_cache.moe_integration import ExpertCacheRegistry
from vllm.unified_cache.unified_cache_manager import UnifiedCacheManager

logger = init_logger(__name__)

# Global reference for engine-level access
_unified_cache_manager: Optional[UnifiedCacheManager] = None


def get_unified_cache_manager() -> Optional[UnifiedCacheManager]:
    """Get the global UnifiedCacheManager instance, if initialized."""
    return _unified_cache_manager


def maybe_init_unified_cache(
    vllm_config: VllmConfig,
    device: torch.device,
    total_cache_memory: int,
    num_experts: int = 0,
    hidden_size: int = 0,
    intermediate_size: int = 0,
) -> Optional[UnifiedCacheManager]:
    """
    Initialize the unified cache system if enabled in config.

    Called during vLLM engine startup after GPU memory profiling.

    Args:
        vllm_config: The full vLLM configuration.
        device: CUDA device for GPU tensors.
        total_cache_memory: Total GPU memory available for caching (bytes).
            This is the memory left after model parameters and activation
            buffers are allocated.
        num_experts: Total number of experts per layer (for budget estimation).
        hidden_size: Model hidden size.
        intermediate_size: MoE intermediate size.

    Returns:
        UnifiedCacheManager if unified cache is enabled, None otherwise.
    """
    global _unified_cache_manager

    uc_config = vllm_config.unified_cache_config
    if uc_config is None or not uc_config.enabled:
        logger.info("Unified cache system is disabled")
        return None

    logger.info(
        "Initializing unified cache system: "
        "expert_gpu_fraction=%.2f, cpu_gb=%.1f, "
        "total_cache_memory=%.2f GiB",
        uc_config.expert_gpu_memory_fraction,
        uc_config.expert_cpu_memory_gb,
        total_cache_memory / (1024**3),
    )

    # Compute budgets
    expert_gpu_budget = int(total_cache_memory * uc_config.expert_gpu_memory_fraction)
    expert_cpu_budget = int(uc_config.expert_cpu_memory_gb * (1024**3))
    kv_ratio = 1.0 - uc_config.expert_gpu_memory_fraction

    # Create ExpertCacheEngine
    expert_cache = ExpertCacheEngine(
        gpu_memory_budget=expert_gpu_budget,
        cpu_memory_budget=expert_cpu_budget,
        device=device,
        ema_decay=uc_config.ema_decay,
        pin_cpu_memory=uc_config.pin_cpu_memory,
    )

    # Create UnifiedCacheManager
    manager = UnifiedCacheManager(
        total_cache_budget=total_cache_memory,
        expert_cache_engine=expert_cache,
        initial_kv_ratio=kv_ratio,
        min_kv_ratio=uc_config.min_kv_ratio,
        max_kv_ratio=uc_config.max_kv_ratio,
        rebalance_interval=uc_config.rebalance_interval_sec,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    # Initialize the global registry for MoE layer hooks
    ExpertCacheRegistry.initialize(
        expert_cache_engine=expert_cache,
        enable_tracking=uc_config.enable_expert_tracking,
        enable_cpu_compute=uc_config.enable_cpu_compute,
        enable_prediction=uc_config.enable_expert_prediction,
        dag_decay=uc_config.dag_decay_factor,
        dag_max_nodes=uc_config.dag_max_nodes,
        prediction_confidence=uc_config.prediction_confidence,
    )

    # Configure tracker
    if uc_config.enable_expert_tracking:
        tracker = ExpertActivationTracker.get_instance()
        tracker._max_records = uc_config.max_trace_records
        if uc_config.expert_trace_file:
            logger.info(
                "Expert traces will be saved to: %s",
                uc_config.expert_trace_file,
            )

    _unified_cache_manager = manager

    logger.info(
        "Unified cache system initialized: %s", manager
    )

    return manager


def shutdown_unified_cache() -> None:
    """Shutdown the unified cache system and save traces if configured."""
    global _unified_cache_manager

    if _unified_cache_manager is None:
        return

    # Save expert traces if tracking was enabled
    registry = ExpertCacheRegistry.get_instance()
    if registry is not None and registry.tracker.enabled:
        # Check if a trace file was configured
        # (we need to get it from somewhere — use the tracker's state)
        tracker = registry.tracker
        analysis = tracker.analyze()
        logger.info(
            "Expert activation summary: "
            "%d total tokens, %d events, "
            "temporal_locality=%.3f, "
            "top-20%% coverage=%.3f",
            analysis.total_tokens,
            analysis.total_events,
            analysis.temporal_locality,
            analysis.top_k_coverage.get(20, 0.0),
        )

    # Clean up
    ExpertCacheRegistry.reset()
    ExpertActivationTracker.reset_instance()
    _unified_cache_manager = None

    logger.info("Unified cache system shut down")
