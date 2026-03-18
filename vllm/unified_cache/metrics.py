# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prometheus metrics for the unified cache system.

Exports metrics that can be scraped by llm-d's inference scheduler
for cache-aware routing decisions.

Metrics:
- vllm_unified_cache_expert_hit_rate: Expert cache hit rate
- vllm_unified_cache_expert_gpu_memory_bytes: GPU memory used by expert cache
- vllm_unified_cache_expert_cpu_memory_bytes: CPU memory used by expert cache
- vllm_unified_cache_expert_gpu_count: Number of experts on GPU
- vllm_unified_cache_expert_total_count: Total number of experts
- vllm_unified_cache_expert_loads_total: Total expert loads from CPU→GPU
- vllm_unified_cache_expert_evictions_total: Total expert evictions from GPU→CPU
- vllm_unified_cache_expert_load_time_ms: Average expert load time
- vllm_unified_cache_kv_budget_ratio: Current KV cache budget ratio
- vllm_unified_cache_rebalance_total: Total rebalancing operations
"""

from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from prometheus_client import Gauge, Counter, Histogram

    EXPERT_HIT_RATE = Gauge(
        "vllm_unified_cache_expert_hit_rate",
        "Expert cache hit rate (0-1)",
        ["model_name"],
    )
    EXPERT_GPU_MEMORY = Gauge(
        "vllm_unified_cache_expert_gpu_memory_bytes",
        "GPU memory used by expert cache in bytes",
        ["model_name"],
    )
    EXPERT_CPU_MEMORY = Gauge(
        "vllm_unified_cache_expert_cpu_memory_bytes",
        "CPU memory used by expert cache in bytes",
        ["model_name"],
    )
    EXPERT_GPU_COUNT = Gauge(
        "vllm_unified_cache_expert_gpu_count",
        "Number of experts currently cached on GPU",
        ["model_name"],
    )
    EXPERT_TOTAL_COUNT = Gauge(
        "vllm_unified_cache_expert_total_count",
        "Total number of experts registered",
        ["model_name"],
    )
    EXPERT_LOADS = Counter(
        "vllm_unified_cache_expert_loads_total",
        "Total expert loads from CPU to GPU",
        ["model_name"],
    )
    EXPERT_EVICTIONS = Counter(
        "vllm_unified_cache_expert_evictions_total",
        "Total expert evictions from GPU to CPU",
        ["model_name"],
    )
    EXPERT_LOAD_TIME = Histogram(
        "vllm_unified_cache_expert_load_time_ms",
        "Expert load time in milliseconds",
        ["model_name"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
    )
    KV_BUDGET_RATIO = Gauge(
        "vllm_unified_cache_kv_budget_ratio",
        "Current fraction of cache budget allocated to KV cache",
        ["model_name"],
    )
    REBALANCE_TOTAL = Counter(
        "vllm_unified_cache_rebalance_total",
        "Total cache budget rebalancing operations",
        ["model_name"],
    )
    EXPERT_GPU_BUDGET = Gauge(
        "vllm_unified_cache_expert_gpu_budget_bytes",
        "GPU memory budget for expert cache in bytes",
        ["model_name"],
    )

    _METRICS_AVAILABLE = True

except ImportError:
    _METRICS_AVAILABLE = False
    logger.warning("prometheus_client not available; unified cache metrics disabled")


class UnifiedCacheMetricsCollector:
    """
    Collects and exports Prometheus metrics from the unified cache system.

    Should be instantiated once and called periodically (e.g., every scheduler step).
    """

    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self._last_loads = 0
        self._last_evictions = 0

    def update(
        self,
        hit_rate: float = 0.0,
        gpu_memory_used: int = 0,
        cpu_memory_used: int = 0,
        gpu_expert_count: int = 0,
        total_expert_count: int = 0,
        total_loads: int = 0,
        total_evictions: int = 0,
        avg_load_time_ms: float = 0.0,
        kv_budget_ratio: float = 0.5,
        rebalance_count: int = 0,
        expert_gpu_budget: int = 0,
    ) -> None:
        """Update all metrics with current values."""
        if not _METRICS_AVAILABLE:
            return

        labels = {"model_name": self.model_name}

        EXPERT_HIT_RATE.labels(**labels).set(hit_rate)
        EXPERT_GPU_MEMORY.labels(**labels).set(gpu_memory_used)
        EXPERT_CPU_MEMORY.labels(**labels).set(cpu_memory_used)
        EXPERT_GPU_COUNT.labels(**labels).set(gpu_expert_count)
        EXPERT_TOTAL_COUNT.labels(**labels).set(total_expert_count)
        KV_BUDGET_RATIO.labels(**labels).set(kv_budget_ratio)
        EXPERT_GPU_BUDGET.labels(**labels).set(expert_gpu_budget)

        # Counters: increment by delta since last update
        new_loads = total_loads - self._last_loads
        if new_loads > 0:
            EXPERT_LOADS.labels(**labels).inc(new_loads)
            self._last_loads = total_loads

        new_evictions = total_evictions - self._last_evictions
        if new_evictions > 0:
            EXPERT_EVICTIONS.labels(**labels).inc(new_evictions)
            self._last_evictions = total_evictions

    def update_from_manager(self, manager) -> None:
        """Update metrics directly from a UnifiedCacheManager instance."""
        stats = manager.get_stats()
        state = manager.get_expert_cache_state()

        self.update(
            hit_rate=state["hit_rate"],
            gpu_memory_used=state["gpu_memory_used"],
            cpu_memory_used=manager.expert_cache.cpu_memory_used,
            gpu_expert_count=stats.expert_gpu_count,
            total_expert_count=stats.expert_total_count,
            total_loads=state["total_loads"],
            total_evictions=state["total_evictions"],
            avg_load_time_ms=state["avg_load_time_ms"],
            kv_budget_ratio=stats.kv_budget_ratio,
            rebalance_count=stats.rebalance_count,
            expert_gpu_budget=state["gpu_memory_budget"],
        )
