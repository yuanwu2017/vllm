# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
UnifiedCacheManager: Single coordinator for KV cache and expert weight caching.

This is the key contribution of the unified cache system. It manages a shared
GPU memory budget between KV cache blocks and MoE expert weights, dynamically
rebalancing based on workload pressure.

Design:
- Maintains separate sub-managers for KV and expert caches
- Coordinates eviction decisions across both cache types
- Dynamically adjusts memory budgets based on pressure signals
- Provides unified metrics and cache state reporting

Memory layout on GPU:
    total_gpu_memory = model_params + kv_cache + expert_cache + overhead
    cache_budget = total_gpu_memory - model_params - overhead
    kv_budget = cache_budget × kv_ratio  (dynamic)
    expert_budget = cache_budget × (1 - kv_ratio)  (dynamic)
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.unified_cache.expert_cache_engine import ExpertCacheEngine
from vllm.unified_cache.expert_evictor import (
    CacheItemType,
    EvictionCandidate,
    ExpertEvictionPolicy,
    UnifiedEvictionPolicy,
)

logger = init_logger(__name__)


@dataclass
class MemoryPressure:
    """Memory pressure signals for a cache subsystem."""
    utilization: float  # 0.0 - 1.0
    eviction_rate: float  # Evictions per second
    miss_rate: float  # Cache misses / total accesses
    pending_bytes: int  # Bytes waiting to be allocated


@dataclass
class UnifiedCacheStats:
    """Aggregate statistics across KV and expert caches."""
    # KV cache stats
    kv_gpu_blocks_used: int = 0
    kv_gpu_blocks_total: int = 0
    kv_hit_rate: float = 0.0
    # Expert cache stats
    expert_gpu_memory_used: int = 0
    expert_gpu_memory_budget: int = 0
    expert_hit_rate: float = 0.0
    expert_gpu_count: int = 0
    expert_total_count: int = 0
    # Unified stats
    total_gpu_memory_used: int = 0
    total_gpu_memory_budget: int = 0
    kv_budget_ratio: float = 0.5
    rebalance_count: int = 0
    last_rebalance_time: float = 0.0


class UnifiedCacheManager:
    """
    Coordinates KV cache and expert weight cache under a unified GPU memory budget.

    The manager monitors memory pressure from both subsystems and dynamically
    adjusts their GPU memory budgets. When one subsystem is under pressure
    and the other has headroom, memory is redistributed.

    Args:
        total_cache_budget: Total GPU memory available for both caches (bytes).
        expert_cache_engine: The expert weight cache engine.
        initial_kv_ratio: Initial fraction of cache budget allocated to KV cache.
        min_kv_ratio: Minimum KV cache budget ratio (prevents starvation).
        max_kv_ratio: Maximum KV cache budget ratio.
        rebalance_interval: Minimum seconds between rebalance operations.
        hidden_size: Model hidden size (for eviction scoring).
        intermediate_size: MoE intermediate size (for eviction scoring).
        device: CUDA device.
    """

    def __init__(
        self,
        total_cache_budget: int,
        expert_cache_engine: ExpertCacheEngine,
        initial_kv_ratio: float = 0.5,
        min_kv_ratio: float = 0.2,
        max_kv_ratio: float = 0.8,
        rebalance_interval: float = 5.0,
        hidden_size: int = 0,
        intermediate_size: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.total_cache_budget = total_cache_budget
        self.expert_cache = expert_cache_engine
        self.device = device or torch.device("cuda")

        # Budget ratios
        self.kv_ratio = initial_kv_ratio
        self.min_kv_ratio = min_kv_ratio
        self.max_kv_ratio = max_kv_ratio

        # Compute initial budgets
        self._kv_budget = int(total_cache_budget * self.kv_ratio)
        self._expert_budget = total_cache_budget - self._kv_budget
        self.expert_cache.update_gpu_budget(self._expert_budget)

        # Rebalancing
        self.rebalance_interval = rebalance_interval
        self._last_rebalance_time = time.monotonic()

        # Eviction policies
        self.expert_eviction_policy = ExpertEvictionPolicy(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.unified_eviction_policy = UnifiedEvictionPolicy(
            expert_policy=self.expert_eviction_policy,
        )

        # Pressure tracking (exponential moving average)
        self._kv_pressure_ema = 0.0
        self._expert_pressure_ema = 0.0
        self._pressure_decay = 0.9

        # Stats
        self.stats = UnifiedCacheStats()
        self.stats.total_gpu_memory_budget = total_cache_budget
        self.stats.kv_budget_ratio = self.kv_ratio

        logger.info(
            "UnifiedCacheManager initialized: total_budget=%.2f GiB, "
            "kv_ratio=%.2f, kv_budget=%.2f GiB, expert_budget=%.2f GiB",
            total_cache_budget / (1024**3),
            self.kv_ratio,
            self._kv_budget / (1024**3),
            self._expert_budget / (1024**3),
        )

    @property
    def kv_budget(self) -> int:
        """Current GPU memory budget for KV cache (bytes)."""
        return self._kv_budget

    @property
    def expert_budget(self) -> int:
        """Current GPU memory budget for expert cache (bytes)."""
        return self._expert_budget

    def report_kv_pressure(
        self,
        utilization: float,
        miss_rate: float = 0.0,
        eviction_rate: float = 0.0,
    ) -> None:
        """
        Report current KV cache memory pressure.

        Called periodically by the KV cache subsystem to inform the unified
        manager about its memory state.

        Args:
            utilization: Fraction of KV cache budget currently used (0-1).
            miss_rate: Cache miss rate (0-1).
            eviction_rate: Evictions per second.
        """
        pressure = utilization * 0.5 + miss_rate * 0.3 + min(eviction_rate / 100, 1.0) * 0.2
        self._kv_pressure_ema = (
            self._pressure_decay * self._kv_pressure_ema
            + (1 - self._pressure_decay) * pressure
        )

    def report_expert_pressure(self) -> None:
        """
        Compute and update expert cache memory pressure from internal stats.

        Called periodically to track expert cache utilization.
        """
        utilization = self.expert_cache.gpu_utilization
        miss_rate = 1.0 - self.expert_cache.hit_rate
        eviction_rate = self.expert_cache.stats.total_evictions  # cumulative

        pressure = utilization * 0.5 + miss_rate * 0.3 + min(eviction_rate / 100, 1.0) * 0.2
        self._expert_pressure_ema = (
            self._pressure_decay * self._expert_pressure_ema
            + (1 - self._pressure_decay) * pressure
        )

    def maybe_rebalance(self) -> bool:
        """
        Check if rebalancing is needed and perform it if so.

        Rebalancing adjusts the GPU memory split between KV and expert caches
        based on their relative pressure signals.

        Returns:
            True if rebalancing was performed.
        """
        now = time.monotonic()
        if now - self._last_rebalance_time < self.rebalance_interval:
            return False

        self._last_rebalance_time = now

        # Compute pressure-based target ratio
        total_pressure = self._kv_pressure_ema + self._expert_pressure_ema
        if total_pressure < 0.01:
            # Both subsystems are under very low pressure — no change
            return False

        # Higher KV pressure → larger KV budget ratio
        target_kv_ratio = self._kv_pressure_ema / total_pressure

        # Clamp to allowed range
        target_kv_ratio = max(self.min_kv_ratio,
                              min(self.max_kv_ratio, target_kv_ratio))

        # Only rebalance if the change is significant (>2%)
        if abs(target_kv_ratio - self.kv_ratio) < 0.02:
            return False

        # Smooth the transition (move 30% toward target)
        new_kv_ratio = self.kv_ratio + 0.3 * (target_kv_ratio - self.kv_ratio)
        new_kv_ratio = max(self.min_kv_ratio,
                           min(self.max_kv_ratio, new_kv_ratio))

        old_kv_ratio = self.kv_ratio
        self.kv_ratio = new_kv_ratio

        # Recompute budgets
        self._kv_budget = int(self.total_cache_budget * self.kv_ratio)
        self._expert_budget = self.total_cache_budget - self._kv_budget

        # Update expert cache budget (may trigger evictions)
        self.expert_cache.update_gpu_budget(self._expert_budget)

        self.stats.rebalance_count += 1
        self.stats.kv_budget_ratio = self.kv_ratio
        self.stats.last_rebalance_time = now

        logger.info(
            "Cache rebalanced: kv_ratio %.3f → %.3f, "
            "kv_budget=%.2f GiB, expert_budget=%.2f GiB "
            "(kv_pressure=%.3f, expert_pressure=%.3f)",
            old_kv_ratio,
            self.kv_ratio,
            self._kv_budget / (1024**3),
            self._expert_budget / (1024**3),
            self._kv_pressure_ema,
            self._expert_pressure_ema,
        )

        return True

    def request_gpu_memory(
        self,
        requester: str,
        bytes_needed: int,
    ) -> bool:
        """
        Request GPU memory from the unified budget.

        If the requesting subsystem's budget is insufficient, try to
        free memory from the other subsystem via cross-type eviction.

        Args:
            requester: Either "kv" or "expert".
            bytes_needed: Number of bytes needed.

        Returns:
            True if the memory was successfully allocated/freed.
        """
        if requester == "kv":
            available = self._kv_budget - self._get_kv_usage()
            if available >= bytes_needed:
                return True
            # Try evicting experts to free space
            deficit = bytes_needed - available
            return self._cross_evict("expert", deficit)
        elif requester == "expert":
            available = self._expert_budget - self.expert_cache.gpu_memory_used
            if available >= bytes_needed:
                return True
            # Try evicting KV blocks to free space is handled by the
            # KV cache manager itself, but we can signal need for rebalance
            self.report_expert_pressure()
            return self.maybe_rebalance()
        return False

    def _cross_evict(self, target: str, bytes_needed: int) -> bool:
        """
        Evict items from the target cache to free GPU memory.

        Args:
            target: Which cache to evict from ("kv" or "expert").
            bytes_needed: Minimum bytes to free.

        Returns:
            True if enough memory was freed.
        """
        if target == "expert":
            candidates = self.expert_cache.get_eviction_candidates(
                count=max(1, bytes_needed // (1024 * 1024))  # rough estimate
            )
            freed = 0
            for layer_id, expert_id, score in candidates:
                if freed >= bytes_needed:
                    break
                entry = self.expert_cache._cache.get((layer_id, expert_id))
                if entry is None:
                    continue
                if self.expert_cache.evict_expert(layer_id, expert_id, sync=True):
                    freed += entry.memory_bytes

            return freed >= bytes_needed

        # KV eviction would be handled by the KV cache manager
        return False

    def _get_kv_usage(self) -> int:
        """Get current KV cache GPU memory usage. Override in integration."""
        return 0

    def get_stats(self) -> UnifiedCacheStats:
        """Get aggregate cache statistics."""
        self.stats.expert_gpu_memory_used = self.expert_cache.gpu_memory_used
        self.stats.expert_gpu_memory_budget = self._expert_budget
        self.stats.expert_hit_rate = self.expert_cache.hit_rate
        self.stats.expert_gpu_count = len(self.expert_cache._gpu_lru)
        self.stats.expert_total_count = len(self.expert_cache._cache)
        self.stats.total_gpu_memory_used = (
            self._get_kv_usage() + self.expert_cache.gpu_memory_used
        )
        return self.stats

    def get_expert_cache_state(self) -> dict:
        """
        Get expert cache state for external consumption (e.g., metrics export).

        Returns dict with:
        - gpu_experts: dict of layer_id -> list of expert_ids on GPU
        - hit_rate: overall expert cache hit rate
        - gpu_utilization: expert cache GPU memory utilization
        """
        return {
            "gpu_experts": self.expert_cache.get_all_gpu_expert_ids(),
            "hit_rate": self.expert_cache.hit_rate,
            "gpu_utilization": self.expert_cache.gpu_utilization,
            "gpu_memory_used": self.expert_cache.gpu_memory_used,
            "gpu_memory_budget": self._expert_budget,
            "total_loads": self.expert_cache.stats.total_loads,
            "total_evictions": self.expert_cache.stats.total_evictions,
            "avg_load_time_ms": self.expert_cache.stats.avg_load_time_ms,
        }

    def __repr__(self) -> str:
        return (
            f"UnifiedCacheManager("
            f"total={self.total_cache_budget / (1024**3):.2f} GiB, "
            f"kv_ratio={self.kv_ratio:.3f}, "
            f"kv_pressure={self._kv_pressure_ema:.3f}, "
            f"expert_pressure={self._expert_pressure_ema:.3f}, "
            f"rebalances={self.stats.rebalance_count})"
        )
