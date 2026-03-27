# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration for the Unified KV + Expert Cache system.
"""

from pydantic import Field
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@config
@dataclass
class UnifiedCacheConfig:
    """Configuration for the unified KV + Expert cache system.

    When enabled, MoE expert weights are treated as cacheable state,
    managed alongside KV cache blocks under a unified eviction policy.
    """

    enabled: bool = False
    """Whether to enable the unified cache system. When False, vLLM operates
    with its standard behavior (all experts permanently on GPU)."""

    expert_gpu_memory_fraction: float = Field(default=0.3, gt=0, le=0.8)
    """Fraction of total cache budget to initially allocate for expert weights.
    The remaining fraction is for KV cache. This ratio is dynamically adjusted
    at runtime based on workload pressure."""

    expert_cpu_memory_gb: float = Field(default=50.0, ge=0)
    """CPU memory (in GiB) to allocate for expert weight storage. Expert
    weights evicted from GPU are stored here in pinned memory for fast
    reload."""

    min_kv_ratio: float = Field(default=0.2, ge=0.1, le=0.9)
    """Minimum fraction of cache budget guaranteed for KV cache. Prevents
    KV cache starvation when expert cache pressure is high."""

    max_kv_ratio: float = Field(default=0.8, ge=0.1, le=0.99)
    """Maximum fraction of cache budget for KV cache. Ensures some memory
    is always available for expert caching."""

    ema_decay: float = Field(default=0.95, gt=0, lt=1)
    """Decay factor for exponential moving average of expert access rates.
    Higher values give more weight to historical access patterns."""

    rebalance_interval_sec: float = Field(default=5.0, gt=0)
    """Minimum interval (seconds) between cache budget rebalancing operations.
    Lower values enable faster adaptation but add overhead."""

    pin_cpu_memory: bool = True
    """Whether to use pinned (page-locked) CPU memory for expert weights.
    Pinned memory enables faster PCIe transfers but consumes more
    host resources."""

    enable_expert_tracking: bool = False
    """Whether to enable expert activation tracking for offline analysis.
    Captures per-token expert selections for generating paper figures
    and validating cache effectiveness."""

    expert_trace_file: str = ""
    """Path to save expert activation traces (JSONL format). Only used
    when enable_expert_tracking is True. If empty, traces are kept in
    memory only."""

    max_trace_records: int = Field(default=1_000_000, ge=0)
    """Maximum number of expert activation records to keep in memory.
    Older records are discarded when the limit is reached."""

    enable_cpu_compute: bool = False
    """Whether to enable CPU active compute for MoE experts. When True,
    expert FFN computations for CPU-resident (evicted) experts are run
    directly on CPU instead of waiting for CPU→GPU weight transfer.
    This can reduce latency when few tokens are routed to a cold expert,
    as the CPU matmul is faster than the PCIe transfer overhead."""

    def __post_init__(self):
        if self.min_kv_ratio >= self.max_kv_ratio:
            raise ValueError(
                f"min_kv_ratio ({self.min_kv_ratio}) must be less than "
                f"max_kv_ratio ({self.max_kv_ratio})"
            )
        initial_kv_ratio = 1.0 - self.expert_gpu_memory_fraction
        if initial_kv_ratio < self.min_kv_ratio:
            logger.warning(
                "Initial KV ratio (%.2f) is below min_kv_ratio (%.2f). "
                "Adjusting expert_gpu_memory_fraction.",
                initial_kv_ratio, self.min_kv_ratio,
            )
            self.expert_gpu_memory_fraction = 1.0 - self.min_kv_ratio
        if initial_kv_ratio > self.max_kv_ratio:
            logger.warning(
                "Initial KV ratio (%.2f) exceeds max_kv_ratio (%.2f). "
                "Adjusting expert_gpu_memory_fraction.",
                initial_kv_ratio, self.max_kv_ratio,
            )
            self.expert_gpu_memory_fraction = 1.0 - self.max_kv_ratio
