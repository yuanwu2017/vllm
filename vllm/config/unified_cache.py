# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the Unified KV + Expert Cache system."""

from pydantic import Field

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@config
class UnifiedCacheConfig:
    """Configuration for the unified KV + Expert cache system.

    When enabled, MoE expert weights are treated as cacheable state,
    managed alongside KV cache blocks under a unified eviction policy.
    """

    enabled: bool = False
    """Whether to enable the unified cache system. When False, vLLM operates
    with its standard behavior (all experts permanently on GPU)."""

    expert_gpu_memory_fraction: float = Field(default=0.3, gt=0, le=0.8)
    """Fraction of loaded expert weight bytes retained on GPU by the experimental
    eager runtime. Freed memory is available to the normal KV cache profiler.
    Runtime resizing of allocated KV cache blocks is not implemented."""

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

    enable_expert_prediction: bool = False
    """Whether to enable prefix-aware expert prediction (Phase 3+4).
    When True, a PrefixExpertDAG is built from observed expert activations
    and used to proactively prefetch experts before MoE layers run."""

    prediction_confidence: float = Field(default=0.1, ge=0, le=1)
    """Minimum predicted probability to trigger async expert prefetch.
    Lower values are more aggressive (prefetch more, waste some bandwidth).
    Higher values are conservative (fewer prefetches, more cache misses)."""

    dag_decay_factor: float = Field(default=0.99, gt=0, le=1)
    """Exponential decay for old observations in the PrefixExpertDAG.
    1.0 = no decay (pure accumulation). Lower values forget older data."""

    dag_max_nodes: int = Field(default=100_000, ge=1000)
    """Maximum number of nodes in the PrefixExpertDAG. Oldest nodes
    are evicted when this limit is exceeded."""

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
                initial_kv_ratio,
                self.min_kv_ratio,
            )
            self.expert_gpu_memory_fraction = 1.0 - self.min_kv_ratio
        if initial_kv_ratio > self.max_kv_ratio:
            logger.warning(
                "Initial KV ratio (%.2f) exceeds max_kv_ratio (%.2f). "
                "Adjusting expert_gpu_memory_fraction.",
                initial_kv_ratio,
                self.max_kv_ratio,
            )
            self.expert_gpu_memory_fraction = 1.0 - self.max_kv_ratio

    def validate_runtime(self, vllm_config) -> None:
        if not self.enabled:
            return
        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        if model is None or not model.enforce_eager:
            raise ValueError("Unified expert cache requires enforce_eager=True")
        if (
            any(
                getattr(parallel, name, 1) != 1
                for name in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "decode_context_parallel_size",
                    "prefill_context_parallel_size",
                )
            )
            or parallel.enable_expert_parallel
            or parallel.enable_eplb
        ):
            raise ValueError("Unified expert cache currently supports single GPU only")
        if model.quantization is not None or vllm_config.lora_config is not None:
            raise ValueError(
                "Unified expert cache does not support quantization or LoRA"
            )
        if vllm_config.speculative_config is not None or model.enable_sleep_mode:
            raise ValueError(
                "Unified expert cache does not support speculation or sleep"
            )
        if self.enable_expert_prediction:
            raise ValueError(
                "Expert prediction is not integrated with request scheduling"
            )
        offload = vllm_config.offload_config
        if offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size:
            raise ValueError(
                "Do not combine unified expert cache with weight offloading"
            )
