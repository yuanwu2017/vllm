# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MoE layer integration for the unified cache system.

This module provides hooks and utilities to integrate the ExpertCacheEngine
with vLLM's FusedMoE layer. It modifies the MoE forward pass to:

1. Record expert activations (for offline analysis and online tracking)
2. Ensure needed expert weights are on GPU before computation
3. Overlap expert loading with computation of already-available experts
4. Report cache statistics for monitoring

Integration approach:
- We use a global registry that MoE layers check during forward pass
- This avoids modifying the core FusedMoE class signature
- The registry is populated during engine initialization when
  unified_cache_config.enabled is True
"""

from typing import Optional, Sequence

import torch

from vllm.logger import init_logger
from vllm.unified_cache.expert_cache_engine import ExpertCacheEngine, ExpertLocation
from vllm.unified_cache.expert_predictor import ExpertPredictor
from vllm.unified_cache.expert_tracker import ExpertActivationTracker
from vllm.unified_cache.prefix_expert_dag import PrefixExpertDAG

logger = init_logger(__name__)


class ExpertCacheRegistry:
    """
    Global registry that connects MoE layers to the expert cache engine.

    Singleton pattern — populated during engine init, queried during forward.
    """

    _instance: Optional["ExpertCacheRegistry"] = None

    @classmethod
    def get_instance(cls) -> Optional["ExpertCacheRegistry"]:
        return cls._instance

    @classmethod
    def initialize(
        cls,
        expert_cache_engine: ExpertCacheEngine,
        enable_tracking: bool = False,
        enable_cpu_compute: bool = False,
        enable_prediction: bool = False,
        dag_decay: float = 0.99,
        dag_max_nodes: int = 100_000,
        prediction_confidence: float = 0.1,
    ) -> "ExpertCacheRegistry":
        instance = cls(
            expert_cache_engine, enable_tracking, enable_cpu_compute,
            enable_prediction, dag_decay, dag_max_nodes,
            prediction_confidence,
        )
        cls._instance = instance
        return instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def __init__(
        self,
        expert_cache_engine: ExpertCacheEngine,
        enable_tracking: bool = False,
        enable_cpu_compute: bool = False,
        enable_prediction: bool = False,
        dag_decay: float = 0.99,
        dag_max_nodes: int = 100_000,
        prediction_confidence: float = 0.1,
    ):
        self.cache_engine = expert_cache_engine
        self.tracker = ExpertActivationTracker.get_instance()
        if enable_tracking:
            self.tracker.enable()
        self.cpu_compute_enabled = enable_cpu_compute
        # Map from layer_name (prefix) to layer_id for tracking
        self._layer_name_to_id: dict[str, int] = {}
        self._next_layer_id = 0

        # Phase 3+4: Prefix-aware expert prediction
        self.prediction_enabled = enable_prediction
        self.dag: Optional[PrefixExpertDAG] = None
        self.predictor: Optional[ExpertPredictor] = None
        if enable_prediction:
            self.dag = PrefixExpertDAG(
                decay_factor=dag_decay,
                max_nodes=dag_max_nodes,
            )
            self.predictor = ExpertPredictor(
                dag=self.dag,
                cache_engine=expert_cache_engine,
                confidence_threshold=prediction_confidence,
            )
            logger.info(
                "Expert prediction enabled: dag_decay=%.3f, "
                "max_nodes=%d, confidence=%.2f",
                dag_decay, dag_max_nodes, prediction_confidence,
            )

    def get_layer_id(self, layer_name: str) -> int:
        """Get or assign a sequential layer ID for a named MoE layer."""
        if layer_name not in self._layer_name_to_id:
            self._layer_name_to_id[layer_name] = self._next_layer_id
            self._next_layer_id += 1
        return self._layer_name_to_id[layer_name]


def unified_cache_pre_forward(
    layer_name: str,
    topk_ids: torch.Tensor,
) -> None:
    """
    Hook called in FusedMoE forward AFTER router selection, BEFORE expert compute.

    This function:
    1. Records which experts were selected (for tracking)
    2. Ensures all selected experts are loaded on GPU
    3. Prefetches experts that are likely needed soon

    Args:
        layer_name: The FusedMoE layer's prefix name (e.g., "model.layers.0.mlp.experts").
        topk_ids: Tensor of shape (num_tokens, top_k) with selected expert indices.
    """
    registry = ExpertCacheRegistry.get_instance()
    if registry is None:
        return

    layer_id = registry.get_layer_id(layer_name)

    # 1. Record expert activations
    if registry.tracker.enabled:
        registry.tracker.record(layer_id=layer_id, topk_ids=topk_ids)

    # 1b. Record feedback into prefix-expert DAG (Phase 3+4)
    if registry.prediction_enabled and registry.predictor is not None:
        expert_ids = topk_ids.detach().cpu().flatten().tolist()
        # Use _current_block_hashes if set by predict_for_request()
        block_hashes = getattr(registry, '_current_block_hashes', None)
        if block_hashes:
            registry.predictor.record_feedback(
                block_hashes=block_hashes,
                layer_id=layer_id,
                actual_expert_ids=expert_ids,
                num_tokens=topk_ids.shape[0],
            )

    # 2. Ensure selected experts are on GPU
    # Get unique expert IDs from the batch
    unique_experts = topk_ids.unique().cpu().tolist()

    # Check which experts need loading
    experts_to_load = []
    for expert_id in unique_experts:
        tensors = registry.cache_engine.get_expert(layer_id, int(expert_id))
        if tensors is None:
            experts_to_load.append(int(expert_id))

    # Load missing experts (sync for correctness)
    if experts_to_load:
        events = registry.cache_engine.prefetch_experts(
            layer_id=layer_id,
            expert_ids=experts_to_load,
        )
        # Must synchronize before compute to ensure weights are available
        for event in events:
            event.synchronize()


def predict_experts_for_request(
    block_hashes: Sequence[int],
) -> dict[int, list[int]]:
    """
    Predict and prefetch experts for an incoming request (Phase 4).

    Should be called at request scheduling time, before any MoE layers run.
    This sets the current block hashes on the registry so that subsequent
    unified_cache_pre_forward() calls can record feedback into the DAG.

    Args:
        block_hashes: Prefix block hashes from vLLM's KV cache lookup.

    Returns:
        Dict of layer_id -> list of prefetched expert_ids.
    """
    registry = ExpertCacheRegistry.get_instance()
    if registry is None or not registry.prediction_enabled:
        return {}
    if registry.predictor is None:
        return {}

    # Store block hashes so forward hooks can record feedback
    registry._current_block_hashes = list(block_hashes)

    return registry.predictor.predict_and_prefetch(block_hashes)


def clear_request_context() -> None:
    """Clear per-request state after a request completes."""
    registry = ExpertCacheRegistry.get_instance()
    if registry is not None:
        registry._current_block_hashes = None


def unified_cache_get_expert_weights(
    layer_name: str,
    expert_id: int,
    weight_name: str,
) -> Optional[torch.Tensor]:
    """
    Get a specific expert weight tensor from the cache.

    Called during MoE computation when the layer needs expert weights.
    Returns the GPU tensor if cached, None otherwise.

    Args:
        layer_name: The FusedMoE layer's prefix name.
        expert_id: The expert index.
        weight_name: Weight tensor name (e.g., "w13_weight", "w2_weight").

    Returns:
        The weight tensor on GPU, or None if not available.
    """
    registry = ExpertCacheRegistry.get_instance()
    if registry is None:
        return None

    layer_id = registry.get_layer_id(layer_name)
    tensors = registry.cache_engine.get_expert(layer_id, expert_id)
    if tensors is None:
        return None
    return tensors.get(weight_name)


def register_experts_from_layer(
    layer_name: str,
    num_experts: int,
    weight_dict: dict[int, dict[str, torch.Tensor]],
    initial_gpu_experts: Optional[set[int]] = None,
) -> None:
    """
    Register all experts from a FusedMoE layer with the cache engine.

    Called during model initialization after weights are loaded.

    Args:
        layer_name: The FusedMoE layer's prefix name.
        num_experts: Total number of experts in this layer.
        weight_dict: Mapping of expert_id -> {weight_name: tensor}.
        initial_gpu_experts: Set of expert IDs to keep on GPU initially.
            If None, all experts start on GPU (default vLLM behavior).
    """
    registry = ExpertCacheRegistry.get_instance()
    if registry is None:
        return

    layer_id = registry.get_layer_id(layer_name)

    if initial_gpu_experts is None:
        initial_gpu_experts = set(range(num_experts))

    for expert_id, weights in weight_dict.items():
        on_gpu = expert_id in initial_gpu_experts
        registry.cache_engine.register_expert(
            layer_id=layer_id,
            expert_id=expert_id,
            weight_tensors=weights,
            on_gpu=on_gpu,
        )

    logger.info(
        "Registered %d experts for layer %s (layer_id=%d), "
        "%d initially on GPU",
        len(weight_dict),
        layer_name,
        layer_id,
        len(initial_gpu_experts & set(weight_dict.keys())),
    )


# ---------------------------------------------------------------------------
# Phase 2: CPU Active Compute — split dispatch
# ---------------------------------------------------------------------------


def unified_cache_split_compute(
    layer_name: str,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
    """
    Decide whether to split MoE computation between GPU and CPU.

    If all selected experts are on GPU, returns (None, None, False) — the
    caller should use the fast fused kernel as usual.

    If some experts are on CPU, returns:
        (cpu_contribution, gpu_topk_weights, True)
    where:
        - cpu_contribution: (num_tokens, hidden_dim) tensor ON GPU containing
          the weighted output of CPU-computed experts.
        - gpu_topk_weights: Modified topk_weights with CPU expert weights
          zeroed out, so the GPU kernel ignores those slots.
        - True: Indicates split compute is active.

    The caller should then run:
        gpu_result = quant_method.apply(layer, x, gpu_topk_weights, topk_ids)
        final = gpu_result + cpu_contribution

    Args:
        layer_name: FusedMoE layer's prefix name.
        x: Input hidden states on GPU. Shape: (num_tokens, hidden_dim).
        topk_weights: Router weights. Shape: (num_tokens, top_k).
        topk_ids: Selected expert IDs. Shape: (num_tokens, top_k).
        activation: Activation function name for the expert FFN.

    Returns:
        Tuple of (cpu_contribution, gpu_topk_weights, split_active).
    """
    registry = ExpertCacheRegistry.get_instance()
    if registry is None:
        return None, None, False

    # Check if CPU active compute is enabled
    if not getattr(registry, 'cpu_compute_enabled', False):
        return None, None, False

    layer_id = registry.get_layer_id(layer_name)
    cache = registry.cache_engine

    # Find which selected experts are on CPU
    unique_experts = topk_ids.unique().cpu().tolist()
    cpu_expert_set = set()
    gpu_expert_set = set()

    for eid in unique_experts:
        eid = int(eid)
        entry = cache._cache.get((layer_id, eid))
        if entry is None:
            continue
        if entry.location == ExpertLocation.CPU:
            cpu_expert_set.add(eid)
        elif entry.location == ExpertLocation.GPU:
            gpu_expert_set.add(eid)
        elif entry.location == ExpertLocation.LOADING:
            # Still loading — treat as GPU (will block on sync)
            gpu_expert_set.add(eid)

    if not cpu_expert_set:
        # All on GPU — fall through to normal fused kernel
        return None, None, False

    # Lazy import to avoid circular dependency at module load
    from vllm.unified_cache.cpu_expert_compute import (
        compute_cpu_expert_contributions,
        create_gpu_only_weights,
    )

    # 1. Compute CPU expert contributions
    def cpu_weight_getter(expert_id: int):
        return cache.get_cpu_tensors(layer_id, expert_id)

    cpu_contribution = compute_cpu_expert_contributions(
        x_gpu=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        cpu_expert_set=cpu_expert_set,
        cpu_weight_getter=cpu_weight_getter,
        activation=activation,
    )

    # 2. Zero out CPU experts in GPU weights
    gpu_topk_weights = create_gpu_only_weights(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        cpu_expert_set=cpu_expert_set,
    )

    # 3. Load the CPU experts to GPU in the background for next time
    #    (non-blocking — the CPU compute already produced the result)
    for eid in cpu_expert_set:
        cache.load_expert(layer_id, eid, sync=False)

    logger.debug(
        "Split compute for layer %s: %d GPU experts, %d CPU experts",
        layer_name, len(gpu_expert_set), len(cpu_expert_set),
    )

    return cpu_contribution, gpu_topk_weights, True
