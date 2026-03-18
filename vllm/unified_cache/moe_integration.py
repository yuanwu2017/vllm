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

from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.unified_cache.expert_cache_engine import ExpertCacheEngine
from vllm.unified_cache.expert_tracker import ExpertActivationTracker

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
    ) -> "ExpertCacheRegistry":
        instance = cls(expert_cache_engine, enable_tracking)
        cls._instance = instance
        return instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def __init__(
        self,
        expert_cache_engine: ExpertCacheEngine,
        enable_tracking: bool = False,
    ):
        self.cache_engine = expert_cache_engine
        self.tracker = ExpertActivationTracker.get_instance()
        if enable_tracking:
            self.tracker.enable()
        # Map from layer_name (prefix) to layer_id for tracking
        self._layer_name_to_id: dict[str, int] = {}
        self._next_layer_id = 0

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
