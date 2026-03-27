# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prefix-aware Expert Prediction: Proactive expert prefetching.

Uses the PrefixExpertDAG to predict which experts will be needed for an
incoming request, based on its prefix hash chain. Predicted experts are
prefetched to GPU asynchronously before the actual MoE router decision,
hiding PCIe transfer latency behind token computation.

Prediction flow:
    1. Request arrives with prefix block hashes (from KV cache lookup)
    2. ExpertPredictor queries PrefixExpertDAG for expert predictions
    3. High-confidence predictions trigger async GPU←CPU prefetches
    4. By the time the MoE layer runs, experts are already on GPU

This module also provides feedback: after each forward pass, the actual
expert selections are recorded back into the DAG to continuously improve
prediction accuracy.
"""

from typing import Optional, Sequence

from vllm.logger import init_logger
from vllm.unified_cache.expert_cache_engine import ExpertCacheEngine, ExpertLocation
from vllm.unified_cache.prefix_expert_dag import PrefixExpertDAG

logger = init_logger(__name__)


class ExpertPredictor:
    """
    Predicts and prefetches experts based on prefix structure.

    Lifecycle:
        1. Created during engine init if unified cache + prediction enabled
        2. `predict_and_prefetch()` called at request scheduling time
        3. `record_feedback()` called in MoE forward hook after router runs
        4. Continuously improves predictions via the DAG

    Args:
        dag: The PrefixExpertDAG to query for predictions.
        cache_engine: The ExpertCacheEngine to issue prefetches to.
        confidence_threshold: Minimum predicted probability to trigger prefetch.
        max_prefetch_per_layer: Max experts to prefetch per layer per request.
        top_k_predict: How many experts to consider per layer in DAG query.
    """

    def __init__(
        self,
        dag: PrefixExpertDAG,
        cache_engine: ExpertCacheEngine,
        confidence_threshold: float = 0.1,
        max_prefetch_per_layer: int = 8,
        top_k_predict: int = 12,
    ):
        self.dag = dag
        self.cache_engine = cache_engine
        self.confidence_threshold = confidence_threshold
        self.max_prefetch_per_layer = max_prefetch_per_layer
        self.top_k_predict = top_k_predict

        # Statistics
        self._predictions_made = 0
        self._experts_prefetched = 0
        self._correct_predictions = 0
        self._total_actual_experts = 0

    def predict_and_prefetch(
        self,
        block_hashes: Sequence[int],
        layer_ids: Optional[Sequence[int]] = None,
    ) -> dict[int, list[int]]:
        """
        Predict experts and issue async prefetches for a request.

        Should be called at request scheduling time (before the first
        MoE layer runs), so prefetches overlap with attention computation.

        Args:
            block_hashes: Prefix block hashes for the incoming request
                (from vLLM's prefix cache lookup).
            layer_ids: Specific layers to predict for. If None, predict
                for all layers known to the DAG.

        Returns:
            Dict of layer_id -> list of expert_ids that were prefetched.
        """
        if not block_hashes:
            return {}

        predictions = self.dag.predict(
            block_hashes=block_hashes,
            top_k_per_layer=self.top_k_predict,
        )

        if not predictions:
            return {}

        prefetched: dict[int, list[int]] = {}
        self._predictions_made += 1

        for layer_id, expert_probs in predictions.items():
            if layer_ids is not None and layer_id not in layer_ids:
                continue

            layer_prefetched = []
            for expert_id, probability in expert_probs:
                if probability < self.confidence_threshold:
                    break  # Sorted descending, so remaining are lower
                if len(layer_prefetched) >= self.max_prefetch_per_layer:
                    break

                # Check if expert is already on GPU
                key = (layer_id, expert_id)
                entry = self.cache_engine._cache.get(key)
                if entry is None:
                    continue
                if entry.location in (ExpertLocation.GPU, ExpertLocation.LOADING):
                    continue  # Already available or in transit

                # Issue async prefetch
                event = self.cache_engine.load_expert(
                    layer_id, expert_id, sync=False
                )
                if event is not None:
                    layer_prefetched.append(expert_id)
                    self._experts_prefetched += 1

            if layer_prefetched:
                prefetched[layer_id] = layer_prefetched

        if prefetched:
            total = sum(len(v) for v in prefetched.values())
            logger.debug(
                "Predictive prefetch: %d experts across %d layers "
                "(prefix depth=%d)",
                total, len(prefetched), len(block_hashes),
            )

        return prefetched

    def record_feedback(
        self,
        block_hashes: Sequence[int],
        layer_id: int,
        actual_expert_ids: Sequence[int],
        num_tokens: int = 1,
    ) -> None:
        """
        Record actual expert selections back into the DAG.

        Called after each MoE forward pass to continuously improve
        prediction accuracy.

        Args:
            block_hashes: Prefix block hashes of the current request.
            layer_id: The MoE layer that just ran.
            actual_expert_ids: Expert IDs actually selected by the router.
            num_tokens: Number of tokens in this batch.
        """
        if not block_hashes:
            return

        self.dag.record(
            block_hashes=block_hashes,
            layer_id=layer_id,
            expert_ids=actual_expert_ids,
            num_tokens=num_tokens,
        )

        self._total_actual_experts += len(actual_expert_ids)

    def evaluate_accuracy(
        self,
        block_hashes: Sequence[int],
        layer_id: int,
        actual_expert_ids: Sequence[int],
    ) -> float:
        """
        Evaluate prediction accuracy for a single forward pass.

        Returns the fraction of actual experts that were predicted
        (hit rate). Used for monitoring and paper metrics.

        Args:
            block_hashes: Prefix block hashes.
            layer_id: MoE layer index.
            actual_expert_ids: Experts actually selected.

        Returns:
            Hit rate in [0, 1]. 1.0 = all actual experts were predicted.
        """
        predictions = self.dag.predict(
            block_hashes=block_hashes,
            top_k_per_layer=self.top_k_predict,
        )

        layer_preds = predictions.get(layer_id, [])
        if not layer_preds or not actual_expert_ids:
            return 0.0

        predicted_set = {eid for eid, _ in layer_preds}
        actual_set = set(actual_expert_ids)
        hits = len(predicted_set & actual_set)
        accuracy = hits / len(actual_set)

        self._correct_predictions += hits
        return accuracy

    def get_stats(self) -> dict:
        """Return predictor statistics."""
        overall_accuracy = 0.0
        if self._total_actual_experts > 0:
            overall_accuracy = (
                self._correct_predictions / self._total_actual_experts
            )
        return {
            "predictions_made": self._predictions_made,
            "experts_prefetched": self._experts_prefetched,
            "correct_predictions": self._correct_predictions,
            "total_actual_experts": self._total_actual_experts,
            "overall_accuracy": overall_accuracy,
            "dag_stats": self.dag.get_stats(),
        }

    def reset_stats(self) -> None:
        """Reset prediction statistics (but keep DAG data)."""
        self._predictions_made = 0
        self._experts_prefetched = 0
        self._correct_predictions = 0
        self._total_actual_experts = 0
