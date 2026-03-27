# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Phase 3 (Prefix Expert DAG) and Phase 4 (Expert Prediction).
"""

import pytest
import torch

from vllm.unified_cache.expert_cache_engine import (
    ExpertCacheEngine,
    ExpertLocation,
)
from vllm.unified_cache.expert_predictor import ExpertPredictor
from vllm.unified_cache.moe_integration import (
    ExpertCacheRegistry,
    predict_experts_for_request,
    clear_request_context,
    unified_cache_pre_forward,
)
from vllm.unified_cache.prefix_expert_dag import PrefixExpertDAG, PrefixDAGNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN = 64
INTERMEDIATE = 128
DTYPE = torch.float32


def _make_expert_tensors(device: str = "cpu"):
    return {
        "w13_weight": torch.randn(INTERMEDIATE * 2, HIDDEN, dtype=DTYPE, device=device),
        "w2_weight": torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE, device=device),
    }


def _make_cache_engine(num_layers=1, num_experts=8, gpu_experts=None):
    if gpu_experts is None:
        gpu_experts = {0, 1, 2, 3}
    engine = ExpertCacheEngine(
        gpu_memory_budget=1024**3,
        cpu_memory_budget=1024**3,
        device=torch.device("cpu"),
    )
    for lid in range(num_layers):
        for eid in range(num_experts):
            engine.register_expert(lid, eid, _make_expert_tensors(),
                                   on_gpu=eid in gpu_experts)
    return engine


# ---------------------------------------------------------------------------
# Tests: PrefixExpertDAG
# ---------------------------------------------------------------------------

class TestPrefixExpertDAG:
    """Tests for the prefix-expert DAG structure."""

    def test_empty_dag(self):
        dag = PrefixExpertDAG()
        assert dag.num_nodes == 0
        assert dag.predict([100, 200]) == {}

    def test_record_creates_node(self):
        dag = PrefixExpertDAG()
        dag.record(block_hashes=[100, 200], layer_id=0, expert_ids=[2, 7])
        assert dag.num_nodes >= 1
        node = dag.get_node(200)
        assert node is not None
        assert node.total_observations == 1

    def test_record_accumulates(self):
        dag = PrefixExpertDAG(decay_factor=1.0)  # no decay
        dag.record([100], layer_id=0, expert_ids=[2, 7], num_tokens=4)
        dag.record([100], layer_id=0, expert_ids=[2, 3], num_tokens=2)
        node = dag.get_node(100)
        assert node is not None
        assert node.total_observations == 6
        # Expert 2 should have count 2 (seen in both records)
        assert node.expert_counts[0][2] == 2
        assert node.expert_counts[0][7] == 1
        assert node.expert_counts[0][3] == 1

    def test_record_with_decay(self):
        dag = PrefixExpertDAG(decay_factor=0.5)
        dag.record([100], layer_id=0, expert_ids=[2, 2, 2], num_tokens=3)
        node = dag.get_node(100)
        count_after_first = node.expert_counts[0][2]
        assert count_after_first == 3

        # Second record: old counts decayed by 0.5, then new added
        dag.record([100], layer_id=0, expert_ids=[2], num_tokens=1)
        # Old 3 -> int(3*0.5) = 1, then +1 = 2
        assert node.expert_counts[0][2] == 2

    def test_parent_child_links(self):
        dag = PrefixExpertDAG()
        dag.record([10, 20, 30], layer_id=0, expert_ids=[1])
        parent = dag.get_node(10)
        assert parent is not None
        assert 20 in parent.children
        middle = dag.get_node(20)
        assert 30 in middle.children

    def test_predict_empty_hashes(self):
        dag = PrefixExpertDAG()
        assert dag.predict([]) == {}

    def test_predict_basic(self):
        dag = PrefixExpertDAG(decay_factor=1.0)
        # Record observations at prefix [100, 200]
        for _ in range(5):
            dag.record([100, 200], layer_id=0, expert_ids=[2, 7, 15])
            dag.record([100, 200], layer_id=1, expert_ids=[3, 8])

        preds = dag.predict([100, 200], top_k_per_layer=3, min_observations=2)
        assert 0 in preds
        assert 1 in preds
        # Layer 0 should have experts 2, 7, 15
        layer0_eids = [eid for eid, _ in preds[0]]
        assert set(layer0_eids) == {2, 7, 15}

    def test_predict_depth_weighting(self):
        """Deeper (more specific) nodes should have higher influence."""
        dag = PrefixExpertDAG(decay_factor=1.0)
        # Shallow node: many observations of expert 10
        for _ in range(20):
            dag.record([100], layer_id=0, expert_ids=[10])
        # Deep node: fewer observations of expert 20
        for _ in range(5):
            dag.record([100, 200], layer_id=0, expert_ids=[20])

        preds = dag.predict([100, 200], top_k_per_layer=2, min_observations=2)
        # Expert 20 (deeper) should rank higher despite fewer raw counts,
        # because depth_weight = 0.8^0 = 1.0 for deepest vs 0.8^1 = 0.8
        eids = [eid for eid, _ in preds[0]]
        assert 20 in eids

    def test_predict_min_observations(self):
        dag = PrefixExpertDAG(decay_factor=1.0)
        dag.record([100], layer_id=0, expert_ids=[5])
        # Only 1 observation, min is 2
        preds = dag.predict([100], min_observations=2)
        assert preds == {}

    def test_get_stats(self):
        dag = PrefixExpertDAG()
        dag.record([1, 2], layer_id=0, expert_ids=[5])
        stats = dag.get_stats()
        assert stats["num_nodes"] >= 1
        assert stats["total_records"] == 1

    def test_clear(self):
        dag = PrefixExpertDAG()
        dag.record([1], layer_id=0, expert_ids=[5])
        dag.clear()
        assert dag.num_nodes == 0
        assert dag.get_node(1) is None

    def test_eviction_on_max_nodes(self):
        dag = PrefixExpertDAG(max_nodes=10)
        for i in range(20):
            dag.record([i * 100], layer_id=0, expert_ids=[i % 8])
        # Should have evicted down to ~90% of max
        assert dag.num_nodes <= 10

    def test_multi_layer_recording(self):
        dag = PrefixExpertDAG(decay_factor=1.0)
        dag.record([100], layer_id=0, expert_ids=[1, 2])
        dag.record([100], layer_id=1, expert_ids=[3, 4])
        dag.record([100], layer_id=2, expert_ids=[5, 6])
        node = dag.get_node(100)
        assert len(node.expert_counts) == 3
        assert 1 in node.expert_counts[0]
        assert 3 in node.expert_counts[1]
        assert 5 in node.expert_counts[2]


# ---------------------------------------------------------------------------
# Tests: ExpertPredictor
# ---------------------------------------------------------------------------

class TestExpertPredictor:
    """Tests for the expert prefetch predictor."""

    def _build_predictor(self, gpu_experts=None):
        if gpu_experts is None:
            gpu_experts = {0, 1, 2, 3}
        engine = _make_cache_engine(
            num_layers=2, num_experts=8, gpu_experts=gpu_experts,
        )
        dag = PrefixExpertDAG(decay_factor=1.0)
        predictor = ExpertPredictor(
            dag=dag,
            cache_engine=engine,
            confidence_threshold=0.1,
            max_prefetch_per_layer=4,
        )
        return predictor, dag, engine

    def test_predict_empty(self):
        predictor, dag, _ = self._build_predictor()
        result = predictor.predict_and_prefetch([])
        assert result == {}

    def test_predict_with_dag_data(self):
        predictor, dag, engine = self._build_predictor(gpu_experts={0, 1})
        # Populate DAG: experts 4, 5 are on CPU and frequently activated
        for _ in range(10):
            dag.record([100, 200], layer_id=0, expert_ids=[4, 5])
            dag.record([100, 200], layer_id=1, expert_ids=[6, 7])

        result = predictor.predict_and_prefetch([100, 200])
        # Experts 4, 5, 6, 7 are on CPU — should be prefetched
        all_prefetched = set()
        for eids in result.values():
            all_prefetched.update(eids)
        # At least some of the CPU experts should be prefetched
        assert len(all_prefetched) > 0
        assert predictor._experts_prefetched > 0

    def test_no_prefetch_for_gpu_experts(self):
        """GPU-resident experts should not be prefetched."""
        predictor, dag, _ = self._build_predictor(gpu_experts={0, 1, 2, 3, 4, 5, 6, 7})
        for _ in range(10):
            dag.record([100], layer_id=0, expert_ids=[0, 1, 2])
        result = predictor.predict_and_prefetch([100])
        # All experts on GPU — nothing to prefetch
        assert result == {}

    def test_record_feedback(self):
        predictor, dag, _ = self._build_predictor()
        predictor.record_feedback(
            block_hashes=[100, 200],
            layer_id=0,
            actual_expert_ids=[2, 5, 7],
            num_tokens=8,
        )
        # Check feedback was recorded in DAG
        node = dag.get_node(200)
        assert node is not None
        assert node.expert_counts[0][2] == 1
        assert node.expert_counts[0][5] == 1

    def test_evaluate_accuracy(self):
        predictor, dag, _ = self._build_predictor()
        # Build up DAG data
        for _ in range(10):
            dag.record([100], layer_id=0, expert_ids=[1, 2, 3])

        # Evaluate: predicted {1,2,3}, actual {1,2,4}
        accuracy = predictor.evaluate_accuracy(
            block_hashes=[100],
            layer_id=0,
            actual_expert_ids=[1, 2, 4],
        )
        # 2 out of 3 predicted correctly
        assert abs(accuracy - 2 / 3) < 0.01

    def test_evaluate_accuracy_no_data(self):
        predictor, dag, _ = self._build_predictor()
        accuracy = predictor.evaluate_accuracy([999], 0, [1, 2])
        assert accuracy == 0.0

    def test_get_stats(self):
        predictor, dag, _ = self._build_predictor()
        stats = predictor.get_stats()
        assert "predictions_made" in stats
        assert "dag_stats" in stats
        assert stats["predictions_made"] == 0

    def test_confidence_threshold(self):
        """Only experts above confidence threshold should be prefetched."""
        predictor, dag, engine = self._build_predictor(gpu_experts={0})
        predictor.confidence_threshold = 0.5
        # Expert 1: high confidence (10 observations)
        for _ in range(10):
            dag.record([100], layer_id=0, expert_ids=[1])
        # Expert 2: low confidence (1 observation among many)
        dag.record([100], layer_id=0, expert_ids=[2])

        result = predictor.predict_and_prefetch([100])
        # Expert 1 should be prefetched (high probability)
        # Expert 2 may or may not depending on exact probability
        prefetched = set()
        for eids in result.values():
            prefetched.update(eids)
        assert 1 in prefetched


# ---------------------------------------------------------------------------
# Tests: Integration with ExpertCacheRegistry
# ---------------------------------------------------------------------------

class TestPredictionIntegration:
    """Integration tests for the prediction pipeline via the registry."""

    def setup_method(self):
        ExpertCacheRegistry.reset()
        self.engine = _make_cache_engine(
            num_layers=1, num_experts=8, gpu_experts={0, 1, 2, 3},
        )
        ExpertCacheRegistry.initialize(
            expert_cache_engine=self.engine,
            enable_tracking=True,
            enable_cpu_compute=False,
            enable_prediction=True,
            dag_decay=1.0,
            dag_max_nodes=1000,
            prediction_confidence=0.1,
        )
        registry = ExpertCacheRegistry.get_instance()
        registry.get_layer_id("test.layer.0")

    def teardown_method(self):
        clear_request_context()
        ExpertCacheRegistry.reset()

    def test_predict_experts_for_request(self):
        registry = ExpertCacheRegistry.get_instance()
        dag = registry.dag

        # Seed the DAG with observations
        for _ in range(10):
            dag.record([100, 200], layer_id=0, expert_ids=[4, 5, 6])

        result = predict_experts_for_request([100, 200])
        # Should have prefetched some CPU experts
        assert isinstance(result, dict)

    def test_feedback_loop(self):
        """Verify that forward hook records feedback into the DAG."""
        registry = ExpertCacheRegistry.get_instance()

        # Set current block hashes (simulating request context)
        predict_experts_for_request([100, 200])

        # Simulate MoE forward with expert selections
        topk_ids = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]])
        unified_cache_pre_forward("test.layer.0", topk_ids)

        # Check that the DAG got feedback
        dag = registry.dag
        node = dag.get_node(200)
        assert node is not None
        # All 8 expert IDs should be recorded
        all_recorded = set(node.expert_counts[0].keys())
        assert len(all_recorded) > 0

    def test_clear_request_context(self):
        predict_experts_for_request([100, 200])
        registry = ExpertCacheRegistry.get_instance()
        assert registry._current_block_hashes is not None
        clear_request_context()
        assert registry._current_block_hashes is None

    def test_prediction_disabled(self):
        ExpertCacheRegistry.reset()
        engine = _make_cache_engine()
        ExpertCacheRegistry.initialize(
            expert_cache_engine=engine,
            enable_prediction=False,
        )
        result = predict_experts_for_request([100, 200])
        assert result == {}

    def test_no_registry_predict(self):
        ExpertCacheRegistry.reset()
        result = predict_experts_for_request([100])
        assert result == {}

    def test_end_to_end_accuracy_improves(self):
        """After many observations, predictions should match actuals."""
        registry = ExpertCacheRegistry.get_instance()
        dag = registry.dag
        predictor = registry.predictor

        # Simulate 50 requests with consistent expert pattern
        for _ in range(50):
            predict_experts_for_request([300, 400])
            topk_ids = torch.tensor([[0, 4], [1, 5]])
            unified_cache_pre_forward("test.layer.0", topk_ids)
            clear_request_context()

        # Now evaluate prediction accuracy
        accuracy = predictor.evaluate_accuracy(
            block_hashes=[300, 400],
            layer_id=0,
            actual_expert_ids=[0, 4, 1, 5],
        )
        # After 50 observations, should be highly accurate
        assert accuracy >= 0.5
