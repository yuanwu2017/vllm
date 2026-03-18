# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the unified cache system.

Tests the core components: ExpertCacheEngine, ExpertEvictor,
UnifiedCacheManager, and the MoE integration hooks.
"""

import time

import pytest
import torch

from vllm.unified_cache.expert_cache_engine import (
    ExpertCacheEngine,
    ExpertCacheEntry,
    ExpertLocation,
)
from vllm.unified_cache.expert_evictor import (
    CacheItemType,
    EvictionCandidate,
    ExpertEvictionPolicy,
    UnifiedEvictionPolicy,
)
from vllm.unified_cache.expert_tracker import (
    ExpertActivationTracker,
)
from vllm.unified_cache.unified_cache_manager import (
    UnifiedCacheManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_expert_tensors(
    hidden_size: int = 256,
    intermediate_size: int = 512,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create mock expert weight tensors (w13 = gate+up fused, w2 = down)."""
    return {
        "w13_weight": torch.randn(
            intermediate_size * 2, hidden_size, dtype=dtype, device=device
        ),
        "w2_weight": torch.randn(
            hidden_size, intermediate_size, dtype=dtype, device=device
        ),
    }


def _expert_memory_bytes(hidden_size: int = 256, intermediate_size: int = 512) -> int:
    """Calculate expected memory for one expert in float16."""
    w13 = intermediate_size * 2 * hidden_size * 2  # float16 = 2 bytes
    w2 = hidden_size * intermediate_size * 2
    return w13 + w2


# ---------------------------------------------------------------------------
# ExpertCacheEngine Tests
# ---------------------------------------------------------------------------

class TestExpertCacheEngine:
    """Tests for the ExpertCacheEngine class."""

    def _make_engine(self, gpu_budget_mb: int = 100, cpu_budget_mb: int = 200):
        device = torch.device("cpu")  # Use CPU for testing without GPU
        return ExpertCacheEngine(
            gpu_memory_budget=gpu_budget_mb * 1024 * 1024,
            cpu_memory_budget=cpu_budget_mb * 1024 * 1024,
            device=device,
            ema_decay=0.9,
            pin_cpu_memory=False,  # No pinning on CPU-only tests
        )

    def test_register_expert(self):
        engine = self._make_engine()
        tensors = _make_expert_tensors(device="cpu")
        engine.register_expert(0, 0, tensors, on_gpu=True)

        assert (0, 0) in engine._cache
        entry = engine._cache[(0, 0)]
        assert entry.location == ExpertLocation.GPU
        assert entry.memory_bytes > 0
        assert (0, 0) in engine._gpu_lru

    def test_register_multiple_experts(self):
        engine = self._make_engine()
        for expert_id in range(10):
            tensors = _make_expert_tensors(device="cpu")
            engine.register_expert(0, expert_id, tensors, on_gpu=True)

        assert len(engine._cache) == 10
        assert len(engine._gpu_lru) == 10

    def test_get_expert_hit(self):
        engine = self._make_engine()
        tensors = _make_expert_tensors(device="cpu")
        engine.register_expert(0, 0, tensors, on_gpu=True)

        result = engine.get_expert(0, 0)
        assert result is not None
        assert "w13_weight" in result
        assert "w2_weight" in result
        assert engine.stats.cache_hits == 1

    def test_get_expert_miss(self):
        engine = self._make_engine()
        tensors = _make_expert_tensors(device="cpu")
        engine.register_expert(0, 0, tensors, on_gpu=False)

        result = engine.get_expert(0, 0)
        assert result is None
        assert engine.stats.cache_misses == 1

    def test_get_nonexistent_expert(self):
        engine = self._make_engine()
        result = engine.get_expert(0, 999)
        assert result is None

    def test_evict_expert(self):
        engine = self._make_engine()
        tensors = _make_expert_tensors(device="cpu")
        mem = sum(t.nelement() * t.element_size() for t in tensors.values())
        engine.register_expert(0, 0, tensors, on_gpu=True)

        initial_gpu = engine.gpu_memory_used
        assert initial_gpu > 0

        result = engine.evict_expert(0, 0, sync=True)
        assert result is True
        assert engine._cache[(0, 0)].location == ExpertLocation.CPU
        assert engine.gpu_memory_used == initial_gpu - mem
        assert (0, 0) not in engine._gpu_lru
        assert engine.stats.total_evictions == 1

    def test_lru_order(self):
        engine = self._make_engine()
        for eid in range(5):
            tensors = _make_expert_tensors(device="cpu")
            engine.register_expert(0, eid, tensors, on_gpu=True)

        # Access experts in order: 0, 1, 2
        engine.get_expert(0, 0)
        engine.get_expert(0, 1)
        engine.get_expert(0, 2)

        # LRU should evict expert 3 or 4 first (least recently used)
        lru_keys = list(engine._gpu_lru.keys())
        # First in LRU = least recently used
        assert lru_keys[0] in [(0, 3), (0, 4)]

    def test_hit_rate(self):
        engine = self._make_engine()
        tensors = _make_expert_tensors(device="cpu")
        engine.register_expert(0, 0, tensors, on_gpu=True)
        engine.register_expert(0, 1, tensors, on_gpu=False)

        engine.get_expert(0, 0)  # hit
        engine.get_expert(0, 0)  # hit
        engine.get_expert(0, 1)  # miss

        assert engine.stats.cache_hits == 2
        assert engine.stats.cache_misses == 1
        assert abs(engine.hit_rate - 2/3) < 0.01

    def test_gpu_utilization(self):
        engine = self._make_engine(gpu_budget_mb=10)
        tensors = _make_expert_tensors(device="cpu")
        engine.register_expert(0, 0, tensors, on_gpu=True)

        assert engine.gpu_utilization > 0
        assert engine.gpu_utilization <= 1.0

    def test_get_gpu_expert_ids(self):
        engine = self._make_engine()
        for eid in range(5):
            tensors = _make_expert_tensors(device="cpu")
            on_gpu = eid < 3  # First 3 on GPU
            engine.register_expert(0, eid, tensors, on_gpu=on_gpu)

        gpu_ids = engine.get_gpu_expert_ids(0)
        assert sorted(gpu_ids) == [0, 1, 2]

    def test_get_all_gpu_expert_ids(self):
        engine = self._make_engine()
        for lid in range(2):
            for eid in range(3):
                tensors = _make_expert_tensors(device="cpu")
                engine.register_expert(lid, eid, tensors, on_gpu=True)

        all_ids = engine.get_all_gpu_expert_ids()
        assert 0 in all_ids
        assert 1 in all_ids
        assert len(all_ids[0]) == 3
        assert len(all_ids[1]) == 3

    def test_update_gpu_budget_triggers_eviction(self):
        engine = self._make_engine(gpu_budget_mb=100)
        for eid in range(10):
            tensors = _make_expert_tensors(device="cpu")
            engine.register_expert(0, eid, tensors, on_gpu=True)

        initial_count = len(engine._gpu_lru)

        # Shrink budget to trigger evictions
        engine.update_gpu_budget(engine.gpu_memory_used // 2)

        assert len(engine._gpu_lru) < initial_count
        assert engine.gpu_memory_used <= engine.gpu_memory_budget

    def test_eviction_candidates(self):
        engine = self._make_engine()
        for eid in range(5):
            tensors = _make_expert_tensors(device="cpu")
            engine.register_expert(0, eid, tensors, on_gpu=True)

        # Access some experts to build up EMA
        for _ in range(10):
            engine.get_expert(0, 0)  # High frequency
            engine.get_expert(0, 1)  # High frequency

        candidates = engine.get_eviction_candidates(count=3)
        assert len(candidates) == 3
        # Cold experts (2, 3, 4) should be candidates
        candidate_ids = {c[1] for c in candidates}
        assert 0 not in candidate_ids or 1 not in candidate_ids


# ---------------------------------------------------------------------------
# ExpertEvictor Tests
# ---------------------------------------------------------------------------

class TestExpertEvictionPolicy:

    def test_score_expert(self):
        policy = ExpertEvictionPolicy(
            hidden_size=256, intermediate_size=512
        )

        # Hot expert should have higher score
        hot_score = policy.score_expert(
            ema_access_rate=0.8, memory_bytes=1024 * 1024
        )
        cold_score = policy.score_expert(
            ema_access_rate=0.1, memory_bytes=1024 * 1024
        )
        assert hot_score > cold_score

    def test_score_with_time_decay(self):
        policy = ExpertEvictionPolicy(
            hidden_size=256, intermediate_size=512
        )

        recent = policy.score_expert(
            ema_access_rate=0.5, memory_bytes=1024 * 1024,
            time_since_last_access=0.0,
        )
        old = policy.score_expert(
            ema_access_rate=0.5, memory_bytes=1024 * 1024,
            time_since_last_access=10.0,
        )
        assert recent > old

    def test_score_kv_block(self):
        policy = ExpertEvictionPolicy(hidden_size=256, intermediate_size=512)

        high_reuse = policy.score_kv_block(
            reuse_probability=0.9, seq_len=128,
            head_dim=64, num_heads=8, memory_bytes=1024,
        )
        low_reuse = policy.score_kv_block(
            reuse_probability=0.1, seq_len=128,
            head_dim=64, num_heads=8, memory_bytes=1024,
        )
        assert high_reuse > low_reuse

    def test_select_eviction_candidates(self):
        policy = ExpertEvictionPolicy(hidden_size=256, intermediate_size=512)

        candidates = [
            EvictionCandidate(
                item_type=CacheItemType.EXPERT_WEIGHT,
                key=(0, i),
                score=float(i),  # Higher id = higher score
                memory_bytes=1024 * 1024,
                last_access_time=0.0,
            )
            for i in range(5)
        ]

        to_evict = policy.select_eviction_candidates(
            candidates, target_bytes=2 * 1024 * 1024
        )
        assert len(to_evict) == 2
        # Lowest-scored should be evicted first
        assert to_evict[0].key == (0, 0)
        assert to_evict[1].key == (0, 1)


class TestUnifiedEvictionPolicy:

    def test_cross_type_eviction(self):
        expert_policy = ExpertEvictionPolicy(hidden_size=256, intermediate_size=512)
        unified = UnifiedEvictionPolicy(expert_policy)

        kv_candidates = [
            EvictionCandidate(
                item_type=CacheItemType.KV_BLOCK,
                key=(0, 0),
                score=0.5,
                memory_bytes=4096,
                last_access_time=0.0,
            ),
        ]
        expert_candidates = [
            EvictionCandidate(
                item_type=CacheItemType.EXPERT_WEIGHT,
                key=(0, 0),
                score=0.1,
                memory_bytes=1024 * 1024,
                last_access_time=0.0,
            ),
        ]

        evictions = unified.select_evictions(
            kv_candidates, expert_candidates,
            target_bytes=4096,
        )
        # Lowest score should be evicted first
        assert len(evictions) >= 1
        assert evictions[0].score <= 0.5

    def test_weighting(self):
        expert_policy = ExpertEvictionPolicy(hidden_size=256, intermediate_size=512)
        # High expert weight means experts are more valuable → harder to evict
        unified = UnifiedEvictionPolicy(
            expert_policy, kv_weight=1.0, expert_weight=10.0
        )

        kv = EvictionCandidate(
            item_type=CacheItemType.KV_BLOCK,
            key=(0, 0), score=1.0, memory_bytes=4096, last_access_time=0.0,
        )
        expert = EvictionCandidate(
            item_type=CacheItemType.EXPERT_WEIGHT,
            key=(0, 0), score=1.0, memory_bytes=4096, last_access_time=0.0,
        )

        kv_unified = unified.score(kv)
        expert_unified = unified.score(expert)
        # Expert should have higher unified score (harder to evict)
        assert expert_unified > kv_unified


# ---------------------------------------------------------------------------
# ExpertActivationTracker Tests
# ---------------------------------------------------------------------------

class TestExpertActivationTracker:

    def setup_method(self):
        ExpertActivationTracker.reset_instance()

    def test_singleton(self):
        t1 = ExpertActivationTracker.get_instance()
        t2 = ExpertActivationTracker.get_instance()
        assert t1 is t2

    def test_disabled_by_default(self):
        tracker = ExpertActivationTracker.get_instance()
        assert not tracker.enabled

    def test_record_when_disabled(self):
        tracker = ExpertActivationTracker.get_instance()
        topk = torch.tensor([[0, 1], [2, 3]])
        tracker.record(layer_id=0, topk_ids=topk)
        assert len(tracker._records) == 0

    def test_record_when_enabled(self):
        tracker = ExpertActivationTracker.get_instance()
        tracker.enable()
        topk = torch.tensor([[0, 1], [2, 3]])
        tracker.record(layer_id=0, topk_ids=topk)
        assert len(tracker._records) == 1
        assert tracker._total_tokens == 2

    def test_get_hot_experts(self):
        tracker = ExpertActivationTracker.get_instance()
        tracker.enable()

        # Expert 0 appears 5 times, expert 1 appears 3 times, expert 2 once
        for _ in range(5):
            tracker.record(0, torch.tensor([[0, 1]]))
        for _ in range(3):
            tracker.record(0, torch.tensor([[1, 2]]))
        tracker.record(0, torch.tensor([[2, 3]]))

        hot = tracker.get_hot_experts(0, top_k=2)
        assert hot[0][0] in (0, 1)  # Most frequent

    def test_analyze(self):
        tracker = ExpertActivationTracker.get_instance()
        tracker.enable()

        for i in range(100):
            expert_a = i % 4
            expert_b = (i + 1) % 4
            tracker.record(0, torch.tensor([[expert_a, expert_b]]))

        analysis = tracker.analyze()
        assert analysis.total_tokens == 100
        assert analysis.total_events == 100
        assert 0 in analysis.expert_frequency
        assert analysis.temporal_locality > 0

    def test_clear(self):
        tracker = ExpertActivationTracker.get_instance()
        tracker.enable()
        tracker.record(0, torch.tensor([[0, 1]]))
        tracker.clear()
        assert len(tracker._records) == 0
        assert tracker._total_tokens == 0


# ---------------------------------------------------------------------------
# UnifiedCacheManager Tests
# ---------------------------------------------------------------------------

class TestUnifiedCacheManager:

    def _make_manager(
        self, total_budget_mb: int = 1000, kv_ratio: float = 0.5
    ):
        device = torch.device("cpu")
        expert_budget = int(total_budget_mb * 1024 * 1024 * (1 - kv_ratio))
        cpu_budget = 500 * 1024 * 1024

        expert_cache = ExpertCacheEngine(
            gpu_memory_budget=expert_budget,
            cpu_memory_budget=cpu_budget,
            device=device,
            pin_cpu_memory=False,
        )

        return UnifiedCacheManager(
            total_cache_budget=total_budget_mb * 1024 * 1024,
            expert_cache_engine=expert_cache,
            initial_kv_ratio=kv_ratio,
            min_kv_ratio=0.2,
            max_kv_ratio=0.8,
            rebalance_interval=0.0,  # Allow immediate rebalancing in tests
            device=device,
        )

    def test_initial_budgets(self):
        manager = self._make_manager(total_budget_mb=1000, kv_ratio=0.6)
        assert abs(manager.kv_budget - 600 * 1024 * 1024) < 1024
        assert abs(manager.expert_budget - 400 * 1024 * 1024) < 1024

    def test_pressure_reporting(self):
        manager = self._make_manager()
        manager.report_kv_pressure(utilization=0.9, miss_rate=0.3)
        assert manager._kv_pressure_ema > 0

        manager.report_expert_pressure()
        # Expert cache is empty, so pressure should be based on utilization=0
        assert manager._expert_pressure_ema >= 0

    def test_rebalance(self):
        manager = self._make_manager(kv_ratio=0.5)

        # Simulate high KV pressure, low expert pressure
        for _ in range(20):
            manager.report_kv_pressure(utilization=0.95, miss_rate=0.5)
            manager.report_expert_pressure()

        rebalanced = manager.maybe_rebalance()
        if rebalanced:
            # KV ratio should increase (more budget for KV)
            assert manager.kv_ratio > 0.5

    def test_get_stats(self):
        manager = self._make_manager()
        stats = manager.get_stats()
        assert stats.total_gpu_memory_budget > 0
        assert 0 <= stats.kv_budget_ratio <= 1

    def test_get_expert_cache_state(self):
        manager = self._make_manager()
        state = manager.get_expert_cache_state()
        assert "gpu_experts" in state
        assert "hit_rate" in state
        assert "gpu_utilization" in state

    def test_request_gpu_memory(self):
        manager = self._make_manager(total_budget_mb=100, kv_ratio=0.5)
        # Small request should succeed
        assert manager.request_gpu_memory("kv", 1024)


# ---------------------------------------------------------------------------
# Integration Smoke Test
# ---------------------------------------------------------------------------

class TestIntegration:
    """Smoke test for the full integration flow."""

    def test_full_flow(self):
        """Test register → get → track → evict → reload cycle."""
        device = torch.device("cpu")

        # Init engine
        engine = ExpertCacheEngine(
            gpu_memory_budget=50 * 1024 * 1024,
            cpu_memory_budget=200 * 1024 * 1024,
            device=device,
            pin_cpu_memory=False,
        )

        # Register experts
        for layer in range(2):
            for expert in range(8):
                tensors = _make_expert_tensors(device="cpu")
                engine.register_expert(layer, expert, tensors, on_gpu=True)

        assert len(engine._cache) == 16
        assert len(engine._gpu_lru) == 16

        # Access some experts
        for _ in range(10):
            for expert in [0, 1]:  # Hot experts
                engine.get_expert(0, expert)
                engine.get_expert(1, expert)

        # Cold experts (2-7) should have lower EMA
        hot_entry = engine._cache[(0, 0)]
        cold_entry = engine._cache[(0, 5)]
        assert hot_entry.ema_access_rate > cold_entry.ema_access_rate

        # Evict a cold expert
        evicted = engine.evict_expert(0, 7, sync=True)
        assert evicted
        assert engine._cache[(0, 7)].location == ExpertLocation.CPU

        # Verify evicted expert returns None
        result = engine.get_expert(0, 7)
        assert result is None

        # Hit rate should reflect the pattern
        assert engine.hit_rate > 0
        assert engine.stats.total_evictions == 1

    def test_unified_manager_with_experts(self):
        """Test UnifiedCacheManager with registered experts."""
        device = torch.device("cpu")

        expert_cache = ExpertCacheEngine(
            gpu_memory_budget=20 * 1024 * 1024,
            cpu_memory_budget=100 * 1024 * 1024,
            device=device,
            pin_cpu_memory=False,
        )

        manager = UnifiedCacheManager(
            total_cache_budget=40 * 1024 * 1024,
            expert_cache_engine=expert_cache,
            initial_kv_ratio=0.5,
            rebalance_interval=0.0,
            device=device,
        )

        # Register some experts
        for expert in range(4):
            tensors = _make_expert_tensors(hidden_size=64, intermediate_size=128,
                                           device="cpu")
            expert_cache.register_expert(0, expert, tensors, on_gpu=True)

        stats = manager.get_stats()
        assert stats.expert_gpu_count == 4
        assert stats.expert_total_count == 4
        assert stats.expert_gpu_memory_used > 0
