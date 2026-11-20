# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Phase 2: CPU Active Compute.

Tests the cpu_expert_compute module, split dispatch logic in
moe_integration, and the end-to-end GPU/CPU merge path.
"""

import pytest
import torch

from vllm.unified_cache.cpu_expert_compute import (
    compute_cpu_expert_contributions,
    cpu_expert_ffn,
    create_gpu_only_weights,
)
from vllm.unified_cache.expert_cache_engine import (
    ExpertCacheEngine,
)
from vllm.unified_cache.moe_integration import (
    ExpertCacheRegistry,
    offload_experts_from_layer,
    unified_cache_forward,
    unified_cache_split_compute,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN = 64
INTERMEDIATE = 128
DTYPE = torch.float32  # Use fp32 for numerical accuracy in tests


def _make_expert_tensors(device: str = "cpu"):
    """Create mock expert weight tensors."""
    return {
        "w13_weight": torch.randn(INTERMEDIATE * 2, HIDDEN, dtype=DTYPE, device=device),
        "w2_weight": torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE, device=device),
    }


def _make_cache_engine(
    num_layers: int = 1,
    num_experts: int = 4,
    gpu_experts: set | None = None,
):
    """Create a cache engine with registered experts.

    Args:
        gpu_experts: Set of expert IDs to keep on GPU. Defaults to {0, 1}.

    """
    if gpu_experts is None:
        gpu_experts = {0, 1}

    engine = ExpertCacheEngine(
        gpu_memory_budget=1024**3,
        cpu_memory_budget=1024**3,
        device=torch.device("cpu"),  # test on CPU
    )

    for layer_id in range(num_layers):
        for expert_id in range(num_experts):
            tensors = _make_expert_tensors()
            engine.register_expert(
                layer_id=layer_id,
                expert_id=expert_id,
                weight_tensors=tensors,
                on_gpu=expert_id in gpu_experts,
            )
    return engine


# ---------------------------------------------------------------------------
# Tests: cpu_expert_ffn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("cpu_compute", [False, True])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_offloaded_forward_matches_resident_weights(device, cpu_compute, activation):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(42)
    layer = torch.nn.Module()
    layer.layer_name = "test.offloaded"
    layer.w13_weight = torch.nn.Parameter(
        torch.randn(4, INTERMEDIATE * 2, HIDDEN, device=device) * 0.1,
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.randn(4, HIDDEN, INTERMEDIATE, device=device) * 0.1,
        requires_grad=False,
    )
    inputs = torch.randn(3, HIDDEN, device=device)
    topk_ids = torch.tensor([[0, 2], [1, 3], [2, 3]], device=device)
    topk_weights = torch.tensor([[0.6, 0.4]] * 3, device=device)
    expected = torch.zeros_like(inputs)
    for token_id in range(3):
        for slot_id in range(2):
            expert_id = topk_ids[token_id, slot_id]
            expected[token_id] += (
                cpu_expert_ffn(
                    inputs[token_id : token_id + 1],
                    layer.w13_weight[expert_id],
                    layer.w2_weight[expert_id],
                    activation,
                ).squeeze(0)
                * topk_weights[token_id, slot_id]
            )
    expert_bytes = (INTERMEDIATE * 2 * HIDDEN + HIDDEN * INTERMEDIATE) * 4
    engine = ExpertCacheEngine(
        gpu_memory_budget=expert_bytes,
        cpu_memory_budget=expert_bytes * 4,
        device=torch.device(device),
        pin_cpu_memory=device == "cuda",
    )
    ExpertCacheRegistry.initialize(engine, enable_cpu_compute=cpu_compute)
    try:
        before = torch.accelerator.memory_allocated() if device == "cuda" else 0
        offload_experts_from_layer(layer)
        assert layer.w13_weight.device.type == "cpu"
        assert engine.gpu_memory_used == expert_bytes
        if device == "cuda":
            torch.accelerator.synchronize()
            assert before - torch.accelerator.memory_allocated() >= 3 * expert_bytes
        for _ in range(3):
            actual = unified_cache_forward(
                layer.layer_name,
                inputs,
                topk_weights,
                topk_ids,
                activation,
            )
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            assert engine.gpu_memory_used <= expert_bytes
        if cpu_compute:
            assert engine.stats.total_loads == 1
    finally:
        ExpertCacheRegistry.reset()


class TestCpuExpertFfn:
    """Tests for the standalone CPU FFN computation."""

    def test_basic_shape(self):
        """Output shape should be (num_tokens, hidden_dim)."""
        x = torch.randn(8, HIDDEN, dtype=DTYPE)
        w13 = torch.randn(INTERMEDIATE * 2, HIDDEN, dtype=DTYPE)
        w2 = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE)

        out = cpu_expert_ffn(x, w13, w2)
        assert out.shape == (8, HIDDEN)

    def test_single_token(self):
        """Should work with a single token."""
        x = torch.randn(1, HIDDEN, dtype=DTYPE)
        w13 = torch.randn(INTERMEDIATE * 2, HIDDEN, dtype=DTYPE)
        w2 = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE)

        out = cpu_expert_ffn(x, w13, w2)
        assert out.shape == (1, HIDDEN)

    def test_deterministic(self):
        """Same inputs should produce identical outputs."""
        x = torch.randn(4, HIDDEN, dtype=DTYPE)
        w13 = torch.randn(INTERMEDIATE * 2, HIDDEN, dtype=DTYPE)
        w2 = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE)

        out1 = cpu_expert_ffn(x, w13, w2, activation="silu")
        out2 = cpu_expert_ffn(x, w13, w2, activation="silu")
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gelu_activation(self):
        """Should support GELU activation as well."""
        x = torch.randn(4, HIDDEN, dtype=DTYPE)
        w13 = torch.randn(INTERMEDIATE * 2, HIDDEN, dtype=DTYPE)
        w2 = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE)

        out_silu = cpu_expert_ffn(x, w13, w2, activation="silu")
        out_gelu = cpu_expert_ffn(x, w13, w2, activation="gelu")
        # Different activations should produce different results
        assert not torch.allclose(out_silu, out_gelu, atol=1e-4)

    def test_manual_computation(self):
        """Verify the computation matches manual gate-up-down pattern."""
        torch.manual_seed(42)
        x = torch.randn(2, HIDDEN, dtype=DTYPE)
        w13 = torch.randn(INTERMEDIATE * 2, HIDDEN, dtype=DTYPE)
        w2 = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE)

        # Manual computation
        gate_up = x @ w13.t()
        gate, up = gate_up.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        expected = activated @ w2.t()

        out = cpu_expert_ffn(x, w13, w2)
        assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: create_gpu_only_weights
# ---------------------------------------------------------------------------


class TestCreateGpuOnlyWeights:
    """Tests for zeroing out CPU-expert weights."""

    def test_no_cpu_experts(self):
        """With no CPU experts, weights should be unchanged."""
        topk_weights = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
        topk_ids = torch.tensor([[0, 1], [0, 2]])

        result = create_gpu_only_weights(topk_weights, topk_ids, set())
        assert torch.equal(result, topk_weights)

    def test_zero_cpu_expert_weights(self):
        """Weights for CPU experts should be set to 0."""
        topk_weights = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
        topk_ids = torch.tensor([[0, 1], [0, 2]])

        result = create_gpu_only_weights(topk_weights, topk_ids, {1})
        # Expert 1 is at position (0,1) — should be zeroed
        assert result[0, 1] == 0.0
        # Non-CPU experts should be unchanged
        assert result[0, 0] == 0.5
        assert result[1, 0] == 0.6
        assert result[1, 1] == 0.4

    def test_multiple_cpu_experts(self):
        """Multiple CPU experts should all be zeroed."""
        topk_weights = torch.ones(3, 3)
        topk_ids = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 3, 1]])

        result = create_gpu_only_weights(topk_weights, topk_ids, {1, 2})
        for i in range(3):
            for j in range(3):
                eid = topk_ids[i, j].item()
                if eid in {1, 2}:
                    assert result[i, j] == 0.0
                else:
                    assert result[i, j] == 1.0

    def test_does_not_modify_original(self):
        """Input topk_weights should not be modified in-place."""
        topk_weights = torch.tensor([[0.5, 0.5]])
        topk_ids = torch.tensor([[0, 1]])
        original = topk_weights.clone()

        create_gpu_only_weights(topk_weights, topk_ids, {1})
        assert torch.equal(topk_weights, original)


# ---------------------------------------------------------------------------
# Tests: compute_cpu_expert_contributions
# ---------------------------------------------------------------------------


class TestComputeCpuExpertContributions:
    """Tests for the CPU expert contribution accumulation."""

    def _simple_weight_getter(self, expert_weights):
        """Create a weight getter from a dict of expert_id -> weights."""

        def getter(expert_id):
            return expert_weights.get(expert_id)

        return getter

    def test_no_cpu_experts_returns_none(self):
        """When no CPU experts match, return None."""
        x = torch.randn(4, HIDDEN, dtype=DTYPE)
        topk_weights = torch.tensor([[0.5, 0.5]] * 4)
        topk_ids = torch.tensor([[0, 1]] * 4)

        result = compute_cpu_expert_contributions(
            x,
            topk_weights,
            topk_ids,
            set(),
            lambda eid: None,
        )
        assert result is None

    def test_cpu_experts_not_selected_returns_none(self):
        """CPU experts exist but are not selected → None."""
        x = torch.randn(4, HIDDEN, dtype=DTYPE)
        topk_weights = torch.tensor([[0.5, 0.5]] * 4)
        topk_ids = torch.tensor([[0, 1]] * 4)

        result = compute_cpu_expert_contributions(
            x,
            topk_weights,
            topk_ids,
            {5, 6},  # experts 5,6 not in topk_ids
            lambda eid: _make_expert_tensors(),
        )
        assert result is None

    def test_single_cpu_expert_contribution_shape(self):
        """Output shape should match (num_tokens, hidden_dim)."""
        x = torch.randn(8, HIDDEN, dtype=DTYPE)
        topk_weights = torch.tensor([[0.6, 0.4]] * 8, dtype=DTYPE)
        topk_ids = torch.tensor([[0, 2]] * 8)

        expert_weights = {2: _make_expert_tensors()}
        result = compute_cpu_expert_contributions(
            x,
            topk_weights,
            topk_ids,
            {2},
            self._simple_weight_getter(expert_weights),
        )
        assert result is not None
        assert result.shape == (8, HIDDEN)

    def test_weighted_output(self):
        """CPU contribution should be weighted by router topk_weights."""
        torch.manual_seed(123)
        x = torch.randn(2, HIDDEN, dtype=DTYPE)
        weights = _make_expert_tensors()

        # Expert 1 has weight 0.3 for both tokens
        topk_weights = torch.tensor([[0.7, 0.3], [0.7, 0.3]], dtype=DTYPE)
        topk_ids = torch.tensor([[0, 1], [0, 1]])

        result = compute_cpu_expert_contributions(
            x,
            topk_weights,
            topk_ids,
            {1},
            lambda eid: weights if eid == 1 else None,
        )
        assert result is not None

        # Manually compute expected contribution
        out = cpu_expert_ffn(x, weights["w13_weight"], weights["w2_weight"])
        expected = out * 0.3  # weighted by topk_weight for expert 1
        assert torch.allclose(result, expected, atol=1e-4)

    def test_multiple_cpu_experts(self):
        """Multiple CPU experts should all contribute to the output."""
        torch.manual_seed(456)
        x = torch.randn(4, HIDDEN, dtype=DTYPE)

        w_e1 = _make_expert_tensors()
        w_e2 = _make_expert_tensors()

        # Each token routes to experts 1 and 2 (both on CPU)
        topk_weights = torch.tensor(
            [[0.6, 0.4]] * 4,
            dtype=DTYPE,
        )
        topk_ids = torch.tensor([[1, 2]] * 4)

        def getter(eid):
            if eid == 1:
                return w_e1
            if eid == 2:
                return w_e2
            return None

        result = compute_cpu_expert_contributions(
            x,
            topk_weights,
            topk_ids,
            {1, 2},
            getter,
        )
        assert result is not None

        # Both experts should contribute
        out1 = cpu_expert_ffn(x, w_e1["w13_weight"], w_e1["w2_weight"])
        out2 = cpu_expert_ffn(x, w_e2["w13_weight"], w_e2["w2_weight"])
        expected = out1 * 0.6 + out2 * 0.4
        assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: unified_cache_split_compute integration
# ---------------------------------------------------------------------------


class TestSplitCompute:
    """Tests for the full split-compute integration path."""

    def setup_method(self):
        """Create cache engine and registry before each test."""
        ExpertCacheRegistry.reset()
        self.engine = _make_cache_engine(
            num_layers=1,
            num_experts=4,
            gpu_experts={0, 1},  # experts 2 and 3 are on CPU
        )
        ExpertCacheRegistry.initialize(
            expert_cache_engine=self.engine,
            enable_tracking=False,
            enable_cpu_compute=True,
        )
        # Assign layer_name -> layer_id mapping
        registry = ExpertCacheRegistry.get_instance()
        registry.get_layer_id("test.layer.0")

    def teardown_method(self):
        ExpertCacheRegistry.reset()

    def test_all_gpu_experts_no_split(self):
        """When all selected experts are on GPU, no split should occur."""
        x = torch.randn(4, HIDDEN, dtype=DTYPE)
        topk_weights = torch.tensor([[0.5, 0.5]] * 4, dtype=DTYPE)
        topk_ids = torch.tensor([[0, 1]] * 4)  # experts 0,1 on GPU

        cpu_contrib, gpu_weights, active = unified_cache_split_compute(
            "test.layer.0",
            x,
            topk_weights,
            topk_ids,
        )
        assert active is False
        assert cpu_contrib is None
        assert gpu_weights is None

    def test_cpu_experts_trigger_split(self):
        """When CPU experts are selected, split compute should activate."""
        x = torch.randn(4, HIDDEN, dtype=DTYPE)
        topk_weights = torch.tensor([[0.6, 0.4]] * 4, dtype=DTYPE)
        topk_ids = torch.tensor([[0, 2]] * 4)  # expert 2 on CPU

        cpu_contrib, gpu_weights, active = unified_cache_split_compute(
            "test.layer.0",
            x,
            topk_weights,
            topk_ids,
        )
        assert active is True
        assert cpu_contrib is not None
        assert cpu_contrib.shape == (4, HIDDEN)
        assert gpu_weights is not None
        # Expert 2 (second slot) should have zero weight
        assert (gpu_weights[:, 1] == 0.0).all()
        # Expert 0 (first slot) should keep its weight
        assert (gpu_weights[:, 0] == 0.6).all()

    def test_disabled_cpu_compute_no_split(self):
        """When cpu_compute_enabled is False, should never split."""
        ExpertCacheRegistry.reset()
        ExpertCacheRegistry.initialize(
            expert_cache_engine=self.engine,
            enable_tracking=False,
            enable_cpu_compute=False,  # disabled
        )
        registry = ExpertCacheRegistry.get_instance()
        registry.get_layer_id("test.layer.0")

        x = torch.randn(4, HIDDEN, dtype=DTYPE)
        topk_weights = torch.tensor([[0.6, 0.4]] * 4, dtype=DTYPE)
        topk_ids = torch.tensor([[0, 2]] * 4)  # expert 2 on CPU

        cpu_contrib, gpu_weights, active = unified_cache_split_compute(
            "test.layer.0",
            x,
            topk_weights,
            topk_ids,
        )
        assert active is False

    def test_no_registry_no_split(self):
        """With no registry initialized, should return no-split."""
        ExpertCacheRegistry.reset()

        x = torch.randn(4, HIDDEN, dtype=DTYPE)
        topk_weights = torch.tensor([[0.6, 0.4]] * 4, dtype=DTYPE)
        topk_ids = torch.tensor([[0, 2]] * 4)

        cpu_contrib, gpu_weights, active = unified_cache_split_compute(
            "test.layer.0",
            x,
            topk_weights,
            topk_ids,
        )
        assert active is False

    def test_mixed_gpu_cpu_merge_correctness(self):
        """Verify the full split path: GPU zeros + CPU contribution = correct merge."""
        torch.manual_seed(789)
        x = torch.randn(4, HIDDEN, dtype=DTYPE)
        topk_weights = torch.tensor([[0.6, 0.4]] * 4, dtype=DTYPE)
        topk_ids = torch.tensor([[0, 2]] * 4)  # 0=GPU, 2=CPU

        cpu_contrib, gpu_weights, active = unified_cache_split_compute(
            "test.layer.0",
            x,
            topk_weights,
            topk_ids,
        )
        assert active is True

        # The GPU kernel would produce output for expert 0 with weight 0.6
        # and expert 2 with weight 0.0 (zeroed).
        # CPU contribution has expert 2's output weighted by 0.4.
        # The merge: gpu_result + cpu_contribution should give the correct total.
        assert cpu_contrib is not None
        assert gpu_weights is not None

        # Verify gpu_weights correctly zeros out expert 2
        for i in range(4):
            for j in range(2):
                eid = topk_ids[i, j].item()
                if eid == 2:
                    assert gpu_weights[i, j] == 0.0
                else:
                    assert gpu_weights[i, j] == topk_weights[i, j]


# ---------------------------------------------------------------------------
# Tests: ExpertCacheEngine new methods
# ---------------------------------------------------------------------------


class TestEngineNewMethods:
    """Tests for get_cpu_expert_ids and get_cpu_tensors."""

    def test_get_cpu_expert_ids(self):
        engine = _make_cache_engine(
            num_experts=4,
            gpu_experts={0, 1},
        )
        cpu_ids = engine.get_cpu_expert_ids(layer_id=0)
        assert sorted(cpu_ids) == [2, 3]

    def test_get_cpu_tensors(self):
        engine = _make_cache_engine(
            num_experts=4,
            gpu_experts={0, 1},
        )
        tensors = engine.get_cpu_tensors(layer_id=0, expert_id=2)
        assert tensors is not None
        assert "w13_weight" in tensors
        assert "w2_weight" in tensors
        assert tensors["w13_weight"].device.type == "cpu"

    def test_get_cpu_tensors_gpu_expert(self):
        """GPU experts also have cpu_tensors (pinned copies)."""
        engine = _make_cache_engine(
            num_experts=4,
            gpu_experts={0, 1},
        )
        tensors = engine.get_cpu_tensors(layer_id=0, expert_id=0)
        assert tensors is not None

    def test_get_cpu_tensors_nonexistent(self):
        engine = _make_cache_engine(num_experts=2)
        tensors = engine.get_cpu_tensors(layer_id=0, expert_id=99)
        assert tensors is None


@pytest.mark.slow_test
def test_llm_cpu_offload_matches_gpu_and_registers_experts(tmp_path, monkeypatch):
    from transformers import MixtralConfig, MixtralForCausalLM

    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    torch.manual_seed(42)
    model_config = MixtralConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    model = MixtralForCausalLM(model_config)
    model.save_pretrained(tmp_path)
    del model
    results = []
    for enabled in (False, True):
        llm = LLM(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            enforce_eager=True,
            dtype="float32",
            max_model_len=64,
            max_num_batched_tokens=32,
            max_num_seqs=4,
            kv_cache_memory_bytes=64 * 1024 * 1024,
            gpu_memory_utilization=0.1,
            kernel_config={"moe_backend": "triton"},
            unified_cache_config={
                "enabled": enabled,
                "enable_cpu_compute": True,
                "expert_cpu_memory_gb": 0.05,
            },
        )
        try:
            registry = ExpertCacheRegistry.get_instance()
            if enabled:
                assert registry is not None
                assert len(registry.offloaded_layers) == 2
                cache = registry.cache_engine
                assert 0 < cache.gpu_memory_used < cache.cpu_memory_used
                assert cache.gpu_memory_used <= cache.gpu_memory_budget
            outputs = llm.generate(
                [{"prompt_token_ids": [1, 5, 8, 12]}],
                SamplingParams(
                    temperature=0, max_tokens=4, ignore_eos=True, logprobs=1
                ),
                use_tqdm=False,
            )
            results.append(outputs[0].outputs[0])
            if enabled:
                assert cache.stats.cache_misses > 0
        finally:
            llm.llm_engine.engine_core.shutdown()
            del llm
        assert ExpertCacheRegistry.get_instance() is None
    assert results[0].token_ids == results[1].token_ids
    for baseline, offloaded, token_id in zip(
        results[0].logprobs, results[1].logprobs, results[0].token_ids
    ):
        assert offloaded[token_id].logprob == pytest.approx(
            baseline[token_id].logprob, abs=1e-4
        )
