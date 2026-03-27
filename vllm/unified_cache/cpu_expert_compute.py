# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU Expert Compute: Run MoE expert FFN on CPU for cache-evicted experts.

Instead of transferring cold expert weights back to GPU (3-4ms per expert),
this module runs the expert's feed-forward computation directly on CPU using
PyTorch matmul. For small batch sizes (few tokens routed to a cold expert),
CPU compute can be faster than the transfer-then-GPU-compute path.

The expert FFN follows the standard gated-MLP pattern used in MoE models:
    gate_up = x @ W_13^T       (fused gate + up projection)
    gate, up = split(gate_up)
    activated = act(gate) * up  (SiLU / GELU gating)
    output = activated @ W_2^T  (down projection)

This module is integrated via moe_integration.py's split-compute path.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


def cpu_expert_ffn(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """
    Compute a single expert's gated FFN on CPU.

    Args:
        x: Input hidden states on CPU. Shape: (num_tokens, hidden_dim).
        w13: Fused gate+up weight on CPU. Shape: (2 * intermediate_dim, hidden_dim).
        w2: Down projection weight on CPU. Shape: (hidden_dim, intermediate_dim).
        activation: Activation function name ("silu" or "gelu").

    Returns:
        Output tensor on CPU. Shape: (num_tokens, hidden_dim).
    """
    # Fused gate + up projection
    gate_up = torch.mm(x, w13.t())  # (num_tokens, 2 * intermediate_dim)

    # Split gate and up projections
    gate, up = gate_up.chunk(2, dim=-1)

    # Gated activation
    if activation == "silu":
        activated = F.silu(gate) * up
    elif activation == "gelu":
        activated = F.gelu(gate) * up
    else:
        activated = F.silu(gate) * up

    # Down projection
    output = torch.mm(activated, w2.t())  # (num_tokens, hidden_dim)
    return output


def compute_cpu_expert_contributions(
    x_gpu: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    cpu_expert_set: set[int],
    cpu_weight_getter: "callable",
    activation: str = "silu",
    w13_key: str = "w13_weight",
    w2_key: str = "w2_weight",
) -> Optional[torch.Tensor]:
    """
    Compute weighted contributions from CPU-resident experts.

    For each expert that is on CPU (not GPU), this function:
    1. Identifies which tokens are routed to that expert
    2. Copies those token hidden states to CPU
    3. Runs the expert FFN on CPU
    4. Weights the output by the router's topk_weights
    5. Accumulates results back on GPU

    Args:
        x_gpu: Input hidden states on GPU. Shape: (num_tokens, hidden_dim).
        topk_weights: Router weights. Shape: (num_tokens, top_k).
        topk_ids: Selected expert IDs. Shape: (num_tokens, top_k).
        cpu_expert_set: Set of expert IDs currently resident on CPU.
        cpu_weight_getter: Callable(expert_id) -> dict[str, Tensor] that
            returns the CPU weight tensors for a given expert.
        activation: Activation function name.
        w13_key: Key for the fused gate+up weight in the weight dict.
        w2_key: Key for the down projection weight in the weight dict.

    Returns:
        Tensor of shape (num_tokens, hidden_dim) on GPU containing the
        weighted sum of CPU expert outputs, or None if no CPU experts
        were selected.
    """
    if not cpu_expert_set:
        return None

    num_tokens, top_k = topk_ids.shape
    hidden_dim = x_gpu.shape[-1]

    # Find which (token, slot) pairs route to a CPU expert
    # Build a mask: True where topk_ids[i, j] is a CPU expert
    cpu_mask = torch.zeros_like(topk_ids, dtype=torch.bool)
    for eid in cpu_expert_set:
        cpu_mask |= (topk_ids == eid)

    if not cpu_mask.any():
        return None

    # Accumulate CPU contributions on GPU
    contribution = torch.zeros(
        num_tokens, hidden_dim, dtype=x_gpu.dtype, device=x_gpu.device
    )

    # Get unique CPU experts that are actually selected
    selected_cpu_experts = set()
    for eid in cpu_expert_set:
        if (topk_ids == eid).any():
            selected_cpu_experts.add(eid)

    if not selected_cpu_experts:
        return None

    # Move input to CPU once (shared across all CPU experts)
    # Only move tokens that actually need CPU compute
    token_needs_cpu = cpu_mask.any(dim=1)  # (num_tokens,)
    cpu_token_indices = token_needs_cpu.nonzero(as_tuple=True)[0]

    if cpu_token_indices.numel() == 0:
        return None

    x_cpu = x_gpu[cpu_token_indices].to("cpu", non_blocking=False)

    for expert_id in selected_cpu_experts:
        # Get CPU weights
        cpu_weights = cpu_weight_getter(expert_id)
        if cpu_weights is None:
            logger.warning("CPU weights missing for expert %d", expert_id)
            continue

        w13 = cpu_weights.get(w13_key)
        w2 = cpu_weights.get(w2_key)
        if w13 is None or w2 is None:
            logger.warning(
                "Missing w13/w2 for expert %d (keys: %s)",
                expert_id, list(cpu_weights.keys()),
            )
            continue

        # Find which tokens in our subset go to this expert
        expert_mask = (topk_ids[cpu_token_indices] == expert_id)  # (subset, top_k)
        tokens_for_expert = expert_mask.any(dim=1)  # (subset,)

        if not tokens_for_expert.any():
            continue

        # Get local indices within cpu_token_indices that route here
        local_indices = tokens_for_expert.nonzero(as_tuple=True)[0]
        x_subset = x_cpu[local_indices]

        # Compute FFN on CPU
        out_cpu = cpu_expert_ffn(x_subset, w13, w2, activation)

        # Compute router weight for this expert's contribution
        # Sum topk_weights across all slots where this expert was selected
        slot_weights = (
            topk_weights[cpu_token_indices[local_indices]]
            * expert_mask[local_indices].float()
        )  # (n, top_k)
        weight_per_token = slot_weights.sum(dim=1, keepdim=True)  # (n, 1)
        weight_per_token_cpu = weight_per_token.to("cpu", non_blocking=False)

        # Weight and move back to GPU
        weighted_out = (out_cpu * weight_per_token_cpu).to(
            x_gpu.device, non_blocking=False
        )

        # Scatter-add back to the global contribution tensor
        global_indices = cpu_token_indices[local_indices]
        contribution[global_indices] += weighted_out

    return contribution


def create_gpu_only_weights(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    cpu_expert_set: set[int],
) -> torch.Tensor:
    """
    Create modified topk_weights with CPU-expert contributions zeroed out.

    The GPU fused kernel will see the same topk_ids, but weights for
    CPU-resident experts will be zero, so their (dummy) GPU computation
    contributes nothing to the output. The real CPU contributions are
    added separately by compute_cpu_expert_contributions().

    Args:
        topk_weights: Original router weights. Shape: (num_tokens, top_k).
        topk_ids: Selected expert IDs. Shape: (num_tokens, top_k).
        cpu_expert_set: Set of expert IDs currently on CPU.

    Returns:
        Modified topk_weights with CPU expert weights set to 0.
    """
    if not cpu_expert_set:
        return topk_weights

    gpu_weights = topk_weights.clone()
    for eid in cpu_expert_set:
        mask = (topk_ids == eid)
        if mask.any():
            gpu_weights[mask] = 0.0

    return gpu_weights
