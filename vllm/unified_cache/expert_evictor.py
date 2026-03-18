# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ExpertEvictor: Eviction policy for MoE expert weights.

Implements a score-based eviction policy that considers:
- Access frequency (EMA-smoothed)
- Compute cost saved by keeping expert on GPU
- Memory footprint of the expert

The unified scoring function allows direct comparison between
KV cache blocks and expert weights for cross-type eviction decisions.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from vllm.logger import init_logger

logger = init_logger(__name__)


class CacheItemType(Enum):
    """Type of item in the unified cache."""
    KV_BLOCK = "kv_block"
    EXPERT_WEIGHT = "expert_weight"


@dataclass
class EvictionCandidate:
    """A candidate for eviction from GPU memory."""
    item_type: CacheItemType
    # For KV blocks: (request_id, block_idx); for experts: (layer_id, expert_id)
    key: tuple[int, int]
    score: float  # Higher = more valuable to keep
    memory_bytes: int
    last_access_time: float


class EvictionScorer(Protocol):
    """Protocol for computing eviction scores."""
    def score(self, candidate: EvictionCandidate) -> float: ...


class ExpertEvictionPolicy:
    """
    Eviction policy for expert weights using a multi-factor scoring function.

    Score formula:
        score = ema_access_rate × compute_cost_factor / memory_bytes

    Where:
        - ema_access_rate: Exponential moving average of access frequency.
          Higher means the expert is accessed more frequently.
        - compute_cost_factor: FLOPs saved by keeping this expert on GPU.
          For MoE FFN: 2 × hidden_size × intermediate_size × 3 (gate+up+down).
        - memory_bytes: GPU memory consumed by this expert.

    Higher score = more valuable to keep on GPU.
    Evict lowest-scored experts first.
    """

    def __init__(
        self,
        hidden_size: int = 0,
        intermediate_size: int = 0,
        ema_decay: float = 0.95,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ema_decay = ema_decay

        # Compute cost is proportional to the FLOPs of the expert FFN
        # For a standard MoE FFN: gate_proj + up_proj + down_proj
        # FLOPs ≈ 2 * hidden * intermediate * 3 (gate+up fused as w13, down as w2)
        if hidden_size > 0 and intermediate_size > 0:
            self.compute_cost_factor = float(
                2 * hidden_size * intermediate_size * 3
            )
        else:
            self.compute_cost_factor = 1.0

    def score_expert(
        self,
        ema_access_rate: float,
        memory_bytes: int,
        time_since_last_access: float = 0.0,
    ) -> float:
        """
        Compute eviction score for an expert.

        Args:
            ema_access_rate: EMA of access frequency (0.0 = never, 1.0 = every step).
            memory_bytes: GPU memory consumed by this expert.
            time_since_last_access: Seconds since last access.

        Returns:
            Score value. Higher = more valuable to keep on GPU.
        """
        if memory_bytes <= 0:
            return float("inf")

        # Time decay factor: recent accesses matter more
        time_decay = 1.0 / (1.0 + time_since_last_access)

        score = (
            ema_access_rate
            * self.compute_cost_factor
            * time_decay
            / memory_bytes
        )
        return score

    def score_kv_block(
        self,
        reuse_probability: float,
        seq_len: int,
        head_dim: int,
        num_heads: int,
        memory_bytes: int,
    ) -> float:
        """
        Compute eviction score for a KV cache block.

        This enables direct comparison with expert scores for unified eviction.

        Args:
            reuse_probability: Probability this block will be reused (from prefix matching).
            seq_len: Sequence length covered by this block.
            head_dim: Attention head dimension.
            num_heads: Number of attention heads.
            memory_bytes: GPU memory consumed by this block.

        Returns:
            Score value. Higher = more valuable to keep on GPU.
        """
        if memory_bytes <= 0:
            return float("inf")

        # Compute cost of recomputing attention for this block
        # FLOPs ≈ 2 * seq_len * head_dim * num_heads (Q*K + attn*V)
        attention_flops = float(2 * seq_len * head_dim * num_heads * 2)

        score = reuse_probability * attention_flops / memory_bytes
        return score

    def select_eviction_candidates(
        self,
        candidates: list[EvictionCandidate],
        target_bytes: int,
    ) -> list[EvictionCandidate]:
        """
        Select candidates to evict to free at least target_bytes of GPU memory.

        Uses a greedy approach: evict lowest-scored items first until
        the target is reached.

        Args:
            candidates: List of eviction candidates with scores.
            target_bytes: Minimum bytes to free.

        Returns:
            Ordered list of candidates to evict.
        """
        # Sort by score ascending (lowest = evict first)
        sorted_candidates = sorted(candidates, key=lambda c: c.score)

        to_evict = []
        freed = 0
        for candidate in sorted_candidates:
            if freed >= target_bytes:
                break
            to_evict.append(candidate)
            freed += candidate.memory_bytes

        return to_evict


class UnifiedEvictionPolicy:
    """
    Unified eviction policy for both KV cache blocks and expert weights.

    This policy computes comparable scores for both types of cache items,
    enabling cross-type eviction decisions. When GPU memory pressure is high,
    it can decide whether to evict a KV block or an expert weight based
    on which provides less value per byte of GPU memory.
    """

    def __init__(
        self,
        expert_policy: ExpertEvictionPolicy,
        kv_weight: float = 1.0,
        expert_weight: float = 1.0,
    ):
        """
        Args:
            expert_policy: Policy for scoring expert weights.
            kv_weight: Multiplicative weight for KV block scores.
            expert_weight: Multiplicative weight for expert scores.
        """
        self.expert_policy = expert_policy
        self.kv_weight = kv_weight
        self.expert_weight = expert_weight

    def score(self, candidate: EvictionCandidate) -> float:
        """
        Compute unified score for any cache item.

        Applies type-specific weighting to enable balancing between
        KV and expert eviction preferences.
        """
        if candidate.item_type == CacheItemType.EXPERT_WEIGHT:
            return candidate.score * self.expert_weight
        elif candidate.item_type == CacheItemType.KV_BLOCK:
            return candidate.score * self.kv_weight
        return candidate.score

    def select_evictions(
        self,
        kv_candidates: list[EvictionCandidate],
        expert_candidates: list[EvictionCandidate],
        target_bytes: int,
    ) -> list[EvictionCandidate]:
        """
        Select items to evict from a mixed pool of KV blocks and experts.

        Args:
            kv_candidates: KV block candidates with pre-computed scores.
            expert_candidates: Expert weight candidates with pre-computed scores.
            target_bytes: Minimum bytes to free.

        Returns:
            Ordered list of items to evict (mixed KV and expert).
        """
        # Apply unified scoring
        all_candidates = []
        for c in kv_candidates:
            unified_score = self.score(c)
            all_candidates.append((unified_score, c))
        for c in expert_candidates:
            unified_score = self.score(c)
            all_candidates.append((unified_score, c))

        # Sort by unified score ascending (lowest = evict first)
        all_candidates.sort(key=lambda x: x[0])

        to_evict = []
        freed = 0
        for _, candidate in all_candidates:
            if freed >= target_bytes:
                break
            to_evict.append(candidate)
            freed += candidate.memory_bytes

        return to_evict
