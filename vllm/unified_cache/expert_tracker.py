# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert activation instrumentation for collecting MoE routing statistics.

This module provides hooks to record which experts are selected for each token
during MoE forward passes. The collected traces can be used for:
1. Offline analysis of expert usage patterns (paper motivation)
2. Online expert cache warm-up and prefetch decisions
3. Validating the effectiveness of the unified cache

Usage:
    # Enable instrumentation
    tracker = ExpertActivationTracker.get_instance()
    tracker.enable()

    # In MoE forward pass (automatically hooked)
    # ... normal inference ...

    # Export traces
    tracker.save_traces("expert_traces.jsonl")

    # Analyze
    analysis = tracker.analyze()
    print(analysis.expert_frequency)
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ExpertActivationRecord:
    """Single activation record for an expert selection event."""
    timestamp: float
    layer_id: int
    expert_ids: list[int]  # top-k expert IDs selected
    num_tokens: int
    # Optional context
    request_id: Optional[str] = None
    batch_idx: Optional[int] = None


@dataclass
class ExpertUsageAnalysis:
    """Analysis results from expert activation traces."""
    # Per-layer expert frequency: layer_id -> {expert_id: count}
    expert_frequency: dict[int, dict[int, int]] = field(default_factory=dict)
    # Global expert frequency across all layers
    global_frequency: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    # Total tokens processed
    total_tokens: int = 0
    # Total activation events
    total_events: int = 0
    # Number of unique experts activated
    unique_experts_per_layer: dict[int, int] = field(default_factory=dict)
    # Top-K coverage: what fraction of traffic is handled by top-K% of experts
    top_k_coverage: dict[int, float] = field(default_factory=dict)
    # Temporal locality: fraction of consecutive activations hitting same experts
    temporal_locality: float = 0.0
    # Per-layer skewness (Gini coefficient of expert usage)
    gini_per_layer: dict[int, float] = field(default_factory=dict)


class ExpertActivationTracker:
    """
    Singleton tracker for MoE expert activation patterns.

    Collects per-token expert selection data during inference for
    offline analysis and online cache optimization.
    """

    _instance: Optional["ExpertActivationTracker"] = None

    @classmethod
    def get_instance(cls) -> "ExpertActivationTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def __init__(self):
        self._enabled = False
        self._records: list[ExpertActivationRecord] = []
        self._max_records = 1_000_000  # Limit memory usage
        # Per-layer running counters for online stats
        self._layer_counters: dict[int, Counter] = defaultdict(Counter)
        self._total_tokens = 0
        self._last_experts: dict[int, set[int]] = {}  # For temporal locality

    def enable(self) -> None:
        """Enable expert activation tracking."""
        self._enabled = True
        logger.info("Expert activation tracking enabled")

    def disable(self) -> None:
        """Disable expert activation tracking."""
        self._enabled = False
        logger.info("Expert activation tracking disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Record expert activation for a batch of tokens.

        Args:
            layer_id: Transformer layer index.
            topk_ids: Tensor of shape (num_tokens, top_k) with expert indices.
            request_id: Optional request identifier.
        """
        if not self._enabled:
            return

        # Move to CPU for recording (avoid GPU memory accumulation)
        topk_cpu = topk_ids.detach().cpu()
        num_tokens = topk_cpu.shape[0]
        expert_ids = topk_cpu.flatten().tolist()

        # Update running counters
        self._layer_counters[layer_id].update(expert_ids)
        self._total_tokens += num_tokens

        # Track temporal locality
        current_experts = set(expert_ids)
        if layer_id in self._last_experts:
            prev = self._last_experts[layer_id]
            # Will be used in analysis
        self._last_experts[layer_id] = current_experts

        # Store detailed record if under limit
        if len(self._records) < self._max_records:
            self._records.append(ExpertActivationRecord(
                timestamp=time.monotonic(),
                layer_id=layer_id,
                expert_ids=expert_ids,
                num_tokens=num_tokens,
                request_id=request_id,
            ))

    def get_expert_frequency(self, layer_id: int) -> dict[int, int]:
        """Get cumulative expert frequency for a specific layer."""
        return dict(self._layer_counters[layer_id])

    def get_hot_experts(
        self,
        layer_id: int,
        top_k: int = 10,
    ) -> list[tuple[int, int]]:
        """
        Get the most frequently accessed experts for a layer.

        Returns list of (expert_id, count) sorted by count descending.
        """
        return self._layer_counters[layer_id].most_common(top_k)

    def analyze(self) -> ExpertUsageAnalysis:
        """
        Perform comprehensive analysis of collected expert traces.

        Returns ExpertUsageAnalysis with frequency distributions,
        coverage metrics, and locality measures.
        """
        analysis = ExpertUsageAnalysis()
        analysis.total_tokens = self._total_tokens
        analysis.total_events = len(self._records)

        for layer_id, counter in self._layer_counters.items():
            freq = dict(counter)
            analysis.expert_frequency[layer_id] = freq
            analysis.unique_experts_per_layer[layer_id] = len(freq)

            for eid, count in freq.items():
                analysis.global_frequency[eid] += count

            # Compute Gini coefficient for this layer
            if freq:
                values = sorted(freq.values())
                n = len(values)
                if n > 0 and sum(values) > 0:
                    cumulative = 0.0
                    total = sum(values)
                    weighted_sum = 0.0
                    for i, v in enumerate(values):
                        cumulative += v
                        weighted_sum += (2 * (i + 1) - n - 1) * v
                    gini = weighted_sum / (n * total)
                    analysis.gini_per_layer[layer_id] = gini

        # Top-K coverage
        if analysis.global_frequency:
            total_activations = sum(analysis.global_frequency.values())
            sorted_experts = sorted(
                analysis.global_frequency.values(), reverse=True
            )
            for k_pct in [10, 20, 30, 50]:
                k = max(1, len(sorted_experts) * k_pct // 100)
                coverage = sum(sorted_experts[:k]) / total_activations
                analysis.top_k_coverage[k_pct] = coverage

        # Temporal locality from records
        if len(self._records) > 1:
            overlap_sum = 0
            overlap_count = 0
            prev_by_layer: dict[int, set[int]] = {}
            for record in self._records:
                current = set(record.expert_ids)
                if record.layer_id in prev_by_layer:
                    prev = prev_by_layer[record.layer_id]
                    if prev and current:
                        overlap = len(prev & current) / len(prev | current)
                        overlap_sum += overlap
                        overlap_count += 1
                prev_by_layer[record.layer_id] = current

            if overlap_count > 0:
                analysis.temporal_locality = overlap_sum / overlap_count

        return analysis

    def save_traces(self, filepath: str) -> None:
        """Save collected traces to a JSONL file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for record in self._records:
                line = json.dumps({
                    "timestamp": record.timestamp,
                    "layer_id": record.layer_id,
                    "expert_ids": record.expert_ids,
                    "num_tokens": record.num_tokens,
                    "request_id": record.request_id,
                })
                f.write(line + "\n")

        # Also save summary stats
        analysis = self.analyze()
        summary_path = path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "total_tokens": analysis.total_tokens,
                "total_events": analysis.total_events,
                "unique_experts_per_layer": analysis.unique_experts_per_layer,
                "top_k_coverage": analysis.top_k_coverage,
                "temporal_locality": analysis.temporal_locality,
                "gini_per_layer": {
                    str(k): v for k, v in analysis.gini_per_layer.items()
                },
            }, f, indent=2)

        logger.info(
            "Expert traces saved: %d records to %s, summary to %s",
            len(self._records), filepath, summary_path,
        )

    def clear(self) -> None:
        """Clear all collected traces and counters."""
        self._records.clear()
        self._layer_counters.clear()
        self._total_tokens = 0
        self._last_experts.clear()
