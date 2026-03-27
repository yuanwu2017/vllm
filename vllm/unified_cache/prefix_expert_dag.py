# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prefix Expert DAG: Maps prefix hashes to observed MoE expert activations.

This module builds a DAG (Directed Acyclic Graph) that mirrors vLLM's
hash-chain based prefix cache structure, augmenting each node with expert
activation statistics. When a request with a known prefix is seen, the
DAG predicts which experts will be needed based on historical observations.

Key insight: requests sharing a prefix (e.g. same system prompt) tend
to activate similar experts. By recording which experts were activated
for each prefix segment, we can proactively prefetch expert weights to
GPU before the actual router decision is made.

Structure:
    Node key   = block_hash (same hash vLLM uses for KV cache blocks)
    Node value  = per-layer expert activation frequencies + metadata

    The DAG implicitly follows the hash chain:
        parent_hash -> child_hash (successive token blocks)

Integration:
    - ExpertActivationTracker tags records with prefix_hash
    - PrefixExpertDAG accumulates these records per node
    - ExpertPredictor queries the DAG to predict experts for new requests
"""

import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PrefixDAGNode:
    """A node in the prefix-expert DAG.

    Each node corresponds to a single prefix block hash and stores
    aggregated expert activation statistics observed after this prefix.
    """

    # The block hash this node represents (matches vLLM's BlockHash).
    block_hash: int

    # Per-layer expert activation frequency.
    # layer_id -> Counter({expert_id: count})
    expert_counts: dict[int, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    # Total number of forward passes (tokens) observed at this prefix.
    total_observations: int = 0

    # Timestamp of the last update (monotonic).
    last_update: float = 0.0

    # Number of unique requests that contributed observations.
    request_count: int = 0

    # Children: block_hash -> PrefixDAGNode (for traversal).
    children: dict[int, "PrefixDAGNode"] = field(default_factory=dict)

    # Parent hash (for back-traversal; 0 = root / no parent).
    parent_hash: int = 0


class PrefixExpertDAG:
    """
    DAG that maps prefix hash chains to expert activation distributions.

    Thread-safe. Designed for concurrent reads (during prediction) and
    writes (during training/recording).

    Usage:
        dag = PrefixExpertDAG()

        # Record observations (in MoE forward hook)
        dag.record(
            block_hashes=[h0, h1, h2],
            layer_id=3,
            expert_ids=[2, 7, 15, 23],
        )

        # Predict experts for a new request with known prefix
        predictions = dag.predict(
            block_hashes=[h0, h1, h2],
            top_k_per_layer=8,
        )
        # predictions: {layer_id: [(expert_id, probability), ...]}
    """

    def __init__(
        self,
        decay_factor: float = 0.99,
        max_nodes: int = 100_000,
    ):
        """
        Args:
            decay_factor: Exponential decay for old observations when
                updating counts. 1.0 = no decay (pure accumulation).
            max_nodes: Maximum number of DAG nodes. Oldest nodes are
                evicted when this limit is exceeded.
        """
        self.decay_factor = decay_factor
        self.max_nodes = max_nodes

        # Hash table of all nodes: block_hash -> PrefixDAGNode
        self._nodes: dict[int, PrefixDAGNode] = {}

        # Lock for thread safety
        self._lock = threading.RLock()

        # Statistics
        self._total_records = 0
        self._total_predictions = 0

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def _get_or_create_node(self, block_hash: int) -> PrefixDAGNode:
        """Get existing node or create a new one (caller holds lock)."""
        node = self._nodes.get(block_hash)
        if node is None:
            node = PrefixDAGNode(block_hash=block_hash)
            self._nodes[block_hash] = node
        return node

    def record(
        self,
        block_hashes: Sequence[int],
        layer_id: int,
        expert_ids: Sequence[int],
        num_tokens: int = 1,
    ) -> None:
        """
        Record expert activations associated with a prefix hash chain.

        Each block_hash in the chain gets the same expert observation,
        because prefetch decisions can be made at any point in the prefix.
        We record at the **last** (deepest) hash in the chain, which is
        the most specific prefix.

        Args:
            block_hashes: Sequence of prefix block hashes for the current
                request, in order (chain from root to current position).
                These match vLLM's hash-chain structure.
            layer_id: The MoE layer that produced these activations.
            expert_ids: Expert IDs that were activated (flattened topk_ids).
            num_tokens: Number of tokens in this observation.
        """
        if not block_hashes:
            return

        with self._lock:
            # Record at the deepest (most specific) prefix node
            target_hash = block_hashes[-1]
            node = self._get_or_create_node(target_hash)

            # Apply decay to existing counts before adding new observations
            if self.decay_factor < 1.0 and node.total_observations > 0:
                for lid in node.expert_counts:
                    for eid in node.expert_counts[lid]:
                        node.expert_counts[lid][eid] = int(
                            node.expert_counts[lid][eid] * self.decay_factor
                        )

            # Add new observations
            node.expert_counts[layer_id].update(expert_ids)
            node.total_observations += num_tokens
            node.last_update = time.monotonic()
            node.request_count += 1

            # Maintain parent-child links along the chain
            for i in range(len(block_hashes) - 1):
                parent = self._get_or_create_node(block_hashes[i])
                child_hash = block_hashes[i + 1]
                if child_hash not in parent.children:
                    child = self._get_or_create_node(child_hash)
                    parent.children[child_hash] = child
                    child.parent_hash = block_hashes[i]

            self._total_records += 1

            # Evict old nodes if over limit
            if len(self._nodes) > self.max_nodes:
                self._evict_oldest()

    def predict(
        self,
        block_hashes: Sequence[int],
        top_k_per_layer: int = 8,
        min_observations: int = 2,
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Predict which experts will be needed for a request with this prefix.

        Walks the DAG from the deepest matching hash toward the root,
        aggregating expert frequencies. Returns per-layer predictions
        sorted by probability.

        Args:
            block_hashes: Prefix block hashes for the incoming request.
            top_k_per_layer: Maximum number of predicted experts per layer.
            min_observations: Minimum observations at a node to trust it.

        Returns:
            Dict of layer_id -> [(expert_id, probability), ...]
            sorted by probability descending. Probabilities sum to ≤ 1.
        """
        if not block_hashes:
            return {}

        with self._lock:
            self._total_predictions += 1

            # Aggregate expert counts from matching nodes.
            # Start from deepest (most specific) and walk toward root.
            aggregated: dict[int, Counter] = defaultdict(Counter)
            total_weight = 0.0

            for i, bh in enumerate(reversed(block_hashes)):
                node = self._nodes.get(bh)
                if node is None:
                    continue
                if node.total_observations < min_observations:
                    continue

                # Weight: deeper (more specific) nodes get higher weight.
                # Depth weight: 1.0 for deepest, decaying for shallower.
                depth_weight = 0.8 ** i  # exponential decay by distance
                for layer_id, counter in node.expert_counts.items():
                    for eid, count in counter.items():
                        aggregated[layer_id][eid] += count * depth_weight
                total_weight += depth_weight

        if not aggregated:
            return {}

        # Normalize to probabilities per layer
        predictions: dict[int, list[tuple[int, float]]] = {}
        for layer_id, counter in aggregated.items():
            total = sum(counter.values())
            if total == 0:
                continue
            ranked = counter.most_common(top_k_per_layer)
            predictions[layer_id] = [
                (eid, count / total) for eid, count in ranked
            ]

        return predictions

    def get_node(self, block_hash: int) -> Optional[PrefixDAGNode]:
        """Get a DAG node by its block hash (read-only, thread-safe)."""
        with self._lock:
            return self._nodes.get(block_hash)

    def get_stats(self) -> dict:
        """Return DAG statistics."""
        with self._lock:
            total_obs = sum(n.total_observations for n in self._nodes.values())
            return {
                "num_nodes": len(self._nodes),
                "total_records": self._total_records,
                "total_predictions": self._total_predictions,
                "total_observations": total_obs,
            }

    def clear(self) -> None:
        """Clear all DAG nodes."""
        with self._lock:
            self._nodes.clear()
            self._total_records = 0
            self._total_predictions = 0

    def _evict_oldest(self) -> None:
        """Evict the oldest 10% of nodes (caller holds lock)."""
        if not self._nodes:
            return
        target = int(self.max_nodes * 0.9)
        nodes_by_time = sorted(
            self._nodes.values(), key=lambda n: n.last_update
        )
        evict_count = len(self._nodes) - target
        for node in nodes_by_time[:evict_count]:
            # Remove from parent's children dict
            parent = self._nodes.get(node.parent_hash)
            if parent and node.block_hash in parent.children:
                del parent.children[node.block_hash]
            del self._nodes[node.block_hash]
