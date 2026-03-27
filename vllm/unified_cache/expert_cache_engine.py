# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ExpertCacheEngine: Manages MoE expert weight tensors across GPU and CPU memory.

Instead of keeping all expert weights permanently on GPU, this engine
dynamically loads/evicts expert weights between GPU HBM and CPU DRAM based
on access patterns. This is the core component that enables treating expert
weights as cacheable, evictable state — analogous to how vLLM's
BlockSpaceManager handles KV cache blocks.

Key operations:
- load_expert(): Async GPU←CPU transfer with CUDA streams
- evict_expert(): Async GPU→CPU transfer to free GPU memory
- prefetch_experts(): Predictive loading based on router logits
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class ExpertLocation(Enum):
    """Where an expert's weights currently reside."""
    GPU = "gpu"
    CPU = "cpu"
    LOADING = "loading"   # Currently being transferred GPU←CPU
    EVICTING = "evicting"  # Currently being transferred GPU→CPU


@dataclass
class ExpertCacheEntry:
    """Metadata for a single cached expert."""
    layer_id: int
    expert_id: int
    location: ExpertLocation
    # Weight tensors: w13 (gate_up fused) and w2 (down_proj)
    gpu_tensors: Optional[dict[str, torch.Tensor]] = None
    cpu_tensors: Optional[dict[str, torch.Tensor]] = None
    # Access statistics
    access_count: int = 0
    last_access_time: float = 0.0
    ema_access_rate: float = 0.0  # Exponential moving average
    # Memory accounting
    memory_bytes: int = 0
    # Transfer state
    transfer_event: Optional[torch.cuda.Event] = None


@dataclass
class ExpertCacheStats:
    """Runtime statistics for the expert cache."""
    total_loads: int = 0
    total_evictions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    gpu_memory_used: int = 0
    cpu_memory_used: int = 0
    avg_load_time_ms: float = 0.0
    _load_time_sum: float = field(default=0.0, repr=False)


class ExpertCacheEngine:
    """
    Manages expert weight tensors across GPU HBM and CPU DRAM.

    Design principles:
    1. Hot experts stay on GPU, cold experts move to CPU
    2. Transfers use dedicated CUDA streams for async overlap
    3. Pinned CPU memory for maximum PCIe bandwidth
    4. EMA-based access tracking for eviction decisions

    Args:
        gpu_memory_budget: Maximum GPU memory (bytes) for expert weights.
        cpu_memory_budget: Maximum CPU memory (bytes) for expert weights.
        device: CUDA device for GPU tensors.
        ema_decay: Decay factor for exponential moving average of access rates.
        pin_cpu_memory: Whether to use pinned (page-locked) CPU memory.
    """

    def __init__(
        self,
        gpu_memory_budget: int,
        cpu_memory_budget: int,
        device: torch.device,
        ema_decay: float = 0.95,
        pin_cpu_memory: bool = True,
    ):
        self.gpu_memory_budget = gpu_memory_budget
        self.cpu_memory_budget = cpu_memory_budget
        self.device = device
        self.ema_decay = ema_decay
        self.pin_cpu_memory = pin_cpu_memory

        # Expert cache: (layer_id, expert_id) -> ExpertCacheEntry
        self._cache: dict[tuple[int, int], ExpertCacheEntry] = {}

        # LRU tracking for GPU-resident experts
        self._gpu_lru: OrderedDict[tuple[int, int], None] = OrderedDict()

        # Memory accounting
        self._gpu_memory_used: int = 0
        self._cpu_memory_used: int = 0

        # Dedicated CUDA stream for async transfers (only on CUDA devices)
        self._is_cuda = device.type == "cuda"
        self._transfer_stream = (
            torch.cuda.Stream(device=device) if self._is_cuda else None
        )

        # Statistics
        self.stats = ExpertCacheStats()

        # Lock for thread safety during concurrent access
        self._lock = threading.Lock()

        logger.info(
            "ExpertCacheEngine initialized: GPU budget=%.2f GiB, "
            "CPU budget=%.2f GiB, device=%s, ema_decay=%.3f",
            gpu_memory_budget / (1024**3),
            cpu_memory_budget / (1024**3),
            device,
            ema_decay,
        )

    def register_expert(
        self,
        layer_id: int,
        expert_id: int,
        weight_tensors: dict[str, torch.Tensor],
        on_gpu: bool = True,
    ) -> None:
        """
        Register an expert's weights with the cache engine.

        Called during model initialization to register all expert weights.
        Initially all experts are on GPU (matching vLLM's default behavior).

        Args:
            layer_id: The transformer layer index.
            expert_id: The expert index within the layer.
            weight_tensors: Dict of weight name -> tensor (e.g., {"w13": ..., "w2": ...}).
            on_gpu: Whether the expert is initially on GPU.
        """
        key = (layer_id, expert_id)
        memory_bytes = sum(t.nelement() * t.element_size()
                           for t in weight_tensors.values())

        if on_gpu:
            # Create CPU copies in pinned memory for fast transfers
            cpu_tensors = {}
            for name, tensor in weight_tensors.items():
                cpu_t = torch.empty_like(tensor, device="cpu")
                if self.pin_cpu_memory:
                    cpu_t = cpu_t.pin_memory()
                cpu_t.copy_(tensor)
                cpu_tensors[name] = cpu_t

            entry = ExpertCacheEntry(
                layer_id=layer_id,
                expert_id=expert_id,
                location=ExpertLocation.GPU,
                gpu_tensors=weight_tensors,
                cpu_tensors=cpu_tensors,
                memory_bytes=memory_bytes,
                last_access_time=time.monotonic(),
            )
            self._gpu_memory_used += memory_bytes
            self._cpu_memory_used += memory_bytes
            self._gpu_lru[key] = None
        else:
            # Expert starts on CPU only
            cpu_tensors = {}
            for name, tensor in weight_tensors.items():
                if tensor.device.type != "cpu":
                    cpu_t = torch.empty_like(tensor, device="cpu")
                    if self.pin_cpu_memory:
                        cpu_t = cpu_t.pin_memory()
                    cpu_t.copy_(tensor)
                    cpu_tensors[name] = cpu_t
                else:
                    if self.pin_cpu_memory and not tensor.is_pinned():
                        cpu_t = torch.empty_like(tensor).pin_memory()
                        cpu_t.copy_(tensor)
                        cpu_tensors[name] = cpu_t
                    else:
                        cpu_tensors[name] = tensor

            entry = ExpertCacheEntry(
                layer_id=layer_id,
                expert_id=expert_id,
                location=ExpertLocation.CPU,
                gpu_tensors=None,
                cpu_tensors=cpu_tensors,
                memory_bytes=memory_bytes,
                last_access_time=time.monotonic(),
            )
            self._cpu_memory_used += memory_bytes

        self._cache[key] = entry

    def get_expert(
        self,
        layer_id: int,
        expert_id: int,
    ) -> dict[str, torch.Tensor] | None:
        """
        Get expert weight tensors on GPU. Returns None if expert needs loading.

        This is the hot path — called during every MoE forward pass.
        Updates access statistics and LRU order.

        Args:
            layer_id: The transformer layer index.
            expert_id: The expert index.

        Returns:
            Dict of weight tensors on GPU, or None if not cached on GPU.
        """
        key = (layer_id, expert_id)
        entry = self._cache.get(key)
        if entry is None:
            return None

        now = time.monotonic()

        if entry.location == ExpertLocation.GPU:
            # Cache hit — update stats
            entry.access_count += 1
            entry.last_access_time = now
            entry.ema_access_rate = (
                self.ema_decay * entry.ema_access_rate + (1 - self.ema_decay)
            )
            # Move to end of LRU (most recently used)
            self._gpu_lru.move_to_end(key)
            self.stats.cache_hits += 1
            return entry.gpu_tensors

        elif entry.location == ExpertLocation.LOADING:
            # Transfer in progress — wait for completion
            if entry.transfer_event is not None and self._is_cuda:
                entry.transfer_event.synchronize()
            entry.location = ExpertLocation.GPU
            entry.transfer_event = None
            entry.access_count += 1
            entry.last_access_time = now
            self._gpu_lru[key] = None
            self.stats.cache_hits += 1
            return entry.gpu_tensors

        # Cache miss
        self.stats.cache_misses += 1
        # Update EMA to reflect that we needed this expert but it wasn't on GPU
        entry.ema_access_rate = (
            self.ema_decay * entry.ema_access_rate + (1 - self.ema_decay)
        )
        return None

    def load_expert(
        self,
        layer_id: int,
        expert_id: int,
        sync: bool = False,
    ) -> torch.cuda.Event | None:
        """
        Load an expert from CPU→GPU asynchronously.

        Uses a dedicated CUDA stream for async transfer. Returns a CUDA event
        that can be used to synchronize when the transfer is complete.

        Args:
            layer_id: The transformer layer index.
            expert_id: The expert index.
            sync: If True, block until transfer completes.

        Returns:
            CUDA event for synchronization, or None if already on GPU.
        """
        key = (layer_id, expert_id)
        entry = self._cache.get(key)
        if entry is None:
            logger.warning("Cannot load unregistered expert (%d, %d)",
                           layer_id, expert_id)
            return None

        if entry.location == ExpertLocation.GPU:
            return None  # Already on GPU

        if entry.location == ExpertLocation.LOADING:
            return entry.transfer_event  # Already loading

        # Ensure GPU has space
        while self._gpu_memory_used + entry.memory_bytes > self.gpu_memory_budget:
            if not self._evict_one():
                logger.warning(
                    "Cannot free GPU memory for expert (%d, %d). "
                    "GPU used: %d, budget: %d, needed: %d",
                    layer_id, expert_id,
                    self._gpu_memory_used,
                    self.gpu_memory_budget,
                    entry.memory_bytes,
                )
                return None

        # Transfer CPU→GPU (async on CUDA, sync on CPU)
        load_start = time.monotonic()
        entry.location = ExpertLocation.LOADING

        if self._is_cuda:
            with torch.cuda.stream(self._transfer_stream):
                gpu_tensors = {}
                for name, cpu_tensor in entry.cpu_tensors.items():
                    gpu_tensor = torch.empty_like(
                        cpu_tensor, device=self.device
                    )
                    gpu_tensor.copy_(cpu_tensor, non_blocking=True)
                    gpu_tensors[name] = gpu_tensor
                entry.gpu_tensors = gpu_tensors

            event = torch.cuda.Event()
            self._transfer_stream.record_event(event)
            entry.transfer_event = event
        else:
            # CPU-to-CPU copy (for testing without GPU)
            gpu_tensors = {}
            for name, cpu_tensor in entry.cpu_tensors.items():
                gpu_tensors[name] = cpu_tensor.clone()
            entry.gpu_tensors = gpu_tensors
            event = None

        self._gpu_memory_used += entry.memory_bytes

        if self._is_cuda and sync and event is not None:
            event.synchronize()

        entry.location = ExpertLocation.GPU
        entry.transfer_event = None
        self._gpu_lru[key] = None
        load_time = (time.monotonic() - load_start) * 1000
        self.stats._load_time_sum += load_time
        self.stats.total_loads += 1
        self.stats.avg_load_time_ms = (
            self.stats._load_time_sum / self.stats.total_loads
        )

        return event

    def evict_expert(
        self,
        layer_id: int,
        expert_id: int,
        sync: bool = False,
    ) -> bool:
        """
        Evict an expert from GPU→CPU.

        The CPU copy is already maintained (updated during load), so eviction
        just frees the GPU tensors.

        Args:
            layer_id: The transformer layer index.
            expert_id: The expert index.
            sync: If True, block until eviction completes.

        Returns:
            True if eviction was performed.
        """
        key = (layer_id, expert_id)
        entry = self._cache.get(key)
        if entry is None or entry.location != ExpertLocation.GPU:
            return False

        # Update CPU copy if GPU tensors were modified (e.g., by EPLB)
        if self._is_cuda and self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                for name, gpu_tensor in entry.gpu_tensors.items():
                    entry.cpu_tensors[name].copy_(
                        gpu_tensor, non_blocking=True
                    )
            if sync:
                self._transfer_stream.synchronize()
        else:
            for name, gpu_tensor in entry.gpu_tensors.items():
                entry.cpu_tensors[name].copy_(gpu_tensor)

        # Free GPU tensors
        entry.gpu_tensors = None
        entry.location = ExpertLocation.CPU
        self._gpu_memory_used -= entry.memory_bytes
        self._gpu_lru.pop(key, None)
        self.stats.total_evictions += 1

        return True

    def prefetch_experts(
        self,
        layer_id: int,
        expert_ids: list[int],
    ) -> list[torch.cuda.Event]:
        """
        Prefetch a set of experts for a given layer.

        Called before MoE forward pass when we know which experts will be
        needed (from router logits). Loads any missing experts asynchronously.

        Args:
            layer_id: The transformer layer index.
            expert_ids: List of expert indices to prefetch.

        Returns:
            List of CUDA events for pending transfers.
        """
        events = []
        for expert_id in expert_ids:
            key = (layer_id, expert_id)
            entry = self._cache.get(key)
            if entry is None:
                continue
            if entry.location == ExpertLocation.GPU:
                continue
            if entry.location == ExpertLocation.LOADING:
                if entry.transfer_event is not None:
                    events.append(entry.transfer_event)
                continue

            event = self.load_expert(layer_id, expert_id, sync=False)
            if event is not None:
                events.append(event)

        return events

    def _evict_one(self) -> bool:
        """
        Evict the least-recently-used GPU expert to CPU.

        Returns True if an expert was evicted.
        """
        if not self._gpu_lru:
            return False

        # Pop the LRU entry (first item in OrderedDict)
        key, _ = self._gpu_lru.popitem(last=False)
        layer_id, expert_id = key
        return self.evict_expert(layer_id, expert_id, sync=True)

    def get_eviction_candidates(
        self,
        count: int = 1,
    ) -> list[tuple[int, int, float]]:
        """
        Get the best candidates for eviction based on access patterns.

        Returns list of (layer_id, expert_id, score) sorted by score
        (lowest score = best eviction candidate).

        The score combines EMA access rate with memory cost:
            score = ema_access_rate * compute_cost / memory_bytes
        """
        candidates = []
        for key in self._gpu_lru:
            entry = self._cache[key]
            if entry.location != ExpertLocation.GPU:
                continue
            # Score: higher = more valuable to keep
            score = entry.ema_access_rate / max(entry.memory_bytes, 1)
            candidates.append((entry.layer_id, entry.expert_id, score))

        # Sort by score ascending (lowest score = evict first)
        candidates.sort(key=lambda x: x[2])
        return candidates[:count]

    def get_gpu_expert_ids(self, layer_id: int) -> list[int]:
        """Get list of expert IDs currently cached on GPU for a given layer."""
        result = []
        for (lid, eid), entry in self._cache.items():
            if lid == layer_id and entry.location == ExpertLocation.GPU:
                result.append(eid)
        return result

    def get_cpu_expert_ids(self, layer_id: int) -> list[int]:
        """Get list of expert IDs currently resident on CPU for a given layer."""
        result = []
        for (lid, eid), entry in self._cache.items():
            if lid == layer_id and entry.location == ExpertLocation.CPU:
                result.append(eid)
        return result

    def get_cpu_tensors(
        self,
        layer_id: int,
        expert_id: int,
    ) -> dict[str, torch.Tensor] | None:
        """
        Get CPU weight tensors for a given expert.

        Used by the CPU active compute path to run expert FFN directly
        on CPU without transferring weights to GPU.

        Returns:
            Dict of weight name -> CPU tensor, or None if not found.
        """
        key = (layer_id, expert_id)
        entry = self._cache.get(key)
        if entry is None:
            return None
        return entry.cpu_tensors

    def get_all_gpu_expert_ids(self) -> dict[int, list[int]]:
        """Get mapping of layer_id -> list of GPU-resident expert IDs."""
        result: dict[int, list[int]] = {}
        for (lid, eid), entry in self._cache.items():
            if entry.location == ExpertLocation.GPU:
                result.setdefault(lid, []).append(eid)
        return result

    @property
    def gpu_memory_used(self) -> int:
        return self._gpu_memory_used

    @property
    def cpu_memory_used(self) -> int:
        return self._cpu_memory_used

    @property
    def gpu_utilization(self) -> float:
        if self.gpu_memory_budget == 0:
            return 0.0
        return self._gpu_memory_used / self.gpu_memory_budget

    @property
    def hit_rate(self) -> float:
        total = self.stats.cache_hits + self.stats.cache_misses
        if total == 0:
            return 0.0
        return self.stats.cache_hits / total

    def update_gpu_budget(self, new_budget: int) -> None:
        """
        Dynamically update the GPU memory budget for expert cache.

        Called by UnifiedCacheManager during memory rebalancing.
        May trigger evictions if the new budget is smaller.
        """
        self.gpu_memory_budget = new_budget
        # Evict experts if over budget
        while self._gpu_memory_used > self.gpu_memory_budget:
            if not self._evict_one():
                break

        logger.info(
            "Expert cache GPU budget updated to %.2f GiB "
            "(used: %.2f GiB, %d experts on GPU)",
            new_budget / (1024**3),
            self._gpu_memory_used / (1024**3),
            len(self._gpu_lru),
        )

    def __repr__(self) -> str:
        return (
            f"ExpertCacheEngine("
            f"gpu={self._gpu_memory_used / (1024**3):.2f}/"
            f"{self.gpu_memory_budget / (1024**3):.2f} GiB, "
            f"cpu={self._cpu_memory_used / (1024**3):.2f}/"
            f"{self.cpu_memory_budget / (1024**3):.2f} GiB, "
            f"entries={len(self._cache)}, "
            f"gpu_entries={len(self._gpu_lru)}, "
            f"hit_rate={self.hit_rate:.2%})"
        )
