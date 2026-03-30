#!/usr/bin/env python3
"""
CUDA Hardware Benchmark for Phase 3+4: Prefix DAG + Expert Prediction.

Measures:
  1. PrefixExpertDAG record/predict throughput
  2. ExpertPredictor prefetch latency on real GPU
  3. End-to-end: prediction → prefetch → MoE forward (vs no prediction)
  4. DAG scalability with increasing node count
  5. Prediction accuracy with varying prefix depth

Run on a CUDA machine:
    python tests/unified_cache/benchmark_prediction.py
"""

import time
import statistics
import torch
import sys

from vllm.unified_cache.expert_cache_engine import ExpertCacheEngine, ExpertLocation
from vllm.unified_cache.expert_predictor import ExpertPredictor
from vllm.unified_cache.prefix_expert_dag import PrefixExpertDAG
from vllm.unified_cache.cpu_expert_compute import cpu_expert_ffn


# ---------------------------------------------------------------------------
# Config — simulates Qwen1.5-MoE-A2.7B
# ---------------------------------------------------------------------------
HIDDEN = 2048
INTERMEDIATE = 5632
NUM_EXPERTS = 60
TOP_K = 4
NUM_LAYERS = 24
DTYPE = torch.float16

# How many experts to keep on GPU initially (simulate partial cache)
GPU_EXPERTS = set(range(20))  # 20 hot experts on GPU, 40 cold on CPU


def _expert_memory_bytes():
    """Memory per expert (w13 + w2)."""
    w13 = 2 * INTERMEDIATE * HIDDEN * 2  # fp16
    w2 = HIDDEN * INTERMEDIATE * 2
    return w13 + w2


def _make_engine(device):
    expert_mem = _expert_memory_bytes()
    gpu_budget = expert_mem * 25  # room for ~25 experts
    cpu_budget = expert_mem * NUM_EXPERTS

    engine = ExpertCacheEngine(
        gpu_memory_budget=gpu_budget,
        cpu_memory_budget=cpu_budget,
        device=device,
        pin_cpu_memory=True,
    )

    print(f"Registering {NUM_EXPERTS} experts per layer "
          f"({NUM_LAYERS} layers) ...")
    print(f"  Expert size: {expert_mem / 1024**2:.1f} MiB")
    print(f"  GPU budget: {gpu_budget / 1024**2:.0f} MiB "
          f"(~{gpu_budget // expert_mem} experts)")
    print(f"  GPU-resident: {len(GPU_EXPERTS)}, CPU-only: "
          f"{NUM_EXPERTS - len(GPU_EXPERTS)}")

    # Only register a single layer for benchmarks (saves memory)
    for eid in range(NUM_EXPERTS):
        w13 = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=DTYPE, device="cpu")
        w2 = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE, device="cpu")
        if eid in GPU_EXPERTS:
            w13_gpu = w13.to(device)
            w2_gpu = w2.to(device)
            engine.register_expert(0, eid, {"w13_weight": w13_gpu, "w2_weight": w2_gpu}, on_gpu=True)
        else:
            engine.register_expert(0, eid, {"w13_weight": w13, "w2_weight": w2}, on_gpu=False)

    return engine


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Benchmark 1: DAG record/predict throughput
# ---------------------------------------------------------------------------
def bench_dag_throughput():
    separator("Benchmark 1: DAG Record/Predict Throughput")

    dag = PrefixExpertDAG(decay_factor=0.99, max_nodes=100_000)

    # Populate DAG with realistic data
    num_prefixes = 1000
    prefix_depth = 8
    experts_per_obs = TOP_K

    print(f"  Recording {num_prefixes} prefixes × {prefix_depth} depth ...")

    record_times = []
    for i in range(num_prefixes):
        hashes = [hash((i, d)) for d in range(prefix_depth)]
        expert_ids = [(i * TOP_K + k) % NUM_EXPERTS for k in range(experts_per_obs)]
        layer_id = i % NUM_LAYERS

        t0 = time.perf_counter()
        dag.record(hashes, layer_id, expert_ids, num_tokens=4)
        record_times.append(time.perf_counter() - t0)

    print(f"  DAG nodes: {dag.num_nodes}")
    print(f"  Record latency: "
          f"median={statistics.median(record_times)*1e6:.1f} µs, "
          f"p99={sorted(record_times)[int(0.99*len(record_times))]*1e6:.1f} µs")

    # Predict
    predict_times = []
    for i in range(500):
        hashes = [hash((i, d)) for d in range(prefix_depth)]
        t0 = time.perf_counter()
        preds = dag.predict(hashes, top_k_per_layer=8, min_observations=1)
        predict_times.append(time.perf_counter() - t0)

    print(f"  Predict latency: "
          f"median={statistics.median(predict_times)*1e6:.1f} µs, "
          f"p99={sorted(predict_times)[int(0.99*len(predict_times))]*1e6:.1f} µs")
    print(f"  Predict throughput: "
          f"{1.0 / statistics.mean(predict_times):.0f} predictions/sec")


# ---------------------------------------------------------------------------
# Benchmark 2: Prefetch latency on GPU
# ---------------------------------------------------------------------------
def bench_prefetch_latency(device):
    separator("Benchmark 2: Predictive Prefetch Latency (GPU)")

    engine = _make_engine(device)
    dag = PrefixExpertDAG(decay_factor=1.0)
    predictor = ExpertPredictor(
        dag=dag,
        cache_engine=engine,
        confidence_threshold=0.05,
        max_prefetch_per_layer=8,
    )

    # Build DAG: prefix [100,200] always activates experts 25-35 (CPU-only)
    target_experts = list(range(25, 33))  # 8 CPU experts
    for _ in range(20):
        dag.record([100, 200], layer_id=0, expert_ids=target_experts)

    # Warm up GPU
    torch.cuda.synchronize(device)

    # Measure prediction + prefetch time
    prefetch_times = []
    for trial in range(10):
        # First evict the target experts back to CPU
        for eid in target_experts:
            entry = engine._cache.get((0, eid))
            if entry and entry.location == ExpertLocation.GPU:
                engine.evict_expert(0, eid)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        result = predictor.predict_and_prefetch([100, 200])

        # Wait for all async prefetches to complete
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        prefetch_times.append(elapsed)

        n_prefetched = sum(len(v) for v in result.values())
        if trial == 0:
            print(f"  Experts prefetched: {n_prefetched}")
            print(f"  Prefetched IDs: {result}")

    print(f"  Prefetch latency (predict+load+sync): "
          f"median={statistics.median(prefetch_times)*1e3:.2f} ms, "
          f"p99={sorted(prefetch_times)[min(9, len(prefetch_times)-1)]*1e3:.2f} ms")

    del engine
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Benchmark 3: End-to-end comparison (with vs without prediction)
# ---------------------------------------------------------------------------
def bench_e2e_comparison(device):
    separator("Benchmark 3: E2E — With vs Without Prediction")

    engine_pred = _make_engine(device)
    engine_nopred = _make_engine(device)

    dag = PrefixExpertDAG(decay_factor=1.0)
    predictor = ExpertPredictor(
        dag=dag,
        cache_engine=engine_pred,
        confidence_threshold=0.05,
        max_prefetch_per_layer=8,
    )

    # Seed DAG with a pattern
    cold_experts = list(range(30, 38))  # 8 CPU experts
    for _ in range(30):
        dag.record([500, 600, 700], layer_id=0, expert_ids=cold_experts)

    num_tokens = 16
    x = torch.randn(num_tokens, HIDDEN, dtype=DTYPE, device=device)

    def simulate_moe_forward(eng, expert_ids):
        """Simulate an MoE forward that needs specific experts on GPU."""
        # Ensure experts are loaded
        events = []
        for eid in expert_ids:
            tensors = eng.get_expert(0, eid)
            if tensors is None:
                ev = eng.load_expert(0, eid, sync=False)
                if ev is not None:
                    events.append(ev)
        for ev in events:
            ev.synchronize()
        # Simulate compute (just a matmul placeholder)
        tensors = eng.get_expert(0, expert_ids[0])
        if tensors is not None:
            w = tensors["w13_weight"]
            _ = torch.mm(x, w.t())
        torch.cuda.synchronize(device)

    # --- WITHOUT prediction ---
    times_nopred = []
    for _ in range(10):
        # Evict cold experts
        for eid in cold_experts:
            entry = engine_nopred._cache.get((0, eid))
            if entry and entry.location == ExpertLocation.GPU:
                engine_nopred.evict_expert(0, eid)
        torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        simulate_moe_forward(engine_nopred, cold_experts)
        times_nopred.append(time.perf_counter() - t0)

    # --- WITH prediction ---
    times_pred = []
    for _ in range(10):
        # Evict cold experts
        for eid in cold_experts:
            entry = engine_pred._cache.get((0, eid))
            if entry and entry.location == ExpertLocation.GPU:
                engine_pred.evict_expert(0, eid)
        torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        # Prediction prefetch (can overlap with earlier computation)
        predictor.predict_and_prefetch([500, 600, 700])
        # Simulate MoE forward (experts should already be loading/loaded)
        simulate_moe_forward(engine_pred, cold_experts)
        times_pred.append(time.perf_counter() - t0)

    med_nopred = statistics.median(times_nopred) * 1e3
    med_pred = statistics.median(times_pred) * 1e3
    speedup = med_nopred / med_pred if med_pred > 0 else float('inf')

    print(f"  Without prediction: {med_nopred:.2f} ms (median)")
    print(f"  With prediction:    {med_pred:.2f} ms (median)")
    print(f"  Speedup:            {speedup:.2f}x")

    del engine_pred, engine_nopred
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Benchmark 4: DAG scalability
# ---------------------------------------------------------------------------
def bench_dag_scalability():
    separator("Benchmark 4: DAG Scalability")

    for num_nodes_target in [1_000, 10_000, 50_000, 100_000]:
        dag = PrefixExpertDAG(decay_factor=1.0, max_nodes=num_nodes_target + 1000)

        # Fill DAG
        for i in range(num_nodes_target):
            dag.record([hash(i)], layer_id=i % NUM_LAYERS,
                       expert_ids=[i % NUM_EXPERTS])

        # Measure predict latency
        times = []
        for _ in range(200):
            hashes = [hash(42 + d) for d in range(8)]
            t0 = time.perf_counter()
            dag.predict(hashes, top_k_per_layer=8, min_observations=0)
            times.append(time.perf_counter() - t0)

        med = statistics.median(times) * 1e6
        print(f"  {num_nodes_target:>7,} nodes → predict median: {med:.1f} µs")


# ---------------------------------------------------------------------------
# Benchmark 5: Prediction accuracy vs prefix depth
# ---------------------------------------------------------------------------
def bench_prediction_accuracy():
    separator("Benchmark 5: Prediction Accuracy vs Prefix Depth")

    dag = PrefixExpertDAG(decay_factor=1.0)

    # Create a fixed pattern: prefix hashes [1..D] always activate experts [10..14]
    true_experts = [10, 11, 12, 13]
    max_depth = 16

    # Train the DAG with 50 observations at full depth
    for _ in range(50):
        hashes = list(range(1, max_depth + 1))
        dag.record(hashes, layer_id=0, expert_ids=true_experts)

    # Also add some noise at shallow depths
    for _ in range(20):
        dag.record([1], layer_id=0, expert_ids=[50, 51, 52])
        dag.record([1, 2], layer_id=0, expert_ids=[40, 41])

    print(f"  True experts: {true_experts}")
    print(f"  {'Depth':>6} | {'Predicted':>30} | {'Accuracy':>8}")
    print(f"  {'-'*6}-+-{'-'*30}-+-{'-'*8}")

    for depth in [1, 2, 4, 8, 12, 16]:
        hashes = list(range(1, depth + 1))
        preds = dag.predict(hashes, top_k_per_layer=8, min_observations=1)
        if 0 not in preds:
            print(f"  {depth:>6} | {'(no data)':>30} | {'N/A':>8}")
            continue
        pred_eids = {eid for eid, _ in preds[0]}
        actual = set(true_experts)
        hits = len(pred_eids & actual)
        accuracy = hits / len(actual)
        pred_str = str(sorted(pred_eids))
        print(f"  {depth:>6} | {pred_str:>30} | {accuracy:>7.0%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Phase 3+4 CUDA Benchmark: Prefix DAG + Expert Prediction")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}")
        print(f"  VRAM: {props.total_memory / 1024**3:.1f} GiB")
        print(f"  Expert size: {_expert_memory_bytes() / 1024**2:.1f} MiB")
    else:
        print("  WARNING: No CUDA device — GPU benchmarks will be skipped")
        device = None

    # CPU-only benchmarks
    bench_dag_throughput()
    bench_dag_scalability()
    bench_prediction_accuracy()

    # GPU benchmarks
    if device is not None:
        bench_prefetch_latency(device)
        bench_e2e_comparison(device)

    print(f"\n{'='*60}")
    print("  All benchmarks complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
