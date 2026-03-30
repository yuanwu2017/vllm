#!/usr/bin/env python3
"""
CUDA Hardware Benchmark for Phase 3+4: Prefix DAG + Expert Prediction.

Measures:
  1. PrefixExpertDAG record/predict throughput
  2. ExpertPredictor prefetch latency on real GPU
  3. Pipeline overlap: prefetch hidden behind attention at varying durations
  4. DAG scalability with increasing node count
  5. Prediction accuracy with varying prefix depth

Run on a CUDA machine:
    python tests/unified_cache/benchmark_prediction.py
"""

import time
import statistics
import torch

from vllm.unified_cache.expert_cache_engine import ExpertCacheEngine, ExpertLocation
from vllm.unified_cache.expert_predictor import ExpertPredictor
from vllm.unified_cache.prefix_expert_dag import PrefixExpertDAG


HIDDEN = 2048
INTERMEDIATE = 5632
NUM_EXPERTS = 60
TOP_K = 4
NUM_LAYERS = 24
DTYPE = torch.float16
GPU_EXPERTS = set(range(20))


def _expert_memory_bytes():
    w13 = 2 * INTERMEDIATE * HIDDEN * 2
    w2 = HIDDEN * INTERMEDIATE * 2
    return w13 + w2


def _make_engine(device):
    expert_mem = _expert_memory_bytes()
    gpu_budget = expert_mem * 25
    cpu_budget = expert_mem * NUM_EXPERTS
    engine = ExpertCacheEngine(
        gpu_memory_budget=gpu_budget, cpu_memory_budget=cpu_budget,
        device=device, pin_cpu_memory=True,
    )
    print("  Registering {} experts (layer 0) ...".format(NUM_EXPERTS))
    print("  Expert={:.1f} MiB | GPU budget={:.0f} MiB (~{} experts)".format(
        expert_mem / 1024**2, gpu_budget / 1024**2, gpu_budget // expert_mem))
    print("  GPU-resident: {} | CPU-only: {}".format(
        len(GPU_EXPERTS), NUM_EXPERTS - len(GPU_EXPERTS)))
    for eid in range(NUM_EXPERTS):
        w13 = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=DTYPE, device="cpu")
        w2 = torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE, device="cpu")
        if eid in GPU_EXPERTS:
            engine.register_expert(0, eid,
                {"w13_weight": w13.to(device), "w2_weight": w2.to(device)},
                on_gpu=True)
        else:
            engine.register_expert(0, eid,
                {"w13_weight": w13, "w2_weight": w2}, on_gpu=False)
    return engine


def separator(title):
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70)


def bench_dag_throughput():
    separator("Benchmark 1: DAG Record / Predict Throughput")
    dag = PrefixExpertDAG(decay_factor=0.99, max_nodes=100_000)
    num_prefixes = 1000
    prefix_depth = 8
    print("  Recording {} prefixes x {} depth ...".format(num_prefixes, prefix_depth))

    record_times = []
    for i in range(num_prefixes):
        hashes = [hash((i, d)) for d in range(prefix_depth)]
        expert_ids = [(i * TOP_K + k) % NUM_EXPERTS for k in range(TOP_K)]
        t0 = time.perf_counter()
        dag.record(hashes, i % NUM_LAYERS, expert_ids, num_tokens=4)
        record_times.append(time.perf_counter() - t0)

    p99i = int(0.99 * len(record_times))
    print("  DAG nodes: {}".format(dag.num_nodes))
    print("  Record:  median={:.1f} us, p99={:.1f} us".format(
        statistics.median(record_times)*1e6,
        sorted(record_times)[p99i]*1e6))

    predict_times = []
    for i in range(500):
        hashes = [hash((i, d)) for d in range(prefix_depth)]
        t0 = time.perf_counter()
        dag.predict(hashes, top_k_per_layer=8, min_observations=1)
        predict_times.append(time.perf_counter() - t0)

    p99i = int(0.99 * len(predict_times))
    print("  Predict: median={:.1f} us, p99={:.1f} us".format(
        statistics.median(predict_times)*1e6,
        sorted(predict_times)[p99i]*1e6))
    print("  Predict throughput: {:.0f} predictions/sec".format(
        1.0 / statistics.mean(predict_times)))


def bench_dag_scalability():
    separator("Benchmark 2: DAG Scalability")
    for n in [1_000, 10_000, 50_000, 100_000]:
        dag = PrefixExpertDAG(decay_factor=1.0, max_nodes=n + 1000)
        for i in range(n):
            dag.record([hash(i)], layer_id=i % NUM_LAYERS,
                       expert_ids=[i % NUM_EXPERTS])
        times = []
        for _ in range(200):
            hashes = [hash(42 + d) for d in range(8)]
            t0 = time.perf_counter()
            dag.predict(hashes, top_k_per_layer=8, min_observations=0)
            times.append(time.perf_counter() - t0)
        print("  {:>7,} nodes -> predict median: {:.1f} us".format(
            n, statistics.median(times)*1e6))


def bench_prediction_accuracy():
    separator("Benchmark 3: Prediction Accuracy vs Prefix Depth")
    dag = PrefixExpertDAG(decay_factor=1.0)
    true_experts = [10, 11, 12, 13]
    max_depth = 16
    for _ in range(50):
        dag.record(list(range(1, max_depth+1)), layer_id=0,
                   expert_ids=true_experts)
    for _ in range(20):
        dag.record([1], layer_id=0, expert_ids=[50, 51, 52])
        dag.record([1, 2], layer_id=0, expert_ids=[40, 41])

    print("  True experts: {}".format(true_experts))
    print("  {:>6} | {:>30} | {:>8}".format("Depth", "Predicted", "Accuracy"))
    print("  " + "-"*6 + "-+-" + "-"*30 + "-+-" + "-"*8)
    for depth in [1, 2, 4, 8, 12, 16]:
        preds = dag.predict(list(range(1, depth+1)),
                            top_k_per_layer=8, min_observations=1)
        if 0 not in preds:
            print("  {:>6} | {:>30} | {:>8}".format(depth, "(no data)", "N/A"))
            continue
        pred_eids = {eid for eid, _ in preds[0]}
        hits = len(pred_eids & set(true_experts))
        acc = hits / len(true_experts)
        print("  {:>6} | {:>30} | {:>7.0%}".format(
            depth, str(sorted(pred_eids)), acc))


def bench_prefetch_latency(device):
    separator("Benchmark 4: Predictive Prefetch Latency (GPU)")
    engine = _make_engine(device)
    dag = PrefixExpertDAG(decay_factor=1.0)
    predictor = ExpertPredictor(dag=dag, cache_engine=engine,
        confidence_threshold=0.05, max_prefetch_per_layer=8)
    target = list(range(25, 33))
    for _ in range(20):
        dag.record([100, 200], layer_id=0, expert_ids=target)
    torch.cuda.synchronize(device)

    prefetch_times = []
    for trial in range(10):
        for eid in target:
            entry = engine._cache.get((0, eid))
            if entry and entry.location == ExpertLocation.GPU:
                engine.evict_expert(0, eid)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        result = predictor.predict_and_prefetch([100, 200])
        torch.cuda.synchronize(device)
        prefetch_times.append(time.perf_counter() - t0)
        if trial == 0:
            n = sum(len(v) for v in result.values())
            print("  Experts prefetched: {}".format(n))
            print("  Prefetched IDs: {}".format(result))

    med = statistics.median(prefetch_times) * 1e3
    p99 = sorted(prefetch_times)[min(9, len(prefetch_times)-1)] * 1e3
    print("  Prefetch (predict+load+sync): median={:.2f} ms, p99={:.2f} ms".format(
        med, p99))
    del engine; torch.cuda.empty_cache()


def bench_pipeline_overlap(device):
    separator("Benchmark 5: Pipeline Overlap -- Prefetch Hidden Behind Attention")
    engine = _make_engine(device)
    dag = PrefixExpertDAG(decay_factor=1.0)
    predictor = ExpertPredictor(dag=dag, cache_engine=engine,
        confidence_threshold=0.05, max_prefetch_per_layer=8)
    cold = list(range(30, 38))
    for _ in range(30):
        dag.record([500, 600, 700], layer_id=0, expert_ids=cold)

    def evict_cold():
        for eid in cold:
            e = engine._cache.get((0, eid))
            if e and e.location == ExpertLocation.GPU:
                engine.evict_expert(0, eid)
        torch.cuda.synchronize(device)

    def do_cold_load():
        events = []
        for eid in cold:
            t = engine.get_expert(0, eid)
            if t is None:
                ev = engine.load_expert(0, eid, sync=False)
                if ev is not None:
                    events.append(ev)
        for ev in events:
            ev.synchronize()
        torch.cuda.synchronize(device)
        return len(events)

    raw_times = []
    for _ in range(5):
        evict_cold()
        t0 = time.perf_counter()
        do_cold_load()
        raw_times.append(time.perf_counter() - t0)
    cold_load_ms = statistics.median(raw_times) * 1e3

    expert_mib = len(cold) * _expert_memory_bytes() / 1024**2
    print("  Cold load {} experts ({:.0f} MiB): {:.2f} ms".format(
        len(cold), expert_mib, cold_load_ms))
    print("  Effective PCIe BW: {:.1f} GB/s".format(expert_mib / cold_load_ms))

    hot_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        for eid in cold:
            engine.get_expert(0, eid)
        hot_times.append(time.perf_counter() - t0)
    print("  Hot hit {} experts: {:.1f} us".format(
        len(cold), statistics.median(hot_times)*1e6))
    print()

    print("  {:>8} | {:>11} | {:>10} | {:>8} | {:>9}".format(
        "Attn(ms)", "NoPred(ms)", "Pred(ms)", "Speedup", "Saved(ms)"))
    print("  " + "-"*8 + "-+-" + "-"*11 + "-+-" + "-"*10 + "-+-"
          + "-"*8 + "-+-" + "-"*9)

    for attn_ms in [1, 5, 10, 15, 20, 25, 30]:
        attn_s = attn_ms / 1000.0
        times_a = []
        for _ in range(5):
            evict_cold()
            t0 = time.perf_counter()
            time.sleep(attn_s)
            do_cold_load()
            times_a.append(time.perf_counter() - t0)

        times_b = []
        for _ in range(5):
            evict_cold()
            t0 = time.perf_counter()
            predictor.predict_and_prefetch([500, 600, 700])
            time.sleep(attn_s)
            do_cold_load()
            times_b.append(time.perf_counter() - t0)

        ma = statistics.median(times_a) * 1e3
        mb = statistics.median(times_b) * 1e3
        speedup = ma / mb if mb > 0 else 0
        print("  {:>8} | {:>11.2f} | {:>10.2f} | {:>7.2f}x | {:>9.2f}".format(
            attn_ms, ma, mb, speedup, ma - mb))

    print()
    print("  Cold load = {:.0f} ms (PCIe bottleneck for {} x {:.0f} MiB)".format(
        cold_load_ms, len(cold), _expert_memory_bytes()/1024**2))
    print("  When attention >= cold load, prediction hides ALL transfer latency.")
    del engine; torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("  Phase 3+4 CUDA Benchmark: Prefix DAG + Expert Prediction")
    print("=" * 70)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        print("  GPU: {}".format(props.name))
        print("  VRAM: {:.1f} GiB".format(props.total_memory / 1024**3))
        print("  Expert size: {:.1f} MiB".format(
            _expert_memory_bytes() / 1024**2))
    else:
        print("  WARNING: No CUDA device -- GPU benchmarks skipped")
        device = None

    bench_dag_throughput()
    bench_dag_scalability()
    bench_prediction_accuracy()

    if device is not None:
        bench_prefetch_latency(device)
        bench_pipeline_overlap(device)

    print("\n" + "=" * 70)
    print("  All benchmarks complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
