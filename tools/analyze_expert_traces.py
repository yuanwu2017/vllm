#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert activation trace analysis and paper figure generation.

This script processes expert activation traces collected during inference
and generates paper-ready figures for the motivation section:
1. Expert usage heatmap (layer × expert)
2. CDF of expert usage frequency
3. Top-K coverage chart
4. Temporal locality analysis
5. Gini coefficient per layer

Usage:
    python analyze_expert_traces.py \
        --trace-file expert_traces.jsonl \
        --output-dir figures/ \
        --model-name "Qwen1.5-MoE-A2.7B"
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_traces(trace_file: str) -> list[dict]:
    """Load expert activation traces from JSONL file."""
    records = []
    with open(trace_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} trace records from {trace_file}")
    return records


def compute_frequency(records: list[dict]) -> dict:
    """Compute per-layer expert frequency distributions."""
    layer_freq: dict[int, Counter] = defaultdict(Counter)
    global_freq: Counter = Counter()
    total_tokens = 0

    for record in records:
        layer_id = record["layer_id"]
        expert_ids = record["expert_ids"]
        num_tokens = record["num_tokens"]
        total_tokens += num_tokens
        layer_freq[layer_id].update(expert_ids)
        global_freq.update(expert_ids)

    return {
        "layer_freq": dict(layer_freq),
        "global_freq": dict(global_freq),
        "total_tokens": total_tokens,
        "num_layers": len(layer_freq),
    }


def compute_gini(values: list[int]) -> float:
    """Compute Gini coefficient for a distribution of values."""
    if not values or sum(values) == 0:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)
    weighted_sum = sum((2 * (i + 1) - n - 1) * v
                       for i, v in enumerate(sorted_values))
    return weighted_sum / (n * total)


def compute_top_k_coverage(freq: dict[int, int]) -> dict[int, float]:
    """Compute what fraction of traffic is covered by top-K% of experts."""
    if not freq:
        return {}
    total = sum(freq.values())
    sorted_counts = sorted(freq.values(), reverse=True)
    n = len(sorted_counts)
    coverage = {}
    for k_pct in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]:
        k = max(1, n * k_pct // 100)
        covered = sum(sorted_counts[:k])
        coverage[k_pct] = covered / total
    return coverage


def compute_temporal_locality(records: list[dict]) -> dict:
    """Compute temporal locality metrics (Jaccard overlap between consecutive steps)."""
    prev_by_layer: dict[int, set] = {}
    overlaps_by_layer: dict[int, list[float]] = defaultdict(list)

    for record in records:
        layer_id = record["layer_id"]
        current = set(record["expert_ids"])

        if layer_id in prev_by_layer:
            prev = prev_by_layer[layer_id]
            if prev and current:
                jaccard = len(prev & current) / len(prev | current)
                overlaps_by_layer[layer_id].append(jaccard)

        prev_by_layer[layer_id] = current

    avg_locality = {}
    for layer_id, overlaps in overlaps_by_layer.items():
        avg_locality[layer_id] = float(np.mean(overlaps))

    global_avg = float(np.mean([v for vs in overlaps_by_layer.values()
                                 for v in vs])) if overlaps_by_layer else 0.0

    return {
        "per_layer": avg_locality,
        "global_avg": global_avg,
    }


def generate_figures(
    freq_data: dict,
    temporal: dict,
    model_name: str,
    output_dir: Path,
) -> None:
    """Generate paper-ready figures using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("matplotlib not available — skipping figure generation")
        print("Install: pip install matplotlib")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    layer_freq = freq_data["layer_freq"]
    global_freq = freq_data["global_freq"]

    # Set style
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "figure.dpi": 150,
    })

    # --- Figure 1: Expert Usage Heatmap ---
    if layer_freq:
        layers = sorted(layer_freq.keys())
        all_experts = sorted(set(
            eid for counter in layer_freq.values()
            for eid in counter
        ))

        heatmap = np.zeros((len(layers), len(all_experts)))
        for i, lid in enumerate(layers):
            for j, eid in enumerate(all_experts):
                heatmap[i, j] = layer_freq[lid].get(eid, 0)

        fig, ax = plt.subplots(figsize=(14, 6))
        # Use log scale for better visualization of skewed distributions
        im = ax.imshow(
            heatmap + 1,  # +1 for log scale
            aspect="auto",
            cmap="YlOrRd",
            norm=LogNorm(),
            interpolation="nearest",
        )
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Layer ID")
        ax.set_title(f"Expert Usage Heatmap — {model_name}")
        plt.colorbar(im, ax=ax, label="Activation Count (log)")

        # Sparse tick labels if too many
        if len(all_experts) > 30:
            tick_step = len(all_experts) // 15
            ax.set_xticks(range(0, len(all_experts), tick_step))
            ax.set_xticklabels(all_experts[::tick_step])
        else:
            ax.set_xticks(range(len(all_experts)))
            ax.set_xticklabels(all_experts)

        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)

        plt.tight_layout()
        fig.savefig(output_dir / "expert_heatmap.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "expert_heatmap.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Generated: {output_dir}/expert_heatmap.pdf")

    # --- Figure 2: CDF of Expert Usage Frequency ---
    if global_freq:
        counts = sorted(global_freq.values())
        cdf = np.arange(1, len(counts) + 1) / len(counts)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(counts, cdf, linewidth=2, color="#2196F3")
        ax.set_xlabel("Activation Count")
        ax.set_ylabel("CDF (Fraction of Experts)")
        ax.set_title(f"CDF of Expert Usage — {model_name}")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()
        fig.savefig(output_dir / "expert_cdf.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "expert_cdf.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Generated: {output_dir}/expert_cdf.pdf")

    # --- Figure 3: Top-K Coverage ---
    coverage = compute_top_k_coverage(global_freq)
    if coverage:
        k_values = sorted(coverage.keys())
        cov_values = [coverage[k] * 100 for k in k_values]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(k_values)), cov_values, color="#4CAF50", alpha=0.8)
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([f"{k}%" for k in k_values])
        ax.set_xlabel("Top-K% of Experts")
        ax.set_ylabel("Traffic Coverage (%)")
        ax.set_title(f"Top-K Expert Coverage — {model_name}")
        ax.set_ylim(0, 105)

        # Add value labels on bars
        for i, v in enumerate(cov_values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.axhline(y=80, color="red", linestyle="--", alpha=0.5, label="80% line")
        ax.legend()

        plt.tight_layout()
        fig.savefig(output_dir / "expert_topk_coverage.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "expert_topk_coverage.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Generated: {output_dir}/expert_topk_coverage.pdf")

    # --- Figure 4: Gini Coefficient per Layer ---
    if layer_freq:
        layers = sorted(layer_freq.keys())
        gini_values = []
        for lid in layers:
            values = list(layer_freq[lid].values())
            gini_values.append(compute_gini(values))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(layers)), gini_values, color="#FF9800", alpha=0.8)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45 if len(layers) > 20 else 0)
        ax.set_xlabel("Layer ID")
        ax.set_ylabel("Gini Coefficient")
        ax.set_title(f"Expert Usage Skewness (Gini) — {model_name}")
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5,
                    label="Moderate inequality")
        ax.legend()

        plt.tight_layout()
        fig.savefig(output_dir / "expert_gini.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "expert_gini.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Generated: {output_dir}/expert_gini.pdf")

    # --- Figure 5: Temporal Locality per Layer ---
    if temporal["per_layer"]:
        layers = sorted(temporal["per_layer"].keys())
        locality_values = [temporal["per_layer"][lid] for lid in layers]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(layers)), locality_values, color="#9C27B0", alpha=0.8)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45 if len(layers) > 20 else 0)
        ax.set_xlabel("Layer ID")
        ax.set_ylabel("Jaccard Overlap (consecutive steps)")
        ax.set_title(f"Expert Temporal Locality — {model_name}")
        ax.set_ylim(0, 1.0)
        ax.axhline(y=temporal["global_avg"], color="red", linestyle="--",
                    alpha=0.5, label=f"Global avg: {temporal['global_avg']:.3f}")
        ax.legend()

        plt.tight_layout()
        fig.savefig(output_dir / "expert_temporal.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "expert_temporal.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Generated: {output_dir}/expert_temporal.pdf")


def generate_summary_table(
    freq_data: dict,
    temporal: dict,
    model_name: str,
    output_dir: Path,
) -> None:
    """Generate summary statistics as a markdown table."""
    global_freq = freq_data["global_freq"]
    coverage = compute_top_k_coverage(global_freq)
    num_experts = len(global_freq)
    total_activations = sum(global_freq.values())

    lines = [
        f"# Expert Activation Analysis — {model_name}\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tokens processed | {freq_data['total_tokens']:,} |",
        f"| Number of layers | {freq_data['num_layers']} |",
        f"| Unique experts (global) | {num_experts} |",
        f"| Total activations | {total_activations:,} |",
        f"| Temporal locality (Jaccard) | {temporal['global_avg']:.4f} |",
        f"",
        f"## Top-K Coverage\n",
        f"| Top-K% Experts | Traffic Coverage |",
        f"|----------------|-----------------|",
    ]
    for k, cov in sorted(coverage.items()):
        lines.append(f"| {k}% | {cov*100:.1f}% |")

    lines.extend([
        f"",
        f"## Gini Coefficients\n",
        f"| Layer | Gini |",
        f"|-------|------|",
    ])
    layer_freq = freq_data["layer_freq"]
    for lid in sorted(layer_freq.keys()):
        values = list(layer_freq[lid].values())
        gini = compute_gini(values)
        lines.append(f"| {lid} | {gini:.4f} |")

    summary_path = output_dir / "analysis_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MoE expert activation traces and generate paper figures"
    )
    parser.add_argument(
        "--trace-file", required=True,
        help="Path to expert_traces.jsonl"
    )
    parser.add_argument(
        "--output-dir", default="figures",
        help="Directory for output figures and summary"
    )
    parser.add_argument(
        "--model-name", default="MoE Model",
        help="Model name for figure titles"
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation (summary only)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load and analyze
    records = load_traces(args.trace_file)
    if not records:
        print("No records found. Exiting.")
        sys.exit(1)

    freq_data = compute_frequency(records)
    temporal = compute_temporal_locality(records)

    # Print summary
    coverage = compute_top_k_coverage(freq_data["global_freq"])
    print(f"\n{'='*60}")
    print(f"Expert Activation Analysis — {args.model_name}")
    print(f"{'='*60}")
    print(f"Total tokens: {freq_data['total_tokens']:,}")
    print(f"Layers: {freq_data['num_layers']}")
    print(f"Unique experts: {len(freq_data['global_freq'])}")
    print(f"Temporal locality (Jaccard): {temporal['global_avg']:.4f}")
    print(f"\nTop-K Coverage:")
    for k in [10, 20, 30, 50]:
        if k in coverage:
            print(f"  Top {k}% experts → {coverage[k]*100:.1f}% traffic")

    # Generate outputs
    generate_summary_table(freq_data, temporal, args.model_name, output_dir)

    if not args.no_figures:
        generate_figures(freq_data, temporal, args.model_name, output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
