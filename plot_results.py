"""
Plot benchmark results from benchmark.py output.

Usage:
    python benchmark.py ... > results.json   # add --json flag first
    python plot_results.py results.json
"""

import json
import sys
import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy")
    sys.exit(1)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_ttft_throughput(data: dict, out: str = "results.png"):
    concurrencies = sorted(data.keys(), key=int)
    systems = list(next(iter(data.values())).keys())  # e.g. ["cache-stack", "vllm"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("cache-stack vs vLLM — Qwen2.5-0.5B-Instruct on MPS", fontsize=13)

    x = np.arange(len(concurrencies))
    width = 0.35 / max(len(systems), 1)

    # TTFT
    ax = axes[0]
    for i, sys_name in enumerate(systems):
        ttfts = [data[c][sys_name]["ttft_avg_ms"] for c in concurrencies]
        ax.bar(x + i * width, ttfts, width, label=sys_name, color=colors[i % len(colors)])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(concurrencies)
    ax.legend()

    # TPOT
    ax = axes[1]
    for i, sys_name in enumerate(systems):
        tpots = [data[c][sys_name].get("tpot_avg_ms", 0) for c in concurrencies]
        ax.bar(x + i * width, tpots, width, label=sys_name, color=colors[i % len(colors)])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("Time per Output Token")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(concurrencies)
    ax.legend()

    # Throughput
    ax = axes[2]
    for i, sys_name in enumerate(systems):
        tputs = [data[c][sys_name]["throughput_tps"] for c in concurrencies]
        ax.bar(x + i * width, tputs, width, label=sys_name, color=colors[i % len(colors)])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Total Throughput")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(concurrencies)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="JSON file from benchmark.py --json")
    parser.add_argument("--out", default="results.png")
    args = parser.parse_args()

    data = load_results(args.results_json)
    plot_ttft_throughput(data, args.out)
