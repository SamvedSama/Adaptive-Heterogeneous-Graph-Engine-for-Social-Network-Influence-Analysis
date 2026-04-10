#!/usr/bin/env python3
"""
python/visualize.py — plot benchmark CSV timings (grouped bars) and OMP/CUDA speedups
relative to the sequential baseline per algorithm.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize benchmark CSV results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "results.csv",
        help="Path to results.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "results",
        help="Directory for PNG outputs",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"error: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required = {"algorithm", "mode", "time_ms"}
    if not required.issubset(df.columns):
        print(f"error: CSV must contain columns {sorted(required)}", file=sys.stderr)
        return 1

    # Use median time per (algorithm, mode) across trials for cleaner plots.
    grouped = df.groupby(["algorithm", "mode"], as_index=False)["time_ms"].median()

    algorithms = sorted(grouped["algorithm"].unique())
    modes = ["seq", "omp", "cuda"]
    pivot = grouped.pivot(index="algorithm", columns="mode", values="time_ms")
    pivot = pivot.reindex(algorithms)
    for m in modes:
        if m not in pivot.columns:
            pivot[m] = float("nan")

    x = range(len(algorithms))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mode in enumerate(modes):
        vals = [pivot.loc[a, mode] if a in pivot.index else float("nan") for a in algorithms]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width=width, label=mode)

    ax.set_xticks(list(x))
    ax.set_xticklabels(algorithms)
    ax.set_ylabel("time_ms (median)")
    ax.set_title("Benchmark runtime by algorithm and mode")
    ax.legend()
    fig.tight_layout()
    bar_path = args.outdir / "benchmark_times.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)

    speed_rows = []
    for algo in algorithms:
        row = pivot.loc[algo]
        base = row.get("seq", float("nan"))
        if base is None or (isinstance(base, float) and (base != base or base == 0)):
            continue
        for mode in ("omp", "cuda"):
            t = row.get(mode, float("nan"))
            if t is None or (isinstance(t, float) and t != t):
                continue
            speed_rows.append(
                {"algorithm": algo, "mode": mode, "speedup": float(base) / float(t)}
            )

    if speed_rows:
        spd = pd.DataFrame(speed_rows)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        algos = sorted(spd["algorithm"].unique())
        w = 0.35
        x2 = range(len(algos))

        def median_speedup(algo: str, mode: str) -> float:
            sub = spd[(spd["algorithm"] == algo) & (spd["mode"] == mode)]
            if sub.empty:
                return float("nan")
            return float(sub["speedup"].median())

        omp_vals = [median_speedup(a, "omp") for a in algos]
        cuda_vals = [median_speedup(a, "cuda") for a in algos]
        ax2.bar([i - w / 2 for i in x2], omp_vals, width=w, label="OMP vs SEQ")
        ax2.bar([i + w / 2 for i in x2], cuda_vals, width=w, label="CUDA vs SEQ")
        ax2.set_xticks(list(x2))
        ax2.set_xticklabels(algos)
        ax2.axhline(1.0, color="gray", linewidth=1.0, linestyle="--")
        ax2.set_ylabel("speedup")
        ax2.set_title("Speedup over sequential baseline (median)")
        ax2.legend()
        fig2.tight_layout()
        spd_path = args.outdir / "benchmark_speedup.png"
        fig2.savefig(spd_path, dpi=150)
        plt.close(fig2)

    print(f"Wrote {bar_path}")
    if speed_rows:
        print(f"Wrote {args.outdir / 'benchmark_speedup.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
