#!/usr/bin/env python3
"""
python/analytics_report.py — summarize graph_engine stdout (saved log) or re-run the
binary via subprocess, then print a compact metrics table. Optionally draws a small
NetworkX plot for a processed ego-Facebook slice.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt


def run_engine(exe: Path, graph: Path, source: int, topk: int) -> str:
    """Execute graph_engine and capture combined stdout/stderr as text."""
    cmd = [
        str(exe),
        "--graph",
        str(graph),
        "--algorithm",
        "all",
        "--source",
        str(source),
        "--topk",
        str(topk),
        "--mode",
        "auto",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"graph_engine failed ({proc.returncode}): {proc.stderr}\n{proc.stdout}")
    return proc.stdout + proc.stderr


def parse_log(text: str) -> dict[str, str]:
    """Extract high-level metrics from engine log lines."""
    metrics: dict[str, str] = {}

    m = re.search(r"Top-\d+ influencers.*?#1: node (\d+) score=([0-9.eE+-]+)", text, re.S)
    if m:
        metrics["Influencer #1"] = f"node {m.group(1)} (score {m.group(2)})"

    m = re.search(r"Communities summary.*?total=(\d+) largest=(\d+) nodes", text)
    if m:
        metrics["Communities"] = f"{m.group(1)} (largest: {m.group(2)} nodes)"

    hops = re.findall(r"reachable in (\d+) hop\(s\): (\d+) vertices", text)
    if hops:
        total_nodes_m = re.search(r"Graph stats: nodes=(\d+)", text)
        n = int(total_nodes_m.group(1)) if total_nodes_m else None
        within3 = sum(int(c) for h, c in hops if int(h) <= 3)
        if n and n > 0:
            pct = 100.0 * float(within3) / float(n)
            src_m = re.search(r"reachability_report: source=(\d+)", text)
            src = src_m.group(1) if src_m else "?"
            metrics[f"BFS from node {src} (≤3 hops)"] = f"{pct:.2f}% of graph ({within3}/{n})"

    return metrics


def maybe_plot_facebook(processed_fb: Path | None, out_png: Path) -> None:
    if processed_fb is None or not processed_fb.is_file():
        return
    g = nx.read_edgelist(processed_fb, nodetype=int, data=False, comments="#")
    if g.number_of_nodes() > 400:
        nodes = list(g.nodes())
        keep = set(nodes[:400])
        g = g.subgraph(keep).copy()

    pos = nx.spring_layout(g, seed=0, iterations=50)
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx(g, pos=pos, node_size=15, width=0.3, with_labels=False, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analytics summary from logs or subprocess.")
    parser.add_argument("--log", type=Path, default=None, help="Path to saved graph_engine output")
    parser.add_argument("--exe", type=Path, default=None, help="Path to graph_engine binary to re-run")
    parser.add_argument("--graph", type=Path, default=None, help="Graph path when using --exe")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--facebook",
        type=Path,
        default=None,
        help="Optional processed facebook edgelist for a small NetworkX render",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "facebook_snippet.png",
    )
    args = parser.parse_args()

    if args.log is not None:
        text = args.log.read_text(encoding="utf-8", errors="replace")
    elif args.exe is not None and args.graph is not None:
        text = run_engine(args.exe, args.graph, args.source, args.topk)
    else:
        print("error: provide --log <file> OR both --exe and --graph", file=sys.stderr)
        return 1

    metrics = parse_log(text)
    print("Metric | Value")
    print("--- | ---")
    for k in sorted(metrics.keys()):
        print(f"{k} | {metrics[k]}")

    try:
        maybe_plot_facebook(args.facebook, args.plot_out)
        if args.facebook and args.facebook.is_file():
            print(f"Wrote plot: {args.plot_out}")
    except Exception as exc:  # pragma: no cover - visualization is best-effort
        print(f"(plot skipped: {exc})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
