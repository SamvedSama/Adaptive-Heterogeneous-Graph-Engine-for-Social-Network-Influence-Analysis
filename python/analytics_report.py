#!/usr/bin/env python3
"""
python/analytics_report.py — summarize graph_engine stdout (saved log) or re-run the
binary via subprocess, then print a compact metrics table.  Optionally draws a small
NetworkX plot for a processed ego-Facebook slice.

Fixes vs original:
  - BFS source parsed from "From node N:" (actual log format) not the non-existent
    "reachability_report: source=N" pattern — source node no longer shows as "?".
  - Communities parsed from "Communities: total=N largest=M nodes" (actual log format)
    not the non-existent "Communities summary" pattern.
  - Empty / missing log file detected early with a clear warning instead of a silent
    empty table.
  - Metrics printed in a fixed, readable order (BFS → Influencer → Communities → WCC
    timing) rather than alphabetical sort of dict keys, which broke when the BFS key
    contained the source-node number.
  - Additional timing metrics extracted: per-algorithm backend timing and speedup lines
    are now parsed and included in the report so the table is meaningfully populated
    even for the --compare logs.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Engine runner
# ---------------------------------------------------------------------------

def run_engine(exe: Path, graph: Path, source: int, topk: int) -> str:
    """Execute graph_engine and capture combined stdout/stderr as text."""
    cmd = [
        str(exe), "--graph", str(graph),
        "--algorithm", "all",
        "--source", str(source),
        "--topk", str(topk),
        "--mode", "auto",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"graph_engine failed ({proc.returncode}): {proc.stderr}\n{proc.stdout}"
        )
    return proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

def parse_log(text: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Returns:
        metrics  — ordered list of (key, value) pairs for the main table
        warnings — list of warning strings to print to stderr
    """
    metrics: List[Tuple[str, str]] = []
    warnings: List[str] = []

    if not text.strip():
        warnings.append("Log is empty — no metrics to report.")
        return {}, warnings

    # -- Graph size -----------------------------------------------------------
    m = re.search(r"Graph stats: nodes=(\d+)\s+edges=(\d+)", text)
    if m:
        metrics.append(("Graph nodes", m.group(1)))
        metrics.append(("Graph edges", m.group(2)))

    # -- BFS reachability -----------------------------------------------------
    # Actual log format:  "From node N:\n  reachable in K hop(s): C vertices"
    # Source node is on the "From node N:" line directly above the hop table.
    src_m = re.search(r"From node (\d+):", text)
    hops  = re.findall(r"reachable in (\d+) hop\(s\): (\d+) vertices", text)
    if hops:
        n_nodes_m = re.search(r"Graph stats: nodes=(\d+)", text)
        n = int(n_nodes_m.group(1)) if n_nodes_m else None
        within3 = sum(int(c) for h, c in hops if int(h) <= 3)
        src = src_m.group(1) if src_m else "0"   # default 0 matches CLI default
        if n and n > 0:
            pct = 100.0 * within3 / n
            metrics.append((
                f"BFS from node {src} (≤3 hops)",
                f"{pct:.2f}% of graph ({within3}/{n})"
            ))
        # Full hop histogram
        for h, c in hops:
            metrics.append((f"  reachable in {h} hop(s)", f"{c} vertices"))

    # -- BFS backend timing (from --compare output) ---------------------------
    bfs_times = re.findall(
        r"\[compute\] BFS (\w+) finished: ([0-9.]+) ms", text
    )
    for backend, ms in bfs_times:
        metrics.append((f"BFS {backend.upper()} time", f"{ms} ms"))

    # -- PageRank influencers -------------------------------------------------
    m = re.search(
        r"Top-\d+ influencers.*?#1: node (\d+) score=([0-9.eE+\-]+)", text, re.S
    )
    if m:
        metrics.append(("Influencer #1", f"node {m.group(1)} (score {m.group(2)})"))

    # Full top-k list
    influencers = re.findall(r"#(\d+): node (\d+) score=([0-9.eE+\-]+)", text)
    for rank, node, score in influencers:
        if rank != "1":   # #1 already captured above
            metrics.append((f"  Influencer #{rank}", f"node {node} (score {score})"))

    # -- PageRank backend timing ----------------------------------------------
    pr_times = re.findall(
        r"\[compute\] PageRank (\w+) finished: ([0-9.]+) ms", text
    )
    for backend, ms in pr_times:
        metrics.append((f"PageRank {backend.upper()} time", f"{ms} ms"))

    # -- WCC communities ------------------------------------------------------
    # Actual log format:  "Communities: total=N largest=M nodes"
    m = re.search(r"Communities: total=(\d+) largest=(\d+) nodes", text)
    if m:
        metrics.append(("Communities total", m.group(1)))
        metrics.append(("Largest community", f"{m.group(2)} nodes"))
    else:
        warnings.append("Communities line not found in log.")

    # -- WCC backend timing ---------------------------------------------------
    wcc_times = re.findall(
        r"\[compute\] WCC (\w+) finished: ([0-9.]+) ms", text
    )
    for backend, ms in wcc_times:
        metrics.append((f"WCC {backend.upper()} time", f"{ms} ms"))

    # -- Total wall time ------------------------------------------------------
    m = re.search(r"Total wall time: ([0-9.]+) ms", text)
    if m:
        metrics.append(("Total wall time", f"{m.group(1)} ms"))

    if not metrics:
        warnings.append("No recognisable metrics found in log.")

    return metrics, warnings


# ---------------------------------------------------------------------------
# Optional NetworkX plot
# ---------------------------------------------------------------------------

def maybe_plot_facebook(processed_fb: Path | None, out_png: Path) -> None:
    if processed_fb is None or not processed_fb.is_file():
        return
    g = nx.read_edgelist(processed_fb, nodetype=int, data=False, comments="#")
    if g.number_of_nodes() > 400:
        nodes = list(g.nodes())[:400]
        g = g.subgraph(nodes).copy()

    pos = nx.spring_layout(g, seed=0, iterations=50)
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx(g, pos=pos, node_size=15, width=0.3, with_labels=False, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analytics summary from graph_engine logs."
    )
    parser.add_argument("--log",      type=Path, default=None,
                        help="Path to saved graph_engine output")
    parser.add_argument("--exe",      type=Path, default=None,
                        help="Path to graph_engine binary to re-run")
    parser.add_argument("--graph",    type=Path, default=None,
                        help="Graph path when using --exe")
    parser.add_argument("--source",   type=int,  default=0)
    parser.add_argument("--topk",     type=int,  default=5)
    parser.add_argument("--facebook", type=Path, default=None,
                        help="Optional processed facebook edgelist for NetworkX render")
    parser.add_argument("--plot-out", type=Path,
                        default=Path(__file__).resolve().parents[1]
                                / "benchmarks" / "results" / "facebook_snippet.png")
    args = parser.parse_args()

    # -- Obtain log text ------------------------------------------------------
    if args.log is not None:
        log_path = args.log
        if not log_path.exists():
            print(f"error: log file not found: {log_path}", file=sys.stderr)
            return 1
        text = log_path.read_text(encoding="utf-8", errors="replace")
    elif args.exe is not None and args.graph is not None:
        try:
            text = run_engine(args.exe, args.graph, args.source, args.topk)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        print("error: provide --log <file> OR both --exe and --graph", file=sys.stderr)
        return 1

    # -- Parse and print ------------------------------------------------------
    metrics, warnings = parse_log(text)

    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    print("Metric | Value")
    print("--- | ---")
    for key, val in metrics:
        print(f"{key} | {val}")

    # -- Optional plot --------------------------------------------------------
    try:
        maybe_plot_facebook(args.facebook, args.plot_out)
        if args.facebook and args.facebook.is_file():
            print(f"Wrote plot: {args.plot_out}")
    except Exception as exc:
        print(f"(plot skipped: {exc})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
