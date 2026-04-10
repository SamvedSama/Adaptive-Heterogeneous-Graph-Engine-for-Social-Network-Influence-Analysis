#!/usr/bin/env python3
"""
python/preprocess.py — clean SNAP edge lists: strip comments, remap IDs to a dense
0-based range, drop self-loops, and write processed files under data/processed/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def load_edges(path: Path) -> tuple[list[tuple[int, int]], bool]:
    """Parse edges; returns (edges, saw_undirected_hint)."""
    edges: list[tuple[int, int]] = []
    undirected_hint = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                low = s.lower()
                if "undirected" in low:
                    undirected_hint = True
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
    return edges, undirected_hint


def remap_and_write(edges: list[tuple[int, int]], out_path: Path) -> tuple[int, int, float]:
    """Remap raw IDs to 0..N-1, remove self-loops, return (N, M, density)."""
    id_map: dict[int, int] = {}
    next_id = 0

    def mid(x: int) -> int:
        nonlocal next_id
        if x not in id_map:
            id_map[x] = next_id
            next_id += 1
        return id_map[x]

    canon: list[tuple[int, int]] = []
    self_loops = 0
    for u, v in edges:
        if u == v:
            self_loops += 1
            continue
        canon.append((mid(u), mid(v)))

    canon = sorted(set(canon))
    n = next_id
    m = len(canon)
    density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        out.write("# undirected edge list; vertices are 0-indexed contiguous ints\n")
        for u, v in canon:
            out.write(f"{u} {v}\n")

    return n, m, density


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess SNAP edge lists.")
    parser.add_argument("input", type=Path, help="Raw SNAP edge list path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: data/processed/<input_stem>.txt next to repo root)",
    )
    args = parser.parse_args()

    inp: Path = args.input
    if not inp.is_file():
        print(f"error: input not found: {inp}", file=sys.stderr)
        return 1

    edges, hint = load_edges(inp)
    if not edges:
        print("error: no edges parsed", file=sys.stderr)
        return 1

    if args.output is not None:
        out_path = args.output
    else:
        root = Path(__file__).resolve().parents[1]
        out_path = root / "data" / "processed" / f"{inp.stem}.txt"

    n, m, density = remap_and_write(edges, out_path)

    print(f"Input: {inp}")
    print(f"Output: {out_path}")
    print(f"Undirected hint in comments: {hint}")
    print(f"N (nodes) = {n}")
    print(f"M (unique edges after remap, no self-loops) = {m}")
    print(f"Density (undirected approx) = {density:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
