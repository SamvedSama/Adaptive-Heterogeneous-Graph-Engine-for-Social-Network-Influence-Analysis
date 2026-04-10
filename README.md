# Adaptive Heterogeneous Graph Engine for Social Network Influence Analysis

Semester project: a small C++/CUDA/OpenMP graph engine with an adaptive scheduler, analytics layer, benchmarks, and Python helpers for SNAP-style edge lists.

## Layout

- `include/` — shared headers (`common.h`, `timer.h`)
- `src/graph/` — CSR `Graph`, loader, statistics
- `src/algorithms/` — BFS, PageRank, WCC (sequential, OpenMP, CUDA)
- `src/scheduler/` — workload profile + heuristic `ExecutionMode` selection
- `src/analytics/` — reachability, influencers, communities orchestration
- `benchmarks/` — CSV benchmark harness
- `python/` — preprocessing, plots, log summaries
- `data/` — raw/processed datasets (`download.sh` helper)

## Build (Google Colab, T4, `/content` layout)

The Colab-oriented flow matches the comment block in `src/main.cpp`:

```bash
apt-get update -qq && apt-get install -y -qq cmake libgomp1 build-essential
nvcc --version
cd /content/adaptive-graph-engine
mkdir -p build && cd build
cmake ..
cmake --build . -j4
./graph_engine --graph ../data/processed/facebook_combined.txt --algorithm all --source 0 --topk 10 --mode auto
```

CUDA is targeted at **sm_75** (T4) via `CMAKE_CUDA_ARCHITECTURES=75` and `-O3` for both host and device code.

## Build (local, out-of-source)

```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

Run the benchmark harness (writes `benchmarks/results/results.csv`; from `build/`, the runner falls back to `../benchmarks/results/` when needed):

```bash
./benchmark_runner --graph ../data/processed/facebook_combined.txt --dataset facebook --trials 3
```

## CLI (`graph_engine`)

- `--graph <path>` — edge list (comments allowed; see loader)
- `--algorithm bfs|pagerank|wcc|all`
- `--source <id>` — BFS source (remapped index)
- `--topk <k>` — influencer list length
- `--mode auto|seq|omp|cuda` — override adaptive policy

## Python

```bash
pip install -r requirements.txt
python python/preprocess.py data/facebook_combined.txt
python python/visualize.py --csv benchmarks/results/results.csv
python python/analytics_report.py --log my_run.log
# or: python python/analytics_report.py --exe ./build/graph_engine --graph data/processed/facebook_combined.txt
```

Standard Colab images already ship `matplotlib`, `pandas`, and `networkx`; `requirements.txt` pins minimal versions for reproducibility.

## Data (verified SNAP links)

There are **no ML model downloads** in this project—only graph edge lists. The helper script and Makefile use these **official Stanford SNAP** files (each link appears on the dataset’s HTML page in the “Files” table):

| Dataset | Dataset page | Combined edge list (gzip) |
|--------|--------------|---------------------------|
| ego-Facebook | [ego-Facebook.html](https://snap.stanford.edu/data/ego-Facebook.html) | [facebook_combined.txt.gz](https://snap.stanford.edu/data/facebook_combined.txt.gz) |
| ego-Twitter | [ego-Twitter.html](https://snap.stanford.edu/data/ego-Twitter.html) | [twitter_combined.txt.gz](https://snap.stanford.edu/data/twitter_combined.txt.gz) |
| ego-G+ | [ego-Gplus.html](https://snap.stanford.edu/data/ego-Gplus.html) | [gplus_combined.txt.gz](https://snap.stanford.edu/data/gplus_combined.txt.gz) |

Shared readme: [readme-Ego.txt](https://snap.stanford.edu/data/readme-Ego.txt).

Run `data/download.sh` on a Unix shell (or `make download`) to pull the three `.gz` files into `data/`, then preprocess into `data/processed/`.

## Recommended order of execution

1. **Dependencies**: CMake, C++17 compiler, CUDA toolkit (for `nvcc` / device code), OpenMP runtime (`libgomp1` on Debian/Ubuntu/Colab).
2. **`make download`** (or `bash data/download.sh`) — downloads SNAP archives.
3. **`python3 python/preprocess.py data/<name>_combined.txt`** — writes remapped edge lists to `data/processed/`.
4. **`make build`** — out-of-source build in `build/`.
5. **`make run`** — loads the graph, runs analytics with scheduler-chosen backends; logs lines prefixed with `[compute]` and `[report]`.
6. **`make run-compare`** — same as run but adds **`--compare`**: times **SEQ, OMP, CUDA** (if a GPU exists) per phase and prints **speedup vs SEQ**.
7. **`make benchmark`** — runs all backends × trials; each trial prints `[compute] … starting/finished` and a final **median speedup summary** per algorithm.
8. **`make visualize`** — reads `benchmarks/results/results.csv` and writes PNGs under `benchmarks/results/`.

Convenience: **`make all`** runs **build** then **run** (not benchmark).

## Terminal output (computation phases)

- **`graph_engine`**: Each of BFS / PageRank / WCC prints `[compute] … starting backend=…` and `… total time: … ms`. With **`--compare`**, you also get a **speedup table** (SEQ baseline) after all three backends for that phase.
- **`benchmark_runner`**: Per kernel and trial, `[compute] <name> trial … starting/finished: … ms`; at the end, **`[benchmark] === Median speedup summary ===`** for `bfs`, `pagerank`, and `wcc`.

## Notes

- All CUDA API calls use the `checkCuda` macro from `include/common.h` (enabled when `GRAPH_ENGINE_BUILD_CUDA` is defined).
- The scheduler thresholds are documented in `src/scheduler/scheduler.cpp` and echoed to stdout with a reason string at runtime.
