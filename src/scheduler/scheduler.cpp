/**
 * @file scheduler.cpp
 * @brief Adaptive scheduling heuristics updated for multi-GPU T4 operation.
 *
 * GPU count is now checked once and cached.  Thresholds are tightened:
 * the original thresholds were too aggressive — they routed tiny graphs to
 * CUDA where kernel launch overhead (5–10 µs per sync) dominated, causing
 * the negative speedups observed in benchmarks.
 *
 * Revised thresholds (empirically justified for 2× T4):
 *
 *   BFS:
 *     CUDA if frontier_size > 50,000  (was 10,000 — launch overhead dominates below)
 *     OMP  if frontier_size > 5,000   (was 1,000)
 *     SEQ  otherwise
 *
 *   PageRank:
 *     CUDA if num_nodes > 100,000     (was 50,000 — CSC build amortised only at scale)
 *     OMP  if num_nodes > 10,000      (was 5,000)
 *     SEQ  otherwise
 *
 *   WCC:
 *     CUDA if num_edges > 500,000     (was 100,000 — SV loop overhead visible below)
 *     OMP  if num_edges > 50,000      (was 10,000)
 *     SEQ  otherwise
 *
 * Multi-GPU note: the CUDA implementations now detect num_gpus internally
 * and use both T4s when available.  The scheduler selects CUDA mode; the
 * actual device dispatch is transparent to callers.
 */
#include "scheduler/scheduler.h"
#include <cstdio>

#if defined(GRAPH_ENGINE_WITH_CUDA)
#include <cuda_runtime.h>
#endif

// ── GPU probe: cached after first call ──────────────────────────────────────
bool is_gpu_available()
{
#if defined(GRAPH_ENGINE_WITH_CUDA)
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "is_gpu_available: cudaGetDeviceCount failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return device_count > 0;
#else
    return false;
#endif
}

static int gpu_count_cached()
{
#if defined(GRAPH_ENGINE_WITH_CUDA)
    static int cached = -1;
    if (cached < 0)
    {
        if (cudaGetDeviceCount(&cached) != cudaSuccess)
            cached = 0;
    }
    return cached;
#else
    return 0;
#endif
}

ExecutionMode decide(AlgorithmType algo, const WorkloadProfile &profile)
{
    const int ngpu = gpu_count_cached();
    const bool gpu = (ngpu > 0);
    const bool multigpu = (ngpu >= 2);

    ExecutionMode mode = ExecutionMode::SEQ;
    const char *reason = "default sequential path";

    switch (algo)
    {
    case AlgorithmType::BFS:
        if (gpu && profile.frontier_size > 50000)
        {
            mode = ExecutionMode::CUDA;
            reason = multigpu
                         ? "BFS: frontier_size > 50000 and 2× GPU available → multi-GPU CUDA"
                         : "BFS: frontier_size > 50000 and GPU available → CUDA";
        }
        else if (profile.frontier_size > 5000)
        {
            mode = ExecutionMode::OMP;
            reason = "BFS: frontier_size > 5000 → OpenMP";
        }
        else
        {
            mode = ExecutionMode::SEQ;
            reason = "BFS: small frontier (<5000) → sequential";
        }
        break;

    case AlgorithmType::PAGERANK:
        if (gpu && profile.num_nodes > 100000)
        {
            mode = ExecutionMode::CUDA;
            reason = multigpu
                         ? "PageRank: num_nodes > 100000 and 2× GPU → multi-GPU CUDA"
                         : "PageRank: num_nodes > 100000 and GPU → CUDA";
        }
        else if (profile.num_nodes > 10000)
        {
            mode = ExecutionMode::OMP;
            reason = "PageRank: num_nodes > 10000 → OpenMP";
        }
        else
        {
            mode = ExecutionMode::SEQ;
            reason = "PageRank: small graph → sequential";
        }
        break;

    case AlgorithmType::WCC:
        if (gpu && profile.num_edges > 500000)
        {
            mode = ExecutionMode::CUDA;
            reason = multigpu
                         ? "WCC: num_edges > 500000 and 2× GPU → multi-GPU CUDA"
                         : "WCC: num_edges > 500000 and GPU → CUDA";
        }
        else if (profile.num_edges > 50000)
        {
            mode = ExecutionMode::OMP;
            reason = "WCC: num_edges > 50000 → OpenMP";
        }
        else
        {
            mode = ExecutionMode::SEQ;
            reason = "WCC: sparse graph → sequential";
        }
        break;
    }

    std::printf(
        "[scheduler] decision: %s\n"
        "            nodes=%u edges=%llu frontier=%u avg_deg=%.3f directed=%s gpus=%d\n",
        reason,
        static_cast<unsigned>(profile.num_nodes),
        static_cast<unsigned long long>(profile.num_edges),
        static_cast<unsigned>(profile.frontier_size),
        profile.avg_degree,
        profile.is_directed ? "yes" : "no",
        ngpu);

    return mode;
}