/**
 * @file scheduler.cpp
 * @brief Adaptive scheduling heuristics:
 *
 *        BFS:
 *          - CUDA if GPU is available AND frontier_size > 10,000
 *          - else OMP if frontier_size > 1,000
 *          - else SEQ
 *
 *        PageRank:
 *          - CUDA if num_nodes > 50,000 (and GPU available)
 *          - else OMP if num_nodes > 5,000
 *          - else SEQ
 *
 *        WCC:
 *          - CUDA if num_edges > 100,000 (and GPU available)
 *          - else OMP if num_edges > 10,000
 *          - else SEQ
 */
#include "scheduler/scheduler.h"

#include <cstdio>

#if defined(GRAPH_ENGINE_WITH_CUDA)
#include <cuda_runtime.h>
#endif

bool is_gpu_available() {
#if defined(GRAPH_ENGINE_WITH_CUDA)
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "is_gpu_available: cudaGetDeviceCount failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return device_count > 0;
#else
    return false;
#endif
}

ExecutionMode decide(AlgorithmType algo, const WorkloadProfile& profile) {
    const bool gpu = is_gpu_available();
    ExecutionMode mode = ExecutionMode::SEQ;
    const char* reason = "default sequential path";

    switch (algo) {
    case AlgorithmType::BFS:
        if (gpu && profile.frontier_size > 10000) {
            mode = ExecutionMode::CUDA;
            reason = "BFS: frontier_size > 10000 and GPU available → CUDA";
        } else if (profile.frontier_size > 1000) {
            mode = ExecutionMode::OMP;
            reason = "BFS: frontier_size > 1000 → OpenMP";
        } else {
            mode = ExecutionMode::SEQ;
            reason = "BFS: small frontier → sequential";
        }
        if (!gpu && mode == ExecutionMode::CUDA) {
            mode = ExecutionMode::OMP;
            reason = "BFS: CUDA requested by frontier rule but no GPU → OpenMP fallback";
        }
        break;

    case AlgorithmType::PAGERANK:
        if (gpu && profile.num_nodes > 50000) {
            mode = ExecutionMode::CUDA;
            reason = "PageRank: num_nodes > 50000 and GPU available → CUDA";
        } else if (profile.num_nodes > 5000) {
            mode = ExecutionMode::OMP;
            reason = "PageRank: num_nodes > 5000 → OpenMP";
        } else {
            mode = ExecutionMode::SEQ;
            reason = "PageRank: modest graph → sequential";
        }
        if (!gpu && mode == ExecutionMode::CUDA) {
            mode = ExecutionMode::OMP;
            reason = "PageRank: CUDA threshold met but no GPU → OpenMP fallback";
        }
        break;

    case AlgorithmType::WCC:
        if (gpu && profile.num_edges > 100000) {
            mode = ExecutionMode::CUDA;
            reason = "WCC: num_edges > 100000 and GPU available → CUDA";
        } else if (profile.num_edges > 10000) {
            mode = ExecutionMode::OMP;
            reason = "WCC: num_edges > 10000 → OpenMP";
        } else {
            mode = ExecutionMode::SEQ;
            reason = "WCC: sparse graph → sequential";
        }
        if (!gpu && mode == ExecutionMode::CUDA) {
            mode = ExecutionMode::OMP;
            reason = "WCC: CUDA threshold met but no GPU → OpenMP fallback";
        }
        break;
    }

    std::printf("[scheduler] decision: %s | nodes=%u edges=%llu frontier=%u avg_deg=%.3f directed=%s\n",
                reason,
                static_cast<unsigned>(profile.num_nodes),
                static_cast<unsigned long long>(profile.num_edges),
                static_cast<unsigned>(profile.frontier_size),
                profile.avg_degree,
                profile.is_directed ? "yes" : "no");

    return mode;
}
