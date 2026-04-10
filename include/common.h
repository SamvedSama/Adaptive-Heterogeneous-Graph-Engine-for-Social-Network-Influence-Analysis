/**
 * @file common.h
 * @brief Shared types, CSR layout, execution enums, and tuning constants for the
 *        adaptive heterogeneous graph engine. Also provides the CUDA error-check
 *        macro used by all device code paths.
 */
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#ifdef GRAPH_ENGINE_BUILD_CUDA
#include <cuda_runtime.h>
#endif

using NodeID = std::uint32_t;
using EdgeID = std::uint64_t;
using Weight = float;

/**
 * @brief Compressed sparse row (CSR) storage for a graph's adjacency.
 *        row_ptr has length num_nodes + 1; col_idx and values have length num_edges.
 *        For unweighted graphs, values may store 1.0f per edge or structural weights.
 */
struct CSR {
    std::vector<EdgeID> row_ptr;
    std::vector<NodeID> col_idx;
    std::vector<Weight> values;
    NodeID num_nodes = 0;
    EdgeID num_edges = 0;
};

enum class ExecutionMode { SEQ, OMP, CUDA };

enum class AlgorithmType { BFS, PAGERANK, WCC };

/** Default OpenMP thread count when not overridden by the environment. */
inline constexpr int kDefaultOmpThreads = 8;

/** Default CUDA block size for 1-D kernels. */
inline constexpr int kDefaultCudaBlockSize = 256;

/** PageRank damping factor (classic 0.85). */
inline constexpr float kPageRankDamping = 0.85f;

/** PageRank convergence threshold on L1 norm of per-iteration change. */
inline constexpr float kPageRankEpsilon = 1e-6f;

/** Maximum PageRank power iterations before stopping. */
inline constexpr int kPageRankMaxIter = 100;

/** Standard output for all BFS implementations (distances -1 = unreachable). */
struct BfsResult {
    std::vector<int> distances;
    std::vector<int> per_hop_frontier_sizes;
};

/** PageRank scores and the number of executed power iterations. */
struct PageRankResult {
    std::vector<Weight> ranks;
    int iterations = 0;
};

/** Weakly connected components: per-node label and (root, size) pairs. */
struct WccResult {
    std::vector<NodeID> component_id;
    std::vector<std::pair<NodeID, int>> component_sizes;
};

#ifdef GRAPH_ENGINE_BUILD_CUDA
/**
 * @brief Logs CUDA API failures with file/line context and terminates the process.
 */
inline void graph_engine_gpu_assert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "checkCuda: %s %s %d\n", cudaGetErrorString(code), file, line);
        std::exit(EXIT_FAILURE);
    }
}

/** Wrap CUDA driver calls; expands to a checked statement. */
#define checkCuda(ans) graph_engine_gpu_assert((ans), __FILE__, __LINE__)
#else
/** No-op stub when CUDA is disabled at compile time. */
#define checkCuda(ans) ((void)0)
#endif
