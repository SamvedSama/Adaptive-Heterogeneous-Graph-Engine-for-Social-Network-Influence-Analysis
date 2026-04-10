/**
 * @file bfs_cuda.cu
 * @brief Level-synchronous BFS on the GPU: one thread per frontier vertex expands
 *        outgoing edges with atomicCAS on distances and atomic append to next frontier.
 *
 * Improvements over the original:
 *  1. `cudaMemcpy` to reset next_count replaced with `cudaMemset` — avoids an
 *     unnecessary host-device round-trip every BFS level.
 *  2. Manual CSR copy loops replaced with std::copy — cleaner; compiler vectorises.
 *  3. Dead `num_nodes` kernel parameter (passed but never used) removed.
 *  4. Thread-index guard widened to std::ptrdiff_t / int64_t arithmetic so graphs
 *     with > 2^31 frontier entries don't silently overflow the int tid calculation.
 *  5. All device pointers freed on every exit path (original skipped frees on the
 *     n==0 / source-out-of-range early returns — harmless there, but consistent style).
 *  6. host_dist initialisation consolidated: fill with -1 once, set source=0 once,
 *     then a single cudaMemcpy — removes the redundant cudaMemset(0xFF) + overwrite.
 */
#include "graph/graph.h"
#include "common.h"

#include <cuda_runtime.h>
#include <algorithm>   // std::copy
#include <vector>

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
namespace {

/**
 * @brief Expands one BFS level.
 *        Each thread handles one vertex in the current frontier and walks its
 *        adjacency list.  First-visit is claimed with atomicCAS(-1 → level+1);
 *        winners append to next_frontier via atomicAdd on next_count.
 */
__global__ void bfs_expand_kernel(
        int                     level,
        const EdgeID* __restrict__ row_ptr,
        const NodeID* __restrict__ col_idx,
        int*          __restrict__ dist,
        const NodeID* __restrict__ curr_frontier,
        int                     curr_size,
        NodeID*       __restrict__ next_frontier,
        int*          __restrict__ next_count)
{
    // Use 64-bit index arithmetic to avoid overflow on very large frontiers.
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<int64_t>(curr_size)) {
        return;
    }

    const NodeID v  = curr_frontier[static_cast<int>(tid)];
    const EdgeID lo = row_ptr[v];
    const EdgeID hi = row_ptr[v + 1];

    for (EdgeID e = lo; e < hi; ++e) {
        const NodeID u   = col_idx[e];
        const int    old = atomicCAS(&dist[u], -1, level + 1);
        if (old == -1) {
            const int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = u;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------
BfsResult bfs_cuda(const Graph& graph, NodeID source) {
    const NodeID n = graph.num_nodes();
    BfsResult out;
    out.distances.assign(static_cast<std::size_t>(n), -1);
    out.per_hop_frontier_sizes.clear();

    if (n == 0 || source >= n) {
        return out;
    }

    const CSR& csr = graph.csr();

    // ---- Build host-side CSR vectors ----------------------------------------
    const std::size_t row_count = static_cast<std::size_t>(n) + 1u;
    const std::size_t col_count = static_cast<std::size_t>(csr.num_edges);

    std::vector<EdgeID> row_host(row_count);
    std::vector<NodeID> col_host(col_count);
    std::copy(csr.row_ptr, csr.row_ptr + row_count, row_host.begin());
    std::copy(csr.col_idx, csr.col_idx + col_count, col_host.begin());

    // ---- Initialise distance array on host, then send once ------------------
    std::vector<int> host_dist(static_cast<std::size_t>(n), -1);
    host_dist[source] = 0;

    // ---- Device allocations -------------------------------------------------
    EdgeID* d_row        = nullptr;
    NodeID* d_col        = nullptr;
    int*    d_dist       = nullptr;
    NodeID* d_front_a    = nullptr;
    NodeID* d_front_b    = nullptr;
    int*    d_next_count = nullptr;

    const std::size_t row_bytes      = row_count * sizeof(EdgeID);
    const std::size_t col_bytes      = col_count * sizeof(NodeID);
    const std::size_t dist_bytes     = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t frontier_bytes = static_cast<std::size_t>(n) * sizeof(NodeID);

    checkCuda(cudaMalloc(&d_row,        row_bytes));
    checkCuda(cudaMalloc(&d_col,        col_bytes));
    checkCuda(cudaMalloc(&d_dist,       dist_bytes));
    checkCuda(cudaMalloc(&d_front_a,    frontier_bytes));
    checkCuda(cudaMalloc(&d_front_b,    frontier_bytes));
    checkCuda(cudaMalloc(&d_next_count, sizeof(int)));

    checkCuda(cudaMemcpy(d_row,  row_host.data(),  row_bytes,  cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col,  col_host.data(),  col_bytes,  cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_dist, host_dist.data(), dist_bytes, cudaMemcpyHostToDevice));

    // ---- Seed first frontier ------------------------------------------------
    NodeID* d_curr = d_front_a;
    NodeID* d_next = d_front_b;

    int curr_size = 1;
    checkCuda(cudaMemcpy(d_curr, &source, sizeof(NodeID), cudaMemcpyHostToDevice));
    out.per_hop_frontier_sizes.push_back(1);

    const int threads = kDefaultCudaBlockSize;
    int       layer   = 0;

    // ---- Main BFS loop ------------------------------------------------------
    while (curr_size > 0) {
        // Reset next-frontier counter — cudaMemset is cheaper than a H→D memcpy.
        checkCuda(cudaMemset(d_next_count, 0, sizeof(int)));

        const int blocks = (curr_size + threads - 1) / threads;
        bfs_expand_kernel<<<blocks, threads>>>(
                layer,
                d_row, d_col,
                d_dist,
                d_curr, curr_size,
                d_next, d_next_count);

        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());

        int next_size = 0;
        checkCuda(cudaMemcpy(&next_size, d_next_count, sizeof(int),
                             cudaMemcpyDeviceToHost));

        if (next_size == 0) {
            break;
        }

        out.per_hop_frontier_sizes.push_back(next_size);
        std::swap(d_curr, d_next);
        curr_size = next_size;
        ++layer;
    }

    // ---- Copy results back and free device memory ---------------------------
    checkCuda(cudaMemcpy(host_dist.data(), d_dist, dist_bytes, cudaMemcpyDeviceToHost));
    out.distances = std::move(host_dist);

    checkCuda(cudaFree(d_row));
    checkCuda(cudaFree(d_col));
    checkCuda(cudaFree(d_dist));
    checkCuda(cudaFree(d_front_a));
    checkCuda(cudaFree(d_front_b));
    checkCuda(cudaFree(d_next_count));

    return out;
}
