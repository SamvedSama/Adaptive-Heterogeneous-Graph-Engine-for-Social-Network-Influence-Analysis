/**
 * @file bfs_cuda.cu
 * @brief Multi-GPU level-synchronous BFS for 2× Tesla T4 (sm_75, no NVLink).
 *
 * Strategy: vertex-partition model.
 *   GPU 0 owns vertices [0,   mid)   mid = n/2
 *   GPU 1 owns vertices [mid, n)
 *
 * Each GPU holds:
 *   - A FULL n+1 row_ptr (zero-length entries outside its partition)
 *   - Only its OWN edge slice in col_idx
 *   - A FULL dist[] array (global node IDs work without remapping)
 *   - A FULL frontier buffer (cross-partition vertices injected here)
 *
 * After every level the host merges both dist[] arrays and injects
 * cross-partition discoveries into the other GPU's next frontier.
 *
 * Falls back to single-GPU (GPU 0) when only 1 device present or
 * n < 2 * kMinPartitionSize.
 */
#include "graph/graph.h"
#include "common.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace
{
    constexpr int kBlock = 256;
    constexpr NodeID kMinPartitionSize = 10000;
} // namespace

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
namespace
{

    __global__ void bfs_expand_kernel(
        int level,
        const EdgeID *__restrict__ row_ptr,
        const NodeID *__restrict__ col_idx,
        int *__restrict__ dist,
        const NodeID *__restrict__ curr_frontier,
        int curr_size,
        NodeID *__restrict__ next_frontier,
        int *__restrict__ next_count)
    {
        const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (tid >= static_cast<int64_t>(curr_size))
            return;

        const NodeID v = curr_frontier[static_cast<int>(tid)];
        const EdgeID lo = row_ptr[v];
        const EdgeID hi = row_ptr[v + 1];

        for (EdgeID e = lo; e < hi; ++e)
        {
            const NodeID u = col_idx[e];
            const int old = atomicCAS(&dist[u], -1, level + 1);
            if (old == -1)
            {
                const int pos = atomicAdd(next_count, 1);
                next_frontier[pos] = u;
            }
        }
    }

} // namespace

// ---------------------------------------------------------------------------
// Per-GPU state
// ---------------------------------------------------------------------------
namespace
{

    struct GpuBfsState
    {
        EdgeID *d_row = nullptr;
        NodeID *d_col = nullptr;
        int *d_dist = nullptr;
        NodeID *d_front_a = nullptr;
        NodeID *d_front_b = nullptr;
        int *d_next_count = nullptr;

        void free_all()
        {
            cudaFree(d_row);
            d_row = nullptr;
            cudaFree(d_col);
            d_col = nullptr;
            cudaFree(d_dist);
            d_dist = nullptr;
            cudaFree(d_front_a);
            d_front_a = nullptr;
            cudaFree(d_front_b);
            d_front_b = nullptr;
            cudaFree(d_next_count);
            d_next_count = nullptr;
        }
    };

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
BfsResult bfs_cuda(const Graph &graph, NodeID source)
{
    const NodeID n = graph.num_nodes();
    BfsResult out;
    out.distances.assign(static_cast<std::size_t>(n), -1);
    out.per_hop_frontier_sizes.clear();

    if (n == 0 || source >= n)
        return out;

    int num_gpus = 0;
    checkCuda(cudaGetDeviceCount(&num_gpus));
    const bool use_multi = (num_gpus >= 2) && (n >= 2 * kMinPartitionSize);

    const CSR &csr = graph.csr();

    // Host dist initialised here — used by BOTH paths
    std::vector<int> host_dist(static_cast<std::size_t>(n), -1);
    host_dist[source] = 0;

    const std::size_t dist_bytes = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t frontier_bytes = static_cast<std::size_t>(n) * sizeof(NodeID);
    const std::size_t row_bytes_full = (static_cast<std::size_t>(n) + 1u) * sizeof(EdgeID);

    // ════════════════════════════════════════════════════════════════════════
    // SINGLE-GPU PATH
    // ════════════════════════════════════════════════════════════════════════
    if (!use_multi)
    {
        checkCuda(cudaSetDevice(0));

        const std::size_t col_bytes = static_cast<std::size_t>(csr.num_edges) * sizeof(NodeID);

        EdgeID *d_row = nullptr;
        NodeID *d_col = nullptr;
        int *d_dist = nullptr;
        NodeID *d_front_a = nullptr;
        NodeID *d_front_b = nullptr;
        int *d_next_count = nullptr;

        checkCuda(cudaMalloc(&d_row, row_bytes_full));
        checkCuda(cudaMalloc(&d_col, col_bytes));
        checkCuda(cudaMalloc(&d_dist, dist_bytes));
        checkCuda(cudaMalloc(&d_front_a, frontier_bytes));
        checkCuda(cudaMalloc(&d_front_b, frontier_bytes));
        checkCuda(cudaMalloc(&d_next_count, sizeof(int)));

        checkCuda(cudaMemcpy(d_row, csr.row_ptr.data(), row_bytes_full, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_col, csr.col_idx.data(), col_bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_dist, host_dist.data(), dist_bytes, cudaMemcpyHostToDevice));

        NodeID *d_curr = d_front_a;
        NodeID *d_next = d_front_b;
        int curr_size = 1;
        checkCuda(cudaMemcpy(d_curr, &source, sizeof(NodeID), cudaMemcpyHostToDevice));
        out.per_hop_frontier_sizes.push_back(1);

        int layer = 0;
        while (curr_size > 0)
        {
            checkCuda(cudaMemset(d_next_count, 0, sizeof(int)));
            const int blocks = (curr_size + kBlock - 1) / kBlock;
            bfs_expand_kernel<<<blocks, kBlock>>>(
                layer, d_row, d_col, d_dist, d_curr, curr_size, d_next, d_next_count);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            int next_size = 0;
            checkCuda(cudaMemcpy(&next_size, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));
            if (next_size == 0)
                break;

            out.per_hop_frontier_sizes.push_back(next_size);
            std::swap(d_curr, d_next);
            curr_size = next_size;
            ++layer;
        }

        checkCuda(cudaMemcpy(host_dist.data(), d_dist, dist_bytes, cudaMemcpyDeviceToHost));
        out.distances = std::move(host_dist);

        cudaFree(d_row);
        cudaFree(d_col);
        cudaFree(d_dist);
        cudaFree(d_front_a);
        cudaFree(d_front_b);
        cudaFree(d_next_count);
        return out;
    }

    // ════════════════════════════════════════════════════════════════════════
    // MULTI-GPU PATH
    // ════════════════════════════════════════════════════════════════════════
    const NodeID mid = n / 2;
    const NodeID part_start[2] = {0, mid};
    const NodeID part_end[2] = {mid, n};

    GpuBfsState gpu[2];

    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));

        // Build full n+1 row_ptr: entries inside the partition are rebased so
        // col_idx[0] corresponds to edge edge_lo of the global CSR.
        // Entries outside the partition remain 0 (hi-lo == 0, no edges expanded).
        const EdgeID edge_lo = csr.row_ptr[part_start[g]];
        const EdgeID edge_hi = csr.row_ptr[part_end[g]];
        const std::size_t local_cols = static_cast<std::size_t>(edge_hi - edge_lo);

        std::vector<EdgeID> full_row(static_cast<std::size_t>(n) + 1u, EdgeID{0});
        for (NodeID v = part_start[g]; v <= part_end[g]; ++v)
        {
            full_row[v] = csr.row_ptr[v] - edge_lo;
        }

        checkCuda(cudaMalloc(&gpu[g].d_row, row_bytes_full));
        checkCuda(cudaMalloc(&gpu[g].d_col, local_cols * sizeof(NodeID)));
        checkCuda(cudaMalloc(&gpu[g].d_dist, dist_bytes));
        checkCuda(cudaMalloc(&gpu[g].d_front_a, frontier_bytes));
        checkCuda(cudaMalloc(&gpu[g].d_front_b, frontier_bytes));
        checkCuda(cudaMalloc(&gpu[g].d_next_count, sizeof(int)));

        checkCuda(cudaMemcpy(gpu[g].d_row,
                             full_row.data(), row_bytes_full, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(gpu[g].d_col,
                             csr.col_idx.data() + static_cast<std::size_t>(edge_lo),
                             local_cols * sizeof(NodeID), cudaMemcpyHostToDevice));
        // Upload initialised dist (source = 0) — same on both GPUs
        checkCuda(cudaMemcpy(gpu[g].d_dist,
                             host_dist.data(), dist_bytes, cudaMemcpyHostToDevice));
    }

    // Seed source vertex on the GPU that owns it
    const int source_gpu = (source < mid) ? 0 : 1;
    checkCuda(cudaSetDevice(source_gpu));
    checkCuda(cudaMemcpy(gpu[source_gpu].d_front_a,
                         &source, sizeof(NodeID), cudaMemcpyHostToDevice));

    // Pinned readback buffers
    NodeID *h_next_frontier[2] = {nullptr, nullptr};
    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaMallocHost(&h_next_frontier[g], frontier_bytes));
    }

    int curr_sizes[2];
    curr_sizes[source_gpu] = 1;
    curr_sizes[1 - source_gpu] = 0;

    NodeID *d_curr[2] = {gpu[0].d_front_a, gpu[1].d_front_a};
    NodeID *d_next[2] = {gpu[0].d_front_b, gpu[1].d_front_b};

    out.per_hop_frontier_sizes.push_back(1);
    int layer = 0;

    while (curr_sizes[0] > 0 || curr_sizes[1] > 0)
    {

        // ── Launch expand on both GPUs ─────────────────────────────────────
        for (int g = 0; g < 2; ++g)
        {
            if (curr_sizes[g] == 0)
                continue;
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaMemset(gpu[g].d_next_count, 0, sizeof(int)));
            const int blocks = (curr_sizes[g] + kBlock - 1) / kBlock;
            bfs_expand_kernel<<<blocks, kBlock>>>(
                layer,
                gpu[g].d_row, gpu[g].d_col,
                gpu[g].d_dist,
                d_curr[g], curr_sizes[g],
                d_next[g], gpu[g].d_next_count);
            checkCuda(cudaGetLastError());
        }

        // ── Sync and read next_count ───────────────────────────────────────
        int next_counts[2] = {0, 0};
        for (int g = 0; g < 2; ++g)
        {
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaMemcpy(&next_counts[g], gpu[g].d_next_count,
                                 sizeof(int), cudaMemcpyDeviceToHost));
        }

        if (next_counts[0] == 0 && next_counts[1] == 0)
            break;

        // ── Read back next-frontiers ───────────────────────────────────────
        for (int g = 0; g < 2; ++g)
        {
            if (next_counts[g] == 0)
                continue;
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaMemcpy(h_next_frontier[g], d_next[g],
                                 static_cast<std::size_t>(next_counts[g]) * sizeof(NodeID),
                                 cudaMemcpyDeviceToHost));
        }

        // ── Cross-partition injection ──────────────────────────────────────
        for (int g = 0; g < 2; ++g)
        {
            if (next_counts[g] == 0)
                continue;
            const int other = 1 - g;
            std::vector<NodeID> cross;
            for (int i = 0; i < next_counts[g]; ++i)
            {
                const NodeID v = h_next_frontier[g][i];
                if (v >= part_start[other] && v < part_end[other])
                {
                    cross.push_back(v);
                }
            }
            if (!cross.empty())
            {
                checkCuda(cudaSetDevice(other));
                checkCuda(cudaMemcpy(
                    d_next[other] + next_counts[other],
                    cross.data(),
                    cross.size() * sizeof(NodeID),
                    cudaMemcpyHostToDevice));
                next_counts[other] += static_cast<int>(cross.size());
            }
        }

        // ── Merge dist arrays ─────────────────────────────────────────────
        {
            std::vector<int> dist0(n), dist1(n);
            checkCuda(cudaSetDevice(0));
            checkCuda(cudaMemcpy(dist0.data(), gpu[0].d_dist, dist_bytes, cudaMemcpyDeviceToHost));
            checkCuda(cudaSetDevice(1));
            checkCuda(cudaMemcpy(dist1.data(), gpu[1].d_dist, dist_bytes, cudaMemcpyDeviceToHost));

            for (NodeID v = 0; v < n; ++v)
            {
                if (host_dist[v] == -1 && dist0[v] != -1)
                    host_dist[v] = dist0[v];
                if (host_dist[v] == -1 && dist1[v] != -1)
                    host_dist[v] = dist1[v];
            }
            for (int g = 0; g < 2; ++g)
            {
                checkCuda(cudaSetDevice(g));
                checkCuda(cudaMemcpy(gpu[g].d_dist, host_dist.data(),
                                     dist_bytes, cudaMemcpyHostToDevice));
            }
        }

        // ── Swap frontier buffers ──────────────────────────────────────────
        out.per_hop_frontier_sizes.push_back(next_counts[0] + next_counts[1]);
        for (int g = 0; g < 2; ++g)
        {
            std::swap(d_curr[g], d_next[g]);
            curr_sizes[g] = next_counts[g];
        }
        ++layer;
    }

    out.distances = std::move(host_dist);

    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));
        gpu[g].free_all();
        cudaFreeHost(h_next_frontier[g]);
    }
    return out;
}
