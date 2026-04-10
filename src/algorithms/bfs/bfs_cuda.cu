/**
 * @file bfs_cuda.cu
 * @brief Level-synchronous BFS on the GPU: one thread per frontier vertex expands
 *        outgoing edges with atomicCAS on distances and atomic append to the next frontier.
 */
#include "graph/graph.h"
#include "common.h"

#include <cuda_runtime.h>
#include <vector>

namespace {

/**
 * @brief Expands one BFS level: threads map to current frontier entries.
 */
__global__ void bfs_expand_kernel(int level,
                                  unsigned num_nodes,
                                  const EdgeID* __restrict__ row_ptr,
                                  const NodeID* __restrict__ col_idx,
                                  int* __restrict__ dist,
                                  const NodeID* __restrict__ curr_frontier,
                                  int curr_size,
                                  NodeID* __restrict__ next_frontier,
                                  int* __restrict__ next_count) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= curr_size) {
        return;
    }

    const NodeID v = curr_frontier[tid];
    const EdgeID lo = row_ptr[v];
    const EdgeID hi = row_ptr[v + 1];
    for (EdgeID e = lo; e < hi; ++e) {
        const NodeID u = col_idx[e];
        const int old = atomicCAS(&dist[u], -1, level + 1);
        if (old == -1) {
            const int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = u;
        }
    }
}

} // namespace

/**
 * @brief Host orchestration: copies CSR to the device and alternates frontiers until empty.
 */
BfsResult bfs_cuda(const Graph& graph, NodeID source) {
    const NodeID n = graph.num_nodes();
    BfsResult out;
    out.distances.assign(static_cast<std::size_t>(n), -1);
    out.per_hop_frontier_sizes.clear();

    if (n == 0 || source >= n) {
        return out;
    }

    const CSR& csr = graph.csr();

    std::vector<EdgeID> row_host(static_cast<std::size_t>(n) + 1u);
    for (std::size_t i = 0; i < row_host.size(); ++i) {
        row_host[i] = csr.row_ptr[i];
    }
    std::vector<NodeID> col_host(static_cast<std::size_t>(csr.num_edges));
    for (EdgeID e = 0; e < csr.num_edges; ++e) {
        col_host[static_cast<std::size_t>(e)] = csr.col_idx[static_cast<std::size_t>(e)];
    }

    EdgeID* d_row = nullptr;
    NodeID* d_col = nullptr;
    int* d_dist = nullptr;
    NodeID* d_front_a = nullptr;
    NodeID* d_front_b = nullptr;
    int* d_next_count = nullptr;

    const std::size_t row_bytes = row_host.size() * sizeof(EdgeID);
    const std::size_t col_bytes = col_host.size() * sizeof(NodeID);
    const std::size_t dist_bytes = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t frontier_cap = static_cast<std::size_t>(n) * sizeof(NodeID);

    checkCuda(cudaMalloc(&d_row, row_bytes));
    checkCuda(cudaMalloc(&d_col, col_bytes));
    checkCuda(cudaMalloc(&d_dist, dist_bytes));
    checkCuda(cudaMalloc(&d_front_a, frontier_cap));
    checkCuda(cudaMalloc(&d_front_b, frontier_cap));
    checkCuda(cudaMalloc(&d_next_count, sizeof(int)));

    checkCuda(cudaMemcpy(d_row, row_host.data(), row_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col, col_host.data(), col_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_dist, 0xFF, dist_bytes));

    std::vector<int> host_dist(static_cast<std::size_t>(n), -1);
    host_dist[source] = 0;
    checkCuda(cudaMemcpy(d_dist, host_dist.data(), dist_bytes, cudaMemcpyHostToDevice));

    NodeID* d_curr = d_front_a;
    NodeID* d_next = d_front_b;

    int curr_size = 1;
    checkCuda(cudaMemcpy(d_curr, &source, sizeof(NodeID), cudaMemcpyHostToDevice));
    out.per_hop_frontier_sizes.push_back(1);

    const int threads = kDefaultCudaBlockSize;
    /** Current BFS layer index; neighbors discovered from this layer receive dist = layer + 1. */
    int layer = 0;

    while (curr_size > 0) {
        int zero = 0;
        checkCuda(cudaMemcpy(d_next_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

        const int blocks = (curr_size + threads - 1) / threads;
        bfs_expand_kernel<<<blocks, threads>>>(layer,
                                                 static_cast<unsigned>(n),
                                                 d_row,
                                                 d_col,
                                                 d_dist,
                                                 d_curr,
                                                 curr_size,
                                                 d_next,
                                                 d_next_count);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());

        int next_size = 0;
        checkCuda(cudaMemcpy(&next_size, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (next_size == 0) {
            break;
        }

        out.per_hop_frontier_sizes.push_back(next_size);
        std::swap(d_curr, d_next);
        curr_size = next_size;
        ++layer;
    }

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
