/**
 * @file pagerank_cuda.cu
 * @brief CUDA PageRank using a pull update: one thread per destination gathers
 *        contributions from in-neighbors via a reverse (CSC) structure built once.
 */
#include "graph/graph.h"
#include "common.h"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <vector>

namespace {

/**
 * @brief Accumulates rank mass from vertices with zero out-degree.
 */
__global__ void pagerank_dangling_kernel(unsigned n,
                                         const EdgeID* __restrict__ row_ptr,
                                         const float* __restrict__ old_rank,
                                         float* __restrict__ d_dangling) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const EdgeID lo = row_ptr[i];
    const EdgeID hi = row_ptr[i + 1];
    if (lo == hi) {
        atomicAdd(d_dangling, old_rank[i]);
    }
}

/**
 * @brief Resets the new rank vector to the teleport + dangling baseline.
 */
__global__ void pagerank_fill_base_kernel(unsigned n, float base_val, float* __restrict__ new_rank) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        new_rank[i] = base_val;
    }
}

/**
 * @brief Pull update: each thread processes one destination using CSC (col_ptr, row_idx).
 */
__global__ void pagerank_pull_kernel(unsigned n,
                                     const EdgeID* __restrict__ col_ptr,
                                     const NodeID* __restrict__ row_idx,
                                     const float* __restrict__ old_rank,
                                     const EdgeID* __restrict__ out_deg,
                                     float damping,
                                     float* __restrict__ new_rank) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const EdgeID lo = col_ptr[i];
    const EdgeID hi = col_ptr[i + 1];
    float sum = 0.0f;
    for (EdgeID e = lo; e < hi; ++e) {
        const NodeID j = row_idx[e];
        const EdgeID od = out_deg[j];
        if (od > 0) {
            sum += old_rank[j] / static_cast<float>(od);
        }
    }
    new_rank[i] += damping * sum;
}

struct AbsDiffZip {
    __host__ __device__ float operator()(const thrust::tuple<float, float>& t) const {
        return fabsf(thrust::get<0>(t) - thrust::get<1>(t));
    }
};

} // namespace

/**
 * @brief Builds a CSC view (column pointers + source row indices) from the host CSR.
 */
static void build_csc_from_csr(const CSR& csr,
                               std::vector<EdgeID>& col_ptr_out,
                               std::vector<NodeID>& row_idx_out,
                               std::vector<EdgeID>& out_deg_out) {
    const NodeID n = csr.num_nodes;
    const EdgeID m = csr.num_edges;
    col_ptr_out.assign(static_cast<std::size_t>(n) + 1u, 0);
    out_deg_out.assign(static_cast<std::size_t>(n), 0);

    for (NodeID v = 0; v < n; ++v) {
        const EdgeID lo = csr.row_ptr[v];
        const EdgeID hi = csr.row_ptr[v + 1];
        out_deg_out[v] = hi - lo;
        for (EdgeID e = lo; e < hi; ++e) {
            const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
            ++col_ptr_out[static_cast<std::size_t>(u) + 1u];
        }
    }

    for (NodeID v = 0; v < n; ++v) {
        col_ptr_out[static_cast<std::size_t>(v + 1)] += col_ptr_out[static_cast<std::size_t>(v)];
    }

    row_idx_out.assign(static_cast<std::size_t>(m), 0);
    std::vector<EdgeID> next = col_ptr_out;
    for (NodeID v = 0; v < n; ++v) {
        const EdgeID lo = csr.row_ptr[v];
        const EdgeID hi = csr.row_ptr[v + 1];
        for (EdgeID e = lo; e < hi; ++e) {
            const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
            const EdgeID pos = next[static_cast<std::size_t>(u)]++;
            row_idx_out[static_cast<std::size_t>(pos)] = v;
        }
    }
}

/**
 * @brief Host driver for device PageRank with Thrust-reduced L1 convergence testing.
 */
PageRankResult pagerank_cuda(const Graph& graph) {
    const NodeID n = graph.num_nodes();
    PageRankResult out;
    out.ranks.assign(static_cast<std::size_t>(n), 0.0f);
    out.iterations = 0;

    if (n == 0) {
        return out;
    }

    const CSR& csr = graph.csr();
    std::vector<EdgeID> col_ptr;
    std::vector<NodeID> row_idx;
    std::vector<EdgeID> out_deg;
    build_csc_from_csr(csr, col_ptr, row_idx, out_deg);

    const float inv_n = 1.0f / static_cast<float>(n);
    const float base_teleport = (1.0f - kPageRankDamping) * inv_n;

    EdgeID* d_row = nullptr;
    EdgeID* d_col_ptr = nullptr;
    NodeID* d_row_idx = nullptr;
    EdgeID* d_out_deg = nullptr;
    float* d_old = nullptr;
    float* d_new = nullptr;
    float* d_dangling = nullptr;

    const std::size_t row_bytes = (static_cast<std::size_t>(n) + 1u) * sizeof(EdgeID);
    const std::size_t col_bytes = col_ptr.size() * sizeof(EdgeID);
    const std::size_t ri_bytes = row_idx.size() * sizeof(NodeID);
    const std::size_t od_bytes = static_cast<std::size_t>(n) * sizeof(EdgeID);
    const std::size_t rank_bytes = static_cast<std::size_t>(n) * sizeof(float);

    checkCuda(cudaMalloc(&d_row, row_bytes));
    checkCuda(cudaMalloc(&d_col_ptr, col_bytes));
    checkCuda(cudaMalloc(&d_row_idx, ri_bytes));
    checkCuda(cudaMalloc(&d_out_deg, od_bytes));
    checkCuda(cudaMalloc(&d_old, rank_bytes));
    checkCuda(cudaMalloc(&d_new, rank_bytes));
    checkCuda(cudaMalloc(&d_dangling, sizeof(float)));

    std::vector<EdgeID> row_host(static_cast<std::size_t>(n) + 1u);
    for (std::size_t i = 0; i < row_host.size(); ++i) {
        row_host[i] = csr.row_ptr[i];
    }

    checkCuda(cudaMemcpy(d_row, row_host.data(), row_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_ptr, col_ptr.data(), col_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_row_idx, row_idx.data(), ri_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_out_deg, out_deg.data(), od_bytes, cudaMemcpyHostToDevice));

    std::vector<float> host_init(static_cast<std::size_t>(n), inv_n);
    checkCuda(cudaMemcpy(d_old, host_init.data(), rank_bytes, cudaMemcpyHostToDevice));

    const int threads = kDefaultCudaBlockSize;
    const int blocks = static_cast<int>((static_cast<unsigned>(n) + threads - 1) / threads);

    for (int iter = 0; iter < kPageRankMaxIter; ++iter) {
        checkCuda(cudaMemset(d_dangling, 0, sizeof(float)));
        pagerank_dangling_kernel<<<blocks, threads>>>(n, d_row, d_old, d_dangling);
        checkCuda(cudaGetLastError());

        float dangling_h = 0.0f;
        checkCuda(cudaMemcpy(&dangling_h, d_dangling, sizeof(float), cudaMemcpyDeviceToHost));
        const float base = base_teleport + kPageRankDamping * dangling_h * inv_n;

        pagerank_fill_base_kernel<<<blocks, threads>>>(n, base, d_new);
        checkCuda(cudaGetLastError());

        pagerank_pull_kernel<<<blocks, threads>>>(
            n, d_col_ptr, d_row_idx, d_old, d_out_deg, kPageRankDamping, d_new);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());

        thrust::device_ptr<float> p_old(d_old);
        thrust::device_ptr<float> p_new(d_new);
        const float l1 = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(p_new, p_old)),
            thrust::make_zip_iterator(thrust::make_tuple(p_new + static_cast<std::ptrdiff_t>(n),
                                                         p_old + static_cast<std::ptrdiff_t>(n))),
            AbsDiffZip(),
            0.0f,
            thrust::plus<float>());

        std::swap(d_old, d_new);
        out.iterations = iter + 1;

        if (l1 < kPageRankEpsilon) {
            break;
        }
    }

    std::vector<float> host_rank(static_cast<std::size_t>(n));
    checkCuda(cudaMemcpy(host_rank.data(), d_old, rank_bytes, cudaMemcpyDeviceToHost));
    out.ranks.assign(host_rank.begin(), host_rank.end());

    checkCuda(cudaFree(d_row));
    checkCuda(cudaFree(d_col_ptr));
    checkCuda(cudaFree(d_row_idx));
    checkCuda(cudaFree(d_out_deg));
    checkCuda(cudaFree(d_old));
    checkCuda(cudaFree(d_new));
    checkCuda(cudaFree(d_dangling));

    return out;
}
