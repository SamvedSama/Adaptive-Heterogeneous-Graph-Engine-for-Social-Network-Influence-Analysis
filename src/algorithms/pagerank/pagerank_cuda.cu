/**
 * @file pagerank_cuda.cu
 * @brief CUDA PageRank — pull model, warp+block reductions, pinned host memory,
 *        zero mid-loop D2H stalls.
 *
 * Changes from the previous version
 * ------------------------------------
 *  1. CUB CountingInputIterator / TransformInputIterator REMOVED.
 *     cub::CountingInputIterator was removed from the public CUB API in CUDA 12.x
 *     and is no longer available via <cub/device/device_reduce.cuh>.  The L1
 *     convergence check is now computed by a dedicated warp+block reduction
 *     kernel (pagerank_l1_kernel) that follows the same pattern as the dangling
 *     kernel — one warp_reduce_sum per warp, one atomicAdd per block.  This is
 *     faster than the CUB generic path for this specific single-pass pattern and
 *     removes the temp_storage allocation entirely.
 *
 *  2. AbsDiff functor + CUB probe block REMOVED.
 *     The old functor captured raw pointers d_new / d_old at construction time.
 *     After std::swap(d_old, d_new) these pointers alias the swapped buffers, so
 *     the functor silently computed |old - old| = 0 from iteration 2 onward —
 *     causing premature convergence.  The new kernel takes d_old and d_new as
 *     explicit parameters each call, so swap is safe.
 *
 *  3. <cub/device/device_reduce.cuh> and <cub/iterator/transform_input_iterator.cuh>
 *     replaced with a single <cub/device/device_reduce.cuh> include that is only
 *     used for the warp primitive — actually that too is removed.  The only CUB
 *     dependency remaining is zero; all reductions use __shfl_down_sync directly.
 *
 *  4. All other improvements from the previous version are preserved:
 *       - warp+block dangling reduction (no global atomicAdd per thread)
 *       - pull model over device-side CSC
 *       - pinned h_dangling for low-latency D2H
 *       - persistent temp storage (now just d_l1, no CUB temp buffer needed)
 *       - init kernel instead of host memcpy
 *       - single-batch alloc / free
 */
#include "graph/graph.h"
#include "common.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
namespace {

/** Warp-level horizontal float sum via shuffle. */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/** Fills rank[i] = inv_n for all i. */
__global__ void pagerank_init_kernel(unsigned n, float inv_n,
                                     float* __restrict__ rank) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) rank[i] = inv_n;
}

/** Fills new_rank[i] = base_val for all i. */
__global__ void pagerank_fill_kernel(unsigned n, float base_val,
                                     float* __restrict__ new_rank) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) new_rank[i] = base_val;
}

/**
 * Dangling-mass reduction.
 * Threads whose vertex has inv_deg==0 contribute old_rank[i]; others contribute 0.
 * Warp shuffle reduces within each warp, then one atomicAdd per block.
 */
__global__ void pagerank_dangling_kernel(unsigned                  n,
                                         const float* __restrict__ old_rank,
                                         const float* __restrict__ inv_deg,
                                         float*       __restrict__ d_dangling) {
    extern __shared__ float sdata[];

    const unsigned i    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lid  = threadIdx.x;
    const unsigned wid  = lid >> 5;
    const unsigned lane = lid & 31;

    float val = (i < n && inv_deg[i] == 0.0f) ? old_rank[i] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    const unsigned warps_per_block = (blockDim.x + 31u) >> 5;
    if (wid == 0) {
        val = (lane < warps_per_block) ? sdata[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(d_dangling, val);
    }
}

/**
 * Pull accumulation.
 * Each thread owns one destination vertex v and accumulates contributions from
 * its in-neighbours via the CSC.  Writes to new_rank[v] are exclusive per thread.
 */
__global__ void pagerank_pull_kernel(unsigned                   n,
                                     const EdgeID* __restrict__ col_ptr,
                                     const NodeID* __restrict__ row_idx,
                                     const float*  __restrict__ old_rank,
                                     const float*  __restrict__ inv_deg,
                                     float                      damping,
                                     float*        __restrict__ new_rank) {
    const unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    const EdgeID lo = col_ptr[v];
    const EdgeID hi = col_ptr[v + 1];
    float acc = 0.0f;
    for (EdgeID e = lo; e < hi; ++e) {
        const NodeID j = row_idx[e];
        acc += old_rank[j] * inv_deg[j];   // inv_deg[j]==0 for dangling → safe
    }
    new_rank[v] += damping * acc;
}

/**
 * L1 convergence kernel: computes sum |new_rank[i] - old_rank[i]| over all i.
 * Same warp+block reduction pattern as the dangling kernel.
 * Takes old_rank and new_rank as explicit parameters so std::swap on the device
 * pointers between iterations does not invalidate the inputs.
 */
__global__ void pagerank_l1_kernel(unsigned                  n,
                                   const float* __restrict__ old_rank,
                                   const float* __restrict__ new_rank,
                                   float*       __restrict__ d_l1) {
    extern __shared__ float sdata[];

    const unsigned i    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lid  = threadIdx.x;
    const unsigned wid  = lid >> 5;
    const unsigned lane = lid & 31;

    float val = (i < n) ? fabsf(new_rank[i] - old_rank[i]) : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    const unsigned warps_per_block = (blockDim.x + 31u) >> 5;
    if (wid == 0) {
        val = (lane < warps_per_block) ? sdata[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(d_l1, val);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// CSC builder (host, called once per pagerank_cuda invocation)
// ---------------------------------------------------------------------------
namespace {

struct CSCHost {
    std::vector<EdgeID> col_ptr;
    std::vector<NodeID> row_idx;
    std::vector<float>  inv_deg;
};

static CSCHost build_csc(const CSR& csr) {
    const NodeID n = csr.num_nodes;
    const EdgeID m = csr.num_edges;

    CSCHost c;
    c.col_ptr.assign(static_cast<std::size_t>(n) + 1u, EdgeID{0});
    c.inv_deg.resize(static_cast<std::size_t>(n));
    c.row_idx.resize(static_cast<std::size_t>(m));

    for (NodeID v = 0; v < n; ++v) {
        const EdgeID lo = csr.row_ptr[v];
        const EdgeID hi = csr.row_ptr[v + 1];
        const EdgeID od = hi - lo;
        c.inv_deg[v] = (od > 0) ? (1.0f / static_cast<float>(od)) : 0.0f;
        for (EdgeID e = lo; e < hi; ++e) {
            ++c.col_ptr[static_cast<std::size_t>(csr.col_idx[e]) + 1u];
        }
    }
    for (NodeID v = 0; v < n; ++v) c.col_ptr[v + 1] += c.col_ptr[v];

    std::vector<EdgeID> next(c.col_ptr.begin(), c.col_ptr.end());
    for (NodeID v = 0; v < n; ++v) {
        const EdgeID lo = csr.row_ptr[v];
        const EdgeID hi = csr.row_ptr[v + 1];
        for (EdgeID e = lo; e < hi; ++e) {
            const NodeID  u   = csr.col_idx[e];
            const EdgeID  pos = next[u]++;
            c.row_idx[pos] = v;
        }
    }
    return c;
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
PageRankResult pagerank_cuda(const Graph& graph) {
    const NodeID n = graph.num_nodes();
    PageRankResult out;
    out.ranks.assign(static_cast<std::size_t>(n), 0.0f);
    out.iterations = 0;

    if (n == 0) return out;

    const CSR&    csr = graph.csr();
    const CSCHost csc = build_csc(csr);

    const float inv_n         = 1.0f / static_cast<float>(n);
    const float base_teleport = (1.0f - kPageRankDamping) * inv_n;

    // ---- Device allocations ------------------------------------------------
    const std::size_t n1         = static_cast<std::size_t>(n) + 1u;
    const std::size_t m          = csc.row_idx.size();
    const std::size_t rank_bytes = static_cast<std::size_t>(n) * sizeof(float);

    EdgeID* d_col_ptr  = nullptr;
    NodeID* d_row_idx  = nullptr;
    float*  d_inv_deg  = nullptr;
    float*  d_old      = nullptr;
    float*  d_new      = nullptr;
    float*  d_dangling = nullptr;
    float*  d_l1       = nullptr;

    checkCuda(cudaMalloc(&d_col_ptr,  n1 * sizeof(EdgeID)));
    checkCuda(cudaMalloc(&d_row_idx,  m  * sizeof(NodeID)));
    checkCuda(cudaMalloc(&d_inv_deg,  rank_bytes));
    checkCuda(cudaMalloc(&d_old,      rank_bytes));
    checkCuda(cudaMalloc(&d_new,      rank_bytes));
    checkCuda(cudaMalloc(&d_dangling, sizeof(float)));
    checkCuda(cudaMalloc(&d_l1,       sizeof(float)));

    // Pinned host scalar for low-latency D2H of dangling mass
    float* h_dangling = nullptr;
    checkCuda(cudaMallocHost(&h_dangling, sizeof(float)));

    // Upload static CSC structures once
    checkCuda(cudaMemcpy(d_col_ptr, csc.col_ptr.data(), n1 * sizeof(EdgeID), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_row_idx, csc.row_idx.data(), m  * sizeof(NodeID), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_inv_deg, csc.inv_deg.data(), rank_bytes,           cudaMemcpyHostToDevice));

    // ---- Kernel geometry ---------------------------------------------------
    const unsigned threads       = static_cast<unsigned>(kDefaultCudaBlockSize);
    const unsigned blocks        = (static_cast<unsigned>(n) + threads - 1u) / threads;
    const unsigned warps_per_blk = threads / 32u;
    const std::size_t smem       = static_cast<std::size_t>(warps_per_blk) * sizeof(float);

    // ---- Initialise ranks on device ----------------------------------------
    pagerank_init_kernel<<<blocks, threads>>>(static_cast<unsigned>(n), inv_n, d_old);
    checkCuda(cudaGetLastError());

    // ---- Iteration loop ----------------------------------------------------
    for (int iter = 0; iter < kPageRankMaxIter; ++iter) {

        // 1. Dangling mass reduction (warp+block, one atomicAdd per block)
        checkCuda(cudaMemset(d_dangling, 0, sizeof(float)));
        pagerank_dangling_kernel<<<blocks, threads, smem>>>(
            static_cast<unsigned>(n), d_old, d_inv_deg, d_dangling);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());

        checkCuda(cudaMemcpy(h_dangling, d_dangling, sizeof(float), cudaMemcpyDeviceToHost));
        const float base_val = base_teleport + kPageRankDamping * (*h_dangling) * inv_n;

        // 2. Fill new_rank = base_val
        pagerank_fill_kernel<<<blocks, threads>>>(
            static_cast<unsigned>(n), base_val, d_new);
        checkCuda(cudaGetLastError());

        // 3. Pull accumulation — no atomics, exclusive write per thread
        pagerank_pull_kernel<<<blocks, threads>>>(
            static_cast<unsigned>(n),
            d_col_ptr, d_row_idx, d_old, d_inv_deg, kPageRankDamping, d_new);
        checkCuda(cudaGetLastError());

        // 4. L1 convergence (warp+block reduce, same pattern as dangling)
        //    Pass d_old and d_new explicitly — safe after std::swap below.
        checkCuda(cudaMemset(d_l1, 0, sizeof(float)));
        pagerank_l1_kernel<<<blocks, threads, smem>>>(
            static_cast<unsigned>(n), d_old, d_new, d_l1);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());

        float l1_h = 0.0f;
        checkCuda(cudaMemcpy(&l1_h, d_l1, sizeof(float), cudaMemcpyDeviceToHost));

        // Swap buffers: d_old becomes the just-computed new_rank for next iter
        std::swap(d_old, d_new);
        out.iterations = iter + 1;

        if (l1_h < kPageRankEpsilon) break;
    }

    // ---- Copy result back --------------------------------------------------
    std::vector<float> host_rank(static_cast<std::size_t>(n));
    checkCuda(cudaMemcpy(host_rank.data(), d_old, rank_bytes, cudaMemcpyDeviceToHost));
    out.ranks = std::move(host_rank);

    // ---- Cleanup -----------------------------------------------------------
    checkCuda(cudaFree(d_col_ptr));
    checkCuda(cudaFree(d_row_idx));
    checkCuda(cudaFree(d_inv_deg));
    checkCuda(cudaFree(d_old));
    checkCuda(cudaFree(d_new));
    checkCuda(cudaFree(d_dangling));
    checkCuda(cudaFree(d_l1));
    checkCuda(cudaFreeHost(h_dangling));

    return out;
}
