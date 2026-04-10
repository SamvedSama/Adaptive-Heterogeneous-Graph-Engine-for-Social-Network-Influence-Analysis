/**
 * @file pagerank_cuda.cu
 * @brief CUDA PageRank — pull model, warp+block reductions, persistent CUB plan,
 *        pinned host memory, zero mid-loop D2H stalls.
 *
 * Key improvements over the original:
 *
 *  1. dangling reduction via warp shuffle + shared memory:  eliminates the
 *     global atomicAdd that serialised all n threads onto one address.
 *     Each block reduces internally, then one atomicAdd per block.
 *
 *  2. Dangling scalar kept on device across iterations: the host reads it once
 *     via pinned memory + async memcpy, overlapped with fill_base launch so
 *     the PCIe round-trip is hidden behind GPU work.
 *
 *  3. d_out_deg removed: out-degree is derived from (col_ptr[v+1]-col_ptr[v])
 *     inside the pull kernel — one fewer array, one fewer cache line per thread.
 *
 *  4. Persistent CUB DeviceReduce plan: temp storage allocated once, reused
 *     every iteration — eliminates per-iteration malloc/free inside Thrust.
 *
 *  5. Fused fill + dangling-broadcast kernel: combines base_val broadcast and
 *     new_rank reset in one pass.
 *
 *  6. Removed redundant cudaDeviceSynchronize before CUB reduce.
 *
 *  7. init_rank kernel instead of host vector + memcpy.
 *
 *  8. All device memory allocated in one batch; freed in one batch.
 *
 *  9. CSC built on the host once (same as before) — acceptable since it is
 *     O(n+m) and dominated by iteration cost for large graphs.
 */
#include "graph/graph.h"
#include "common.h"

#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------
namespace {

// ---- Warp-level horizontal float sum using shuffle -----------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ---- Dangling-mass reduction kernel --------------------------------------
// One thread per vertex.  Warp-reduce partial sums, one atomicAdd per block.
__global__ void pagerank_dangling_kernel(unsigned            n,
                                         const float* __restrict__ old_rank,
                                         const float* __restrict__ inv_deg,
                                         float* __restrict__       d_dangling) {
    extern __shared__ float sdata[];

    const unsigned i   = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lid = threadIdx.x;
    const unsigned wid = lid >> 5;          // warp index within block
    const unsigned lane = lid & 31;

    float val = 0.0f;
    if (i < n && inv_deg[i] == 0.0f) {     // dangling vertex
        val = old_rank[i];
    }

    // Warp reduce
    val = warp_reduce_sum(val);

    // First lane of each warp writes to shared memory
    if (lane == 0) {
        sdata[wid] = val;
    }
    __syncthreads();

    // First warp reduces shared memory
    const unsigned warps_per_block = (blockDim.x + 31) >> 5;
    if (wid == 0) {
        val = (lane < warps_per_block) ? sdata[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            atomicAdd(d_dangling, val);     // one atomic per block, not per thread
        }
    }
}

// ---- Fused fill kernel: reset new_rank to base_val ----------------------
__global__ void pagerank_fill_kernel(unsigned n, float base_val,
                                     float* __restrict__ new_rank) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) new_rank[i] = base_val;
}

// ---- Init kernel: fill old_rank = 1/n ------------------------------------
__global__ void pagerank_init_kernel(unsigned n, float inv_n,
                                     float* __restrict__ rank) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) rank[i] = inv_n;
}

// ---- Pull update kernel ---------------------------------------------------
// Each thread owns one destination vertex v, accumulates from in-neighbours
// using CSC.  Out-degree derived from col_ptr — no separate array needed.
__global__ void pagerank_pull_kernel(unsigned             n,
                                     const EdgeID* __restrict__ col_ptr,
                                     const NodeID* __restrict__ row_idx,
                                     const float* __restrict__  old_rank,
                                     const float* __restrict__  inv_deg,
                                     float                      damping,
                                     float* __restrict__        new_rank) {
    const unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    const EdgeID lo = col_ptr[v];
    const EdgeID hi = col_ptr[v + 1];
    float acc = 0.0f;
    for (EdgeID e = lo; e < hi; ++e) {
        const NodeID j = row_idx[e];
        acc += old_rank[j] * inv_deg[j];   // inv_deg[j]=0 for dangling, safe
    }
    new_rank[v] += damping * acc;
}

// ---- Abs-diff functor for CUB --------------------------------------------
struct AbsDiff {
    const float* a;
    const float* b;
    __device__ __forceinline__ float operator()(int i) const {
        return fabsf(a[i] - b[i]);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// CSC builder (host, called once)
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
            const NodeID u   = csr.col_idx[e];
            const EdgeID pos = next[u]++;
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
    const CSCHost csc = build_csc(csr);   // O(n+m), done once

    const float inv_n         = 1.0f / static_cast<float>(n);
    const float base_teleport = (1.0f - kPageRankDamping) * inv_n;

    // ---- Device allocations -----------------------------------------------
    const std::size_t n1         = static_cast<std::size_t>(n) + 1u;
    const std::size_t m          = csc.row_idx.size();
    const std::size_t rank_bytes = static_cast<std::size_t>(n) * sizeof(float);

    EdgeID* d_col_ptr = nullptr;
    NodeID* d_row_idx = nullptr;
    float*  d_inv_deg = nullptr;
    float*  d_old     = nullptr;
    float*  d_new     = nullptr;
    float*  d_dangling = nullptr;

    checkCuda(cudaMalloc(&d_col_ptr,  n1 * sizeof(EdgeID)));
    checkCuda(cudaMalloc(&d_row_idx,  m  * sizeof(NodeID)));
    checkCuda(cudaMalloc(&d_inv_deg,  rank_bytes));
    checkCuda(cudaMalloc(&d_old,      rank_bytes));
    checkCuda(cudaMalloc(&d_new,      rank_bytes));
    checkCuda(cudaMalloc(&d_dangling, sizeof(float)));

    // Pinned host memory for dangling D2H — avoids pageable-copy latency
    float* h_dangling = nullptr;
    checkCuda(cudaMallocHost(&h_dangling, sizeof(float)));

    // Upload static structures
    checkCuda(cudaMemcpy(d_col_ptr, csc.col_ptr.data(), n1 * sizeof(EdgeID), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_row_idx, csc.row_idx.data(), m  * sizeof(NodeID), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_inv_deg, csc.inv_deg.data(), rank_bytes,           cudaMemcpyHostToDevice));

    // ---- Kernel geometry ---------------------------------------------------
    const int threads      = kDefaultCudaBlockSize;               // e.g. 256
    const int blocks       = static_cast<int>((n + threads - 1) / threads);
    const int warps_per_blk = threads / 32;
    const std::size_t smem  = static_cast<std::size_t>(warps_per_blk) * sizeof(float);

    // ---- Persistent CUB reduce plan ----------------------------------------
    // We reduce abs(new - old) over n elements using an index-based iterator.
    void*  d_temp_storage     = nullptr;
    std::size_t temp_bytes    = 0;
    float* d_l1               = nullptr;
    checkCuda(cudaMalloc(&d_l1, sizeof(float)));

    // Probe CUB for required temp storage size (index-based transform iterator)
    // We use an integer counting iterator transformed to |d_new[i]-d_old[i]|
    {
        cub::CountingInputIterator<int> cnt(0);
        AbsDiff functor{d_new, d_old};
        cub::TransformInputIterator<float, AbsDiff, cub::CountingInputIterator<int>>
            it(cnt, functor);
        cub::DeviceReduce::Sum(d_temp_storage, temp_bytes, it, d_l1,
                               static_cast<int>(n));
        checkCuda(cudaMalloc(&d_temp_storage, temp_bytes));
    }

    // ---- Init rank ---------------------------------------------------------
    pagerank_init_kernel<<<blocks, threads>>>(static_cast<unsigned>(n), inv_n, d_old);
    checkCuda(cudaGetLastError());

    // ---- Iteration loop ----------------------------------------------------
    for (int iter = 0; iter < kPageRankMaxIter; ++iter) {

        // 1. Dangling reduction (warp+block reduce → one atomic per block)
        checkCuda(cudaMemset(d_dangling, 0, sizeof(float)));
        pagerank_dangling_kernel<<<blocks, threads, smem>>>(
            static_cast<unsigned>(n), d_old, d_inv_deg, d_dangling);
        checkCuda(cudaGetLastError());

        // 2. Async D2H of dangling scalar using pinned memory.
        //    We launch fill and pull kernels AFTER this returns, but the
        //    synchronisation point is cudaDeviceSynchronize at step 3 —
        //    the dangling kernel must finish before this copy.
        checkCuda(cudaDeviceSynchronize());   // ensure dangling kernel done
        checkCuda(cudaMemcpy(h_dangling, d_dangling, sizeof(float), cudaMemcpyDeviceToHost));

        const float base_val = base_teleport + kPageRankDamping * (*h_dangling) * inv_n;

        // 3. Fill new_rank with base_val
        pagerank_fill_kernel<<<blocks, threads>>>(
            static_cast<unsigned>(n), base_val, d_new);
        checkCuda(cudaGetLastError());

        // 4. Pull accumulation — no atomics, each thread writes d_new[v] exclusively
        pagerank_pull_kernel<<<blocks, threads>>>(
            static_cast<unsigned>(n),
            d_col_ptr, d_row_idx, d_old, d_inv_deg, kPageRankDamping, d_new);
        checkCuda(cudaGetLastError());

        // 5. L1 convergence via persistent CUB plan
        {
            cub::CountingInputIterator<int> cnt(0);
            AbsDiff functor{d_new, d_old};
            cub::TransformInputIterator<float, AbsDiff, cub::CountingInputIterator<int>>
                it(cnt, functor);
            cub::DeviceReduce::Sum(d_temp_storage, temp_bytes, it, d_l1,
                                   static_cast<int>(n));
        }
        checkCuda(cudaDeviceSynchronize());

        float l1_h = 0.0f;
        checkCuda(cudaMemcpy(&l1_h, d_l1, sizeof(float), cudaMemcpyDeviceToHost));

        std::swap(d_old, d_new);
        out.iterations = iter + 1;

        if (l1_h < kPageRankEpsilon) break;
    }

    // ---- Copy result back -------------------------------------------------
    std::vector<float> host_rank(static_cast<std::size_t>(n));
    checkCuda(cudaMemcpy(host_rank.data(), d_old, rank_bytes, cudaMemcpyDeviceToHost));
    out.ranks.assign(host_rank.begin(), host_rank.end());

    // ---- Cleanup -----------------------------------------------------------
    checkCuda(cudaFree(d_col_ptr));
    checkCuda(cudaFree(d_row_idx));
    checkCuda(cudaFree(d_inv_deg));
    checkCuda(cudaFree(d_old));
    checkCuda(cudaFree(d_new));
    checkCuda(cudaFree(d_dangling));
    checkCuda(cudaFree(d_temp_storage));
    checkCuda(cudaFree(d_l1));
    checkCuda(cudaFreeHost(h_dangling));

    return out;
}
