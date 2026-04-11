/**
 * @file pagerank_cuda.cu
 * @brief Multi-GPU PageRank for 2× Tesla T4 (sm_75, no NVLink/peer-access).
 *
 * Strategy: row-partition the CSC.
 *   GPU 0 computes new_rank[0 .. mid-1]
 *   GPU 1 computes new_rank[mid .. n-1]
 *
 * Both GPUs need a read-only copy of old_rank (full n floats) because a
 * pull edge can come from any in-neighbour.  After each iteration:
 *   1. Each GPU copies its new_rank slice to the host.
 *   2. Host assembles the full new_rank.
 *   3. Host uploads it to both GPUs as the next old_rank.
 * This is the minimum data movement possible without peer access.
 *
 * Dangling mass and L1 norm are computed per-partition; host sums the
 * two partial values.  This removes the dangling D2H sync stall from
 * the critical path of the iteration — each GPU's dangling kernel runs
 * concurrently.
 *
 * Falls back to single-GPU (GPU 0) when only one device is present or
 * the graph is too small.
 *
 * T4 (sm_75) specific tuning:
 *  - Block 256 → 8 warps per block, good for memory-bound pull kernel.
 *  - Pinned host buffers for all scalars transferred per-iteration.
 *  - No CUB dependency; custom warp+block reductions via __shfl_down_sync.
 *  - Persistent CSC upload (static data uploaded once, never re-uploaded).
 */
#include "graph/graph.h"
#include "common.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
namespace
{

    __device__ __forceinline__ float warp_reduce_sum(float val)
    {
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        return val;
    }

    // ── Kernels (same as single-GPU; parameters scoped to a partition) ──────────

    /** rank[i] = inv_n for i in [0, n) */
    __global__ void pagerank_init_kernel(unsigned n, float inv_n, float *__restrict__ rank)
    {
        const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            rank[i] = inv_n;
    }

    /** new_rank[i] = base_val for i in [0, local_n) */
    __global__ void pagerank_fill_kernel(unsigned local_n, float base_val,
                                         float *__restrict__ new_rank)
    {
        const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < local_n)
            new_rank[i] = base_val;
    }

    /**
     * Dangling mass reduction over a PARTITION [offset, offset+local_n).
     * inv_deg and old_rank are indexed by LOCAL index (0..local_n-1).
     */
    __global__ void pagerank_dangling_kernel(unsigned local_n,
                                             const float *__restrict__ old_rank_local,
                                             const float *__restrict__ inv_deg_local,
                                             float *__restrict__ d_dangling)
    {
        extern __shared__ float sdata[];
        const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned lid = threadIdx.x;
        const unsigned wid = lid >> 5;
        const unsigned lane = lid & 31;

        float val = (i < local_n && inv_deg_local[i] == 0.0f) ? old_rank_local[i] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0)
            sdata[wid] = val;
        __syncthreads();

        const unsigned wpb = (blockDim.x + 31u) >> 5;
        if (wid == 0)
        {
            val = (lane < wpb) ? sdata[lane] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane == 0)
                atomicAdd(d_dangling, val);
        }
    }

    /**
     * Pull accumulation for a PARTITION [v_start, v_start+local_n).
     * col_ptr and row_idx are the LOCAL CSC (re-indexed from 0).
     * old_rank_full covers ALL n nodes (cross-partition reads).
     * new_rank_local covers [0, local_n).
     */
    __global__ void pagerank_pull_kernel(unsigned local_n,
                                         const EdgeID *__restrict__ col_ptr,
                                         const NodeID *__restrict__ row_idx,
                                         const float *__restrict__ old_rank_full,
                                         const float *__restrict__ inv_deg_full,
                                         float damping,
                                         float *__restrict__ new_rank_local)
    {
        const unsigned lv = blockIdx.x * blockDim.x + threadIdx.x; // local vertex
        if (lv >= local_n)
            return;

        const EdgeID lo = col_ptr[lv];
        const EdgeID hi = col_ptr[lv + 1];
        float acc = 0.0f;
        for (EdgeID e = lo; e < hi; ++e)
        {
            const NodeID j = row_idx[e]; // global neighbour ID
            acc += old_rank_full[j] * inv_deg_full[j];
        }
        new_rank_local[lv] += damping * acc;
    }

    /**
     * L1 convergence over a PARTITION.
     */
    __global__ void pagerank_l1_kernel(unsigned local_n,
                                       const float *__restrict__ old_rank_local,
                                       const float *__restrict__ new_rank_local,
                                       float *__restrict__ d_l1)
    {
        extern __shared__ float sdata[];
        const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned lid = threadIdx.x;
        const unsigned wid = lid >> 5;
        const unsigned lane = lid & 31;

        float val = (i < local_n) ? fabsf(new_rank_local[i] - old_rank_local[i]) : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0)
            sdata[wid] = val;
        __syncthreads();

        const unsigned wpb = (blockDim.x + 31u) >> 5;
        if (wid == 0)
        {
            val = (lane < wpb) ? sdata[lane] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane == 0)
                atomicAdd(d_l1, val);
        }
    }

} // namespace

// ---------------------------------------------------------------------------
// CSC builder (host, called once per pagerank_cuda invocation)
// ---------------------------------------------------------------------------
namespace
{

    struct CSCHost
    {
        std::vector<EdgeID> col_ptr; // length local_n + 1
        std::vector<NodeID> row_idx; // global source IDs
        std::vector<float> inv_deg;  // length local_n (out-degree of LOCAL nodes only)
    };

    /**
     * Build a LOCAL CSC for vertices [v_start, v_end) of the full CSR.
     * col_ptr is re-indexed from 0.
     * row_idx stores GLOBAL source node IDs (needed for old_rank lookup).
     * inv_deg_full[v] = 1/out_degree(v) for ALL v (global), needed by pull kernel.
     */
    static CSCHost build_local_csc(const CSR &csr,
                                   NodeID v_start, NodeID v_end,
                                   const std::vector<float> &inv_deg_full)
    {
        const NodeID local_n = v_end - v_start;
        const NodeID n_global = csr.num_nodes;

        CSCHost c;
        c.col_ptr.assign(static_cast<std::size_t>(local_n) + 1u, EdgeID{0});
        c.inv_deg.resize(static_cast<std::size_t>(local_n));

        // Copy local inv_deg slice
        for (NodeID lv = 0; lv < local_n; ++lv)
        {
            c.inv_deg[lv] = inv_deg_full[v_start + lv];
        }

        // Count in-edges to local vertices
        for (NodeID src = 0; src < n_global; ++src)
        {
            const EdgeID lo = csr.row_ptr[src];
            const EdgeID hi = csr.row_ptr[src + 1];
            for (EdgeID e = lo; e < hi; ++e)
            {
                const NodeID dst = csr.col_idx[e];
                if (dst >= v_start && dst < v_end)
                {
                    ++c.col_ptr[static_cast<std::size_t>(dst - v_start) + 1u];
                }
            }
        }
        // Prefix sum
        for (NodeID lv = 0; lv < local_n; ++lv)
            c.col_ptr[lv + 1] += c.col_ptr[lv];

        c.row_idx.resize(static_cast<std::size_t>(c.col_ptr[local_n]));

        // Fill row_idx with GLOBAL src IDs
        std::vector<EdgeID> fill_ptr(c.col_ptr.begin(), c.col_ptr.end());
        for (NodeID src = 0; src < n_global; ++src)
        {
            const EdgeID lo = csr.row_ptr[src];
            const EdgeID hi = csr.row_ptr[src + 1];
            for (EdgeID e = lo; e < hi; ++e)
            {
                const NodeID dst = csr.col_idx[e];
                if (dst >= v_start && dst < v_end)
                {
                    const NodeID lv = dst - v_start;
                    c.row_idx[fill_ptr[lv]++] = src; // global src ID
                }
            }
        }
        return c;
    }

    // Build global inv_deg array from CSR
    static std::vector<float> build_inv_deg(const CSR &csr)
    {
        const NodeID n = csr.num_nodes;
        std::vector<float> inv(static_cast<std::size_t>(n));
        for (NodeID v = 0; v < n; ++v)
        {
            const EdgeID od = csr.row_ptr[v + 1] - csr.row_ptr[v];
            inv[v] = (od > 0) ? (1.0f / static_cast<float>(od)) : 0.0f;
        }
        return inv;
    }

} // namespace

// ---------------------------------------------------------------------------
// Per-GPU device state
// ---------------------------------------------------------------------------
namespace
{
    struct GpuPrState
    {
        // Static (uploaded once)
        EdgeID *d_col_ptr = nullptr;
        NodeID *d_row_idx = nullptr;
        float *d_inv_deg_l = nullptr; // local inv_deg (local_n floats)
        float *d_inv_deg_f = nullptr; // full inv_deg  (n floats) — for pull kernel
        // Dynamic (updated each iteration)
        float *d_old_local = nullptr; // old_rank for LOCAL vertices (local_n floats)
        float *d_new_local = nullptr; // new_rank for LOCAL vertices
        float *d_old_full = nullptr;  // full old_rank (n floats) — read-only in pull
        // Scalars
        float *d_dangling = nullptr;
        float *d_l1 = nullptr;

        NodeID local_n = 0;
        NodeID offset = 0;

        void free_all()
        {
            cudaFree(d_col_ptr);
            d_col_ptr = nullptr;
            cudaFree(d_row_idx);
            d_row_idx = nullptr;
            cudaFree(d_inv_deg_l);
            d_inv_deg_l = nullptr;
            cudaFree(d_inv_deg_f);
            d_inv_deg_f = nullptr;
            cudaFree(d_old_local);
            d_old_local = nullptr;
            cudaFree(d_new_local);
            d_new_local = nullptr;
            cudaFree(d_old_full);
            d_old_full = nullptr;
            cudaFree(d_dangling);
            d_dangling = nullptr;
            cudaFree(d_l1);
            d_l1 = nullptr;
        }
    };
} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
PageRankResult pagerank_cuda(const Graph &graph)
{
    const NodeID n = graph.num_nodes();
    PageRankResult out;
    out.ranks.assign(static_cast<std::size_t>(n), 0.0f);
    out.iterations = 0;
    if (n == 0)
        return out;

    int num_gpus = 0;
    checkCuda(cudaGetDeviceCount(&num_gpus));

    const CSR &csr = graph.csr();
    const float inv_n = 1.0f / static_cast<float>(n);
    const float base_teleport = (1.0f - kPageRankDamping) * inv_n;
    const NodeID mid = n / 2;

    // Build global inv_deg once on host
    std::vector<float> inv_deg_full = build_inv_deg(csr);

    // ── Pinned host scalars ────────────────────────────────────────────────
    float *h_dangling[2] = {nullptr, nullptr};
    float *h_l1[2] = {nullptr, nullptr};
    for (int g = 0; g < (num_gpus >= 2 ? 2 : 1); ++g)
    {
        checkCuda(cudaMallocHost(&h_dangling[g], sizeof(float)));
        checkCuda(cudaMallocHost(&h_l1[g], sizeof(float)));
    }

    const bool use_multi = (num_gpus >= 2) && (n >= 20000);

    if (!use_multi)
    {
        // ── SINGLE-GPU PATH ────────────────────────────────────────────────
        checkCuda(cudaSetDevice(0));

        const CSCHost csc = build_local_csc(csr, 0, n, inv_deg_full);
        const std::size_t n1 = static_cast<std::size_t>(n) + 1u;
        const std::size_t m = csc.row_idx.size();
        const std::size_t rank_bytes = static_cast<std::size_t>(n) * sizeof(float);

        EdgeID *d_col_ptr = nullptr;
        NodeID *d_row_idx = nullptr;
        float *d_inv_deg = nullptr;
        float *d_old = nullptr;
        float *d_new = nullptr;
        float *d_dangling = nullptr;
        float *d_l1 = nullptr;

        checkCuda(cudaMalloc(&d_col_ptr, n1 * sizeof(EdgeID)));
        checkCuda(cudaMalloc(&d_row_idx, m * sizeof(NodeID)));
        checkCuda(cudaMalloc(&d_inv_deg, rank_bytes));
        checkCuda(cudaMalloc(&d_old, rank_bytes));
        checkCuda(cudaMalloc(&d_new, rank_bytes));
        checkCuda(cudaMalloc(&d_dangling, sizeof(float)));
        checkCuda(cudaMalloc(&d_l1, sizeof(float)));

        checkCuda(cudaMemcpy(d_col_ptr, csc.col_ptr.data(), n1 * sizeof(EdgeID), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_row_idx, csc.row_idx.data(), m * sizeof(NodeID), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_inv_deg, inv_deg_full.data(), rank_bytes, cudaMemcpyHostToDevice));

        const unsigned threads = static_cast<unsigned>(kDefaultCudaBlockSize);
        const unsigned blocks = (static_cast<unsigned>(n) + threads - 1u) / threads;
        const unsigned wpb = threads / 32u;
        const std::size_t smem = static_cast<std::size_t>(wpb) * sizeof(float);

        pagerank_init_kernel<<<blocks, threads>>>(static_cast<unsigned>(n), inv_n, d_old);
        checkCuda(cudaGetLastError());

        for (int iter = 0; iter < kPageRankMaxIter; ++iter)
        {
            checkCuda(cudaMemset(d_dangling, 0, sizeof(float)));
            pagerank_dangling_kernel<<<blocks, threads, smem>>>(
                static_cast<unsigned>(n), d_old, d_inv_deg, d_dangling);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaMemcpy(h_dangling[0], d_dangling, sizeof(float), cudaMemcpyDeviceToHost));

            const float base_val = base_teleport + kPageRankDamping * (*h_dangling[0]) * inv_n;
            pagerank_fill_kernel<<<blocks, threads>>>(static_cast<unsigned>(n), base_val, d_new);
            checkCuda(cudaGetLastError());
            pagerank_pull_kernel<<<blocks, threads>>>(
                static_cast<unsigned>(n), d_col_ptr, d_row_idx, d_old, d_inv_deg,
                kPageRankDamping, d_new);
            checkCuda(cudaGetLastError());

            checkCuda(cudaMemset(d_l1, 0, sizeof(float)));
            pagerank_l1_kernel<<<blocks, threads, smem>>>(
                static_cast<unsigned>(n), d_old, d_new, d_l1);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaMemcpy(h_l1[0], d_l1, sizeof(float), cudaMemcpyDeviceToHost));

            std::swap(d_old, d_new);
            out.iterations = iter + 1;
            if (*h_l1[0] < kPageRankEpsilon)
                break;
        }

        std::vector<float> host_rank(static_cast<std::size_t>(n));
        checkCuda(cudaMemcpy(host_rank.data(), d_old, rank_bytes, cudaMemcpyDeviceToHost));
        out.ranks = std::move(host_rank);

        cudaFree(d_col_ptr);
        cudaFree(d_row_idx);
        cudaFree(d_inv_deg);
        cudaFree(d_old);
        cudaFree(d_new);
        cudaFree(d_dangling);
        cudaFree(d_l1);
        cudaFreeHost(h_dangling[0]);
        cudaFreeHost(h_l1[0]);
        return out;
    }

    // ── MULTI-GPU PATH ──────────────────────────────────────────────────────
    const NodeID part_start[2] = {0, mid};
    const NodeID part_end[2] = {mid, n};

    // Build local CSCs on host (done once)
    CSCHost local_csc[2];
    for (int g = 0; g < 2; ++g)
    {
        local_csc[g] = build_local_csc(csr, part_start[g], part_end[g], inv_deg_full);
    }

    GpuPrState gpu[2];
    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));
        gpu[g].local_n = part_end[g] - part_start[g];
        gpu[g].offset = part_start[g];

        const std::size_t ln = static_cast<std::size_t>(gpu[g].local_n);
        const std::size_t ln1 = ln + 1u;
        const std::size_t m = local_csc[g].row_idx.size();
        const std::size_t fb = static_cast<std::size_t>(n) * sizeof(float);
        const std::size_t lb = ln * sizeof(float);

        checkCuda(cudaMalloc(&gpu[g].d_col_ptr, ln1 * sizeof(EdgeID)));
        checkCuda(cudaMalloc(&gpu[g].d_row_idx, m * sizeof(NodeID)));
        checkCuda(cudaMalloc(&gpu[g].d_inv_deg_l, lb));
        checkCuda(cudaMalloc(&gpu[g].d_inv_deg_f, fb));
        checkCuda(cudaMalloc(&gpu[g].d_old_local, lb));
        checkCuda(cudaMalloc(&gpu[g].d_new_local, lb));
        checkCuda(cudaMalloc(&gpu[g].d_old_full, fb));
        checkCuda(cudaMalloc(&gpu[g].d_dangling, sizeof(float)));
        checkCuda(cudaMalloc(&gpu[g].d_l1, sizeof(float)));

        // Upload static CSC data
        checkCuda(cudaMemcpy(gpu[g].d_col_ptr, local_csc[g].col_ptr.data(),
                             ln1 * sizeof(EdgeID), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(gpu[g].d_row_idx, local_csc[g].row_idx.data(),
                             m * sizeof(NodeID), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(gpu[g].d_inv_deg_l, local_csc[g].inv_deg.data(),
                             lb, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(gpu[g].d_inv_deg_f, inv_deg_full.data(),
                             fb, cudaMemcpyHostToDevice));
    }

    // Initialise ranks: old_full = 1/n everywhere, old_local = slice
    std::vector<float> host_rank_full(static_cast<std::size_t>(n), inv_n);
    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));
        const std::size_t fb = static_cast<std::size_t>(n) * sizeof(float);
        const std::size_t lb = static_cast<std::size_t>(gpu[g].local_n) * sizeof(float);
        checkCuda(cudaMemcpy(gpu[g].d_old_full,
                             host_rank_full.data(), fb, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(gpu[g].d_old_local,
                             host_rank_full.data() + part_start[g], lb, cudaMemcpyHostToDevice));
    }

    // Pinned buffer for rank slice readback
    float *h_rank_slice[2] = {nullptr, nullptr};
    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaMallocHost(&h_rank_slice[g],
                                 static_cast<std::size_t>(gpu[g].local_n) * sizeof(float)));
    }

    // Iteration loop
    for (int iter = 0; iter < kPageRankMaxIter; ++iter)
    {
        // ── 1. Dangling reduction per GPU (concurrent) ───────────────────────
        for (int g = 0; g < 2; ++g)
        {
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaMemset(gpu[g].d_dangling, 0, sizeof(float)));
            const unsigned ln = static_cast<unsigned>(gpu[g].local_n);
            const unsigned threads = static_cast<unsigned>(kDefaultCudaBlockSize);
            const unsigned blocks = (ln + threads - 1u) / threads;
            const unsigned wpb = threads / 32u;
            const std::size_t smem = static_cast<std::size_t>(wpb) * sizeof(float);
            pagerank_dangling_kernel<<<blocks, threads, smem>>>(
                ln, gpu[g].d_old_local, gpu[g].d_inv_deg_l, gpu[g].d_dangling);
            checkCuda(cudaGetLastError());
        }

        // ── Sync both and read dangling scalars ──────────────────────────────
        float dangling_sum = 0.0f;
        for (int g = 0; g < 2; ++g)
        {
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaMemcpy(h_dangling[g], gpu[g].d_dangling,
                                 sizeof(float), cudaMemcpyDeviceToHost));
            dangling_sum += *h_dangling[g];
        }
        const float base_val = base_teleport + kPageRankDamping * dangling_sum * inv_n;

        // ── 2. Fill + pull per GPU (concurrent) ──────────────────────────────
        for (int g = 0; g < 2; ++g)
        {
            checkCuda(cudaSetDevice(g));
            const unsigned ln = static_cast<unsigned>(gpu[g].local_n);
            const unsigned threads = static_cast<unsigned>(kDefaultCudaBlockSize);
            const unsigned blocks = (ln + threads - 1u) / threads;
            pagerank_fill_kernel<<<blocks, threads>>>(ln, base_val, gpu[g].d_new_local);
            checkCuda(cudaGetLastError());
            pagerank_pull_kernel<<<blocks, threads>>>(
                ln, gpu[g].d_col_ptr, gpu[g].d_row_idx,
                gpu[g].d_old_full,  // full old_rank for cross-partition reads
                gpu[g].d_inv_deg_f, // full inv_deg
                kPageRankDamping, gpu[g].d_new_local);
            checkCuda(cudaGetLastError());
        }

        // ── 3. L1 reduction per GPU (concurrent) ─────────────────────────────
        for (int g = 0; g < 2; ++g)
        {
            checkCuda(cudaSetDevice(g));
            const unsigned ln = static_cast<unsigned>(gpu[g].local_n);
            const unsigned threads = static_cast<unsigned>(kDefaultCudaBlockSize);
            const unsigned blocks = (ln + threads - 1u) / threads;
            const unsigned wpb = threads / 32u;
            const std::size_t smem = static_cast<std::size_t>(wpb) * sizeof(float);
            checkCuda(cudaMemset(gpu[g].d_l1, 0, sizeof(float)));
            pagerank_l1_kernel<<<blocks, threads, smem>>>(
                ln, gpu[g].d_old_local, gpu[g].d_new_local, gpu[g].d_l1);
            checkCuda(cudaGetLastError());
        }

        // ── Sync, read L1 scalars ────────────────────────────────────────────
        float l1_sum = 0.0f;
        for (int g = 0; g < 2; ++g)
        {
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaMemcpy(h_l1[g], gpu[g].d_l1, sizeof(float), cudaMemcpyDeviceToHost));
            l1_sum += *h_l1[g];
        }

        // ── 4. Swap local buffers ────────────────────────────────────────────
        for (int g = 0; g < 2; ++g)
        {
            std::swap(gpu[g].d_old_local, gpu[g].d_new_local);
        }

        // ── 5. Assemble full old_rank and broadcast to both GPUs ─────────────
        // This is the cross-GPU sync: each GPU uploads its new slice to host,
        // host merges, then uploads full array back to both GPUs.
        for (int g = 0; g < 2; ++g)
        {
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaMemcpy(h_rank_slice[g], gpu[g].d_old_local,
                                 static_cast<std::size_t>(gpu[g].local_n) * sizeof(float),
                                 cudaMemcpyDeviceToHost));
        }
        // Merge into host_rank_full
        for (int g = 0; g < 2; ++g)
        {
            std::memcpy(host_rank_full.data() + part_start[g], h_rank_slice[g],
                        static_cast<std::size_t>(gpu[g].local_n) * sizeof(float));
        }
        // Broadcast full rank to both GPUs' d_old_full
        const std::size_t fb = static_cast<std::size_t>(n) * sizeof(float);
        for (int g = 0; g < 2; ++g)
        {
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaMemcpy(gpu[g].d_old_full, host_rank_full.data(),
                                 fb, cudaMemcpyHostToDevice));
        }

        out.iterations = iter + 1;
        if (l1_sum < kPageRankEpsilon)
            break;
    }

    // ── Final rank readback (already in host_rank_full) ───────────────────
    out.ranks = std::move(host_rank_full);

    // ── Cleanup ───────────────────────────────────────────────────────────
    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));
        gpu[g].free_all();
        cudaFreeHost(h_dangling[g]);
        cudaFreeHost(h_l1[g]);
        cudaFreeHost(h_rank_slice[g]);
    }
    return out;
}
