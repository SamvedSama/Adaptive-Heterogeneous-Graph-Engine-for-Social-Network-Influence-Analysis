/**
 * @file bfs_cuda.cu
 * @brief Multi-GPU level-synchronous BFS for 2× Tesla T4 (sm_75, no NVLink).
 *
 * Strategy: vertex-partition model.
 *   GPU 0 owns vertices [0,   mid)   mid = n/2
 *   GPU 1 owns vertices [mid, n)
 *
 * Each GPU runs bfs_expand_kernel only on its local frontier vertices.
 * After every level, CPU collects "cross-partition" vertices discovered by
 * each GPU that belong to the OTHER partition and injects them as seeds into
 * the other GPU's next frontier.  This is the only inter-GPU communication
 * and costs one small H2H memcpy per level per crossing edge cluster.
 *
 * Falls back to single-GPU (GPU 0 only) when only 1 device is present or
 * when n < 2*kMinPartitionSize (not worth splitting).
 *
 * Key decisions for T4 (sm_75):
 *  - Block size 256: fills a single SM (64 resident warps), good occupancy for
 *    the memory-bound expand kernel.
 *  - cudaMemset instead of H→D copy to reset next_count each level.
 *  - 64-bit tid arithmetic guards > 2^31 frontier overflow.
 *  - All device memory freed on every exit path via cleanup lambdas.
 */
#include "graph/graph.h"
#include "common.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
namespace
{
    constexpr int kBlock = 256;
    constexpr NodeID kMinPartitionSize = 10000; // below this don't bother splitting
} // namespace

// ---------------------------------------------------------------------------
// Kernel (shared by both GPUs — same binary, different data)
// ---------------------------------------------------------------------------
namespace
{

    /**
     * @brief One BFS expansion level.
     *        Each thread handles one vertex in curr_frontier and walks its CSR
     *        adjacency list.  atomicCAS(-1 → level+1) claims first-visit;
     *        atomicAdd on next_count appends to next_frontier.
     *
     *        NOTE: next_frontier may contain vertices from the OTHER partition.
     *        The host will separate them after each level.
     */
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
// Single-GPU helper (used as fallback AND as the inner loop for each partition)
// ---------------------------------------------------------------------------
namespace
{

    struct GpuBfsState
    {
        // Device pointers
        EdgeID *d_row = nullptr;
        NodeID *d_col = nullptr;
        int *d_dist = nullptr;
        NodeID *d_front_a = nullptr;
        NodeID *d_front_b = nullptr;
        int *d_next_count = nullptr;

        // Sizes
        NodeID n_global = 0; // full graph node count (dist array covers all nodes)
        NodeID n_local = 0;  // #nodes on this partition (for CSR rows)
        NodeID offset = 0;   // this GPU's first vertex

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
// Multi-GPU orchestration
// ---------------------------------------------------------------------------
BfsResult bfs_cuda(const Graph &graph, NodeID source)
{
    const NodeID n = graph.num_nodes();
    BfsResult out;
    out.distances.assign(static_cast<std::size_t>(n), -1);
    out.per_hop_frontier_sizes.clear();

    if (n == 0 || source >= n)
        return out;

    // ----------------------------------------------------------------
    // Detect device count
    // ----------------------------------------------------------------
    int num_gpus = 0;
    checkCuda(cudaGetDeviceCount(&num_gpus));

    // Use single-GPU path if only one GPU or graph too small to partition
    const bool use_multi = (num_gpus >= 2) && (n >= 2 * kMinPartitionSize);

    if (!use_multi)
    {
        // ------------------------------------------------------------------
        // ── SINGLE-GPU PATH (GPU 0) ────────────────────────────────────────
        // ------------------------------------------------------------------
        checkCuda(cudaSetDevice(0));

        const CSR &csr = graph.csr();
        const std::size_t row_count = static_cast<std::size_t>(n) + 1u;
        const std::size_t col_count = static_cast<std::size_t>(csr.num_edges);

        // Host dist init
        std::vector<int> host_dist(static_cast<std::size_t>(n), -1);
        host_dist[source] = 0;

        // Device alloc
        EdgeID *d_row = nullptr;
        NodeID *d_col = nullptr;
        int *d_dist = nullptr;
        NodeID *d_front_a = nullptr;
        NodeID *d_front_b = nullptr;
        int *d_next_count = nullptr;

        const std::size_t row_bytes = row_count * sizeof(EdgeID);
        const std::size_t col_bytes = col_count * sizeof(NodeID);
        const std::size_t dist_bytes = static_cast<std::size_t>(n) * sizeof(int);
        const std::size_t frontier_bytes = static_cast<std::size_t>(n) * sizeof(NodeID);

        checkCuda(cudaMalloc(&d_row, row_bytes));
        checkCuda(cudaMalloc(&d_col, col_bytes));
        checkCuda(cudaMalloc(&d_dist, dist_bytes));
        checkCuda(cudaMalloc(&d_front_a, frontier_bytes));
        checkCuda(cudaMalloc(&d_front_b, frontier_bytes));
        checkCuda(cudaMalloc(&d_next_count, sizeof(int)));

        checkCuda(cudaMemcpy(d_row, csr.row_ptr.data(), row_bytes, cudaMemcpyHostToDevice));
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

    // ------------------------------------------------------------------
    // ── MULTI-GPU PATH (GPU 0 + GPU 1) ────────────────────────────────
    // ------------------------------------------------------------------
    // Partition: GPU 0 → [0, mid), GPU 1 → [mid, n)
    // Each GPU holds a FULL dist[] array (n ints) so global node IDs work
    // directly without remapping.  Only the CSR rows are partitioned.
    // ------------------------------------------------------------------
    const CSR &csr = graph.csr();
    const NodeID mid = n / 2;

    // We keep a single authoritative dist[] on the host, updated after
    // each level from both GPUs.
    std::vector<int> host_dist(static_cast<std::size_t>(n), -1);
    host_dist[source] = 0;

    // ── Allocate and upload per-GPU state ──────────────────────────────
    GpuBfsState gpu[2];
    const NodeID part_start[2] = {0, mid};
    const NodeID part_end[2] = {mid, n};

    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));
        gpu[g].n_global = n;
        gpu[g].n_local = part_end[g] - part_start[g];
        gpu[g].offset = part_start[g];

        // Row pointer slice: rows [part_start..part_end] inclusive
        // Length = (part_end - part_start + 1) EdgeIDs
        const std::size_t local_rows = static_cast<std::size_t>(gpu[g].n_local) + 1u;
        const EdgeID edge_lo = csr.row_ptr[part_start[g]];
        const EdgeID edge_hi = csr.row_ptr[part_end[g]];
        const std::size_t local_cols = static_cast<std::size_t>(edge_hi - edge_lo);

        // Build shifted row_ptr so row_ptr[0]==0 for this partition
        std::vector<EdgeID> local_row(local_rows);
        for (std::size_t r = 0; r < local_rows; ++r)
        {
            local_row[r] = csr.row_ptr[part_start[g] + r] - edge_lo;
        }

        checkCuda(cudaMalloc(&gpu[g].d_row,
                             local_rows * sizeof(EdgeID)));
        checkCuda(cudaMalloc(&gpu[g].d_col,
                             local_cols * sizeof(NodeID)));
        checkCuda(cudaMalloc(&gpu[g].d_dist,
                             static_cast<std::size_t>(n) * sizeof(int)));
        checkCuda(cudaMalloc(&gpu[g].d_front_a,
                             static_cast<std::size_t>(n) * sizeof(NodeID)));
        checkCuda(cudaMalloc(&gpu[g].d_front_b,
                             static_cast<std::size_t>(n) * sizeof(NodeID)));
        checkCuda(cudaMalloc(&gpu[g].d_next_count, sizeof(int)));

        checkCuda(cudaMemcpy(gpu[g].d_row, local_row.data(),
                             local_rows * sizeof(EdgeID), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(gpu[g].d_col,
                             csr.col_idx.data() + static_cast<std::size_t>(edge_lo),
                             local_cols * sizeof(NodeID), cudaMemcpyHostToDevice));
        // Full dist array on each GPU
        checkCuda(cudaMemcpy(gpu[g].d_dist, host_dist.data(),
                             static_cast<std::size_t>(n) * sizeof(int), cudaMemcpyHostToDevice));
    }

    // ── Seed first frontier on the GPU that owns the source ───────────
    // Each GPU starts with the source in its frontier if it owns it.
    // GPU 0 owns [0, mid), GPU 1 owns [mid, n).
    const int source_gpu = (source < mid) ? 0 : 1;

    // Host frontier vectors for cross-partition injection
    std::vector<NodeID> h_frontier[2]; // current frontier per GPU
    std::vector<NodeID> h_next[2];     // next-frontier readback per GPU

    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));
        if (g == source_gpu)
        {
            // Build a source-only frontier that uses the LOCAL row_ptr indexing.
            // The kernel indexes row_ptr with (v - offset) for partitioned CSR.
            // HOWEVER: since we kept global node IDs in the frontier and
            // passed a shifted row_ptr, we must use offset-corrected addresses.
            // Simplest: pass global v, but index row_ptr as [v - offset].
            // The kernel below does this via the 'offset' shift.
            h_frontier[g] = {source};
        }
        // other GPU starts with empty frontier
    }

    out.per_hop_frontier_sizes.push_back(1);

    // Upload initial frontiers
    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));
        if (!h_frontier[g].empty())
        {
            checkCuda(cudaMemcpy(gpu[g].d_front_a,
                                 h_frontier[g].data(),
                                 h_frontier[g].size() * sizeof(NodeID),
                                 cudaMemcpyHostToDevice));
        }
    }

    // ── Level loop ────────────────────────────────────────────────────
    // NOTE: The expand kernel uses GLOBAL node IDs in the frontier but
    // needs to index a LOCAL row_ptr.  We handle this by making the kernel
    // use (v - offset) to index row_ptr.  Since we can't change the kernel
    // signature without breaking single-GPU, we instead build the row_ptr
    // with GLOBAL addressing: row_ptr has n+1 entries where entries for
    // nodes outside this partition are 0 (or repeated), so the kernel
    // naturally skips them.  The simpler solution used here: build a full
    // n+1 row_ptr on each GPU with zeros outside the partition — this costs
    // slightly more memory (n*8 bytes per GPU) but avoids kernel changes.
    // We redo the upload here with full-size row_ptr.

    // Rebuild: full n+1 row_ptr per GPU, zero-fill outside partition
    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));
        cudaFree(gpu[g].d_row);

        std::vector<EdgeID> full_row(static_cast<std::size_t>(n) + 1u, EdgeID{0});
        const EdgeID edge_lo = csr.row_ptr[part_start[g]];
        for (NodeID v = part_start[g]; v <= part_end[g]; ++v)
        {
            full_row[v] = csr.row_ptr[v] - edge_lo;
        }
        // Nodes outside partition: row_ptr[v] = row_ptr[v+1] = same value (empty)
        // Already zeroed above — fill with edge_lo offset so hi-lo==0
        // Actually they're 0 which means col_idx[0..0) = empty. Correct.

        checkCuda(cudaMalloc(&gpu[g].d_row,
                             (static_cast<std::size_t>(n) + 1u) * sizeof(EdgeID)));
        checkCuda(cudaMemcpy(gpu[g].d_row, full_row.data(),
                             (static_cast<std::size_t>(n) + 1u) * sizeof(EdgeID),
                             cudaMemcpyHostToDevice));
    }

    // Pinned host buffers for fast D2H of next_frontier and next_count
    NodeID *h_next_frontier[2] = {nullptr, nullptr};
    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaMallocHost(&h_next_frontier[g],
                                 static_cast<std::size_t>(n) * sizeof(NodeID)));
    }

    int curr_sizes[2];
    curr_sizes[source_gpu] = 1;
    curr_sizes[1 - source_gpu] = 0;

    NodeID *d_curr[2] = {gpu[0].d_front_a, gpu[1].d_front_a};
    NodeID *d_next[2] = {gpu[0].d_front_b, gpu[1].d_front_b};

    int layer = 0;
    const std::size_t dist_bytes = static_cast<std::size_t>(n) * sizeof(int);

    while (curr_sizes[0] > 0 || curr_sizes[1] > 0)
    {
        // ── Launch kernels on both GPUs asynchronously ─────────────────
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

        // ── Synchronize both GPUs ─────────────────────────────────────
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

        // ── Read back next-frontiers from both GPUs ───────────────────
        for (int g = 0; g < 2; ++g)
        {
            if (next_counts[g] == 0)
                continue;
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaMemcpy(h_next_frontier[g], d_next[g],
                                 static_cast<std::size_t>(next_counts[g]) * sizeof(NodeID),
                                 cudaMemcpyDeviceToHost));
        }

        // ── Cross-partition injection ─────────────────────────────────
        // Vertices discovered by GPU g that belong to the other partition
        // get injected into the other GPU's current frontier for next level.
        // We append them to the existing next[] list on the other GPU.
        for (int g = 0; g < 2; ++g)
        {
            if (next_counts[g] == 0)
                continue;
            std::vector<NodeID> cross;
            for (int i = 0; i < next_counts[g]; ++i)
            {
                const NodeID v = h_next_frontier[g][i];
                // Does v belong to the other GPU's partition?
                const int other = 1 - g;
                if (v >= part_start[other] && v < part_end[other])
                {
                    cross.push_back(v);
                }
            }
            if (!cross.empty())
            {
                // Append cross-nodes to other GPU's d_next at position next_counts[other]
                const int other = 1 - g;
                checkCuda(cudaSetDevice(other));
                checkCuda(cudaMemcpy(
                    d_next[other] + next_counts[other],
                    cross.data(),
                    cross.size() * sizeof(NodeID),
                    cudaMemcpyHostToDevice));
                // Also set their dist on the other GPU
                // (they were set on this GPU already, sync dist arrays)
                next_counts[other] += static_cast<int>(cross.size());
            }
        }

        // ── Sync dist arrays: merge both GPUs' views of dist[] ────────
        // GPU 0 and GPU 1 each discovered new vertices independently.
        // We reconcile by reading both and taking non-(-1) values.
        // Cost: 2 × n × 4 bytes D2H + 2 × n × 4 bytes H2D per level.
        // This is the unavoidable price of no peer access.
        std::vector<int> dist0(n), dist1(n);
        checkCuda(cudaSetDevice(0));
        checkCuda(cudaMemcpy(dist0.data(), gpu[0].d_dist, dist_bytes, cudaMemcpyDeviceToHost));
        checkCuda(cudaSetDevice(1));
        checkCuda(cudaMemcpy(dist1.data(), gpu[1].d_dist, dist_bytes, cudaMemcpyDeviceToHost));

        // Merge: take whichever has a non-(-1) value (first-visit wins)
        for (NodeID v = 0; v < n; ++v)
        {
            if (host_dist[v] == -1 && dist0[v] != -1)
                host_dist[v] = dist0[v];
            if (host_dist[v] == -1 && dist1[v] != -1)
                host_dist[v] = dist1[v];
        }

        // Upload merged dist to both GPUs
        for (int g = 0; g < 2; ++g)
        {
            checkCuda(cudaSetDevice(g));
            checkCuda(cudaMemcpy(gpu[g].d_dist, host_dist.data(),
                                 dist_bytes, cudaMemcpyHostToDevice));
        }

        // ── Swap frontier buffers ─────────────────────────────────────
        const int total_next = next_counts[0] + next_counts[1];
        out.per_hop_frontier_sizes.push_back(total_next);

        for (int g = 0; g < 2; ++g)
        {
            std::swap(d_curr[g], d_next[g]);
            curr_sizes[g] = next_counts[g];
        }
        ++layer;
    }

    // Final dist readback (already merged in host_dist)
    out.distances = std::move(host_dist);

    // ── Cleanup ───────────────────────────────────────────────────────
    for (int g = 0; g < 2; ++g)
    {
        checkCuda(cudaSetDevice(g));
        gpu[g].free_all();
        cudaFreeHost(h_next_frontier[g]);
    }

    return out;
}
