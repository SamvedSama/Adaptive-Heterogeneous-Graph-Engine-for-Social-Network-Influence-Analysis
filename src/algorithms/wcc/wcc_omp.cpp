/**
 * @file wcc_omp.cpp
 * @brief Parallel weakly connected components via a concurrent union–find
 *        with atomic path compression (Jayanti–Tarjan style).
 *
 * Why the original approach was wrong
 * ------------------------------------
 * The original used Jacobi label propagation (min-label spreading) with a
 * convergence bound of `8*n` iterations.  This has two serious flaws:
 *
 *  1. CORRECTNESS: Jacobi propagation on the SYMMETRIC adjacency needs O(diameter)
 *     iterations to converge.  For a path graph of n nodes the diameter is n-1,
 *     making the `8*n` bound technically safe but the "3 extra sweeps" post-loop
 *     is NOT a valid correctness guarantee — if the main loop exits at max_iters
 *     before true convergence, the 3 extra sweeps can leave wrong labels.
 *
 *  2. PERFORMANCE: Building a full symmetric adjacency list (vector-of-vectors)
 *     doubles memory, fragments the heap, and the sort+unique per vertex adds
 *     O(E log E) work before any WCC logic begins.
 *
 * Production approach: parallel union–find (Iyer et al. / Anderson & Wenger style)
 * ----------------------------------------------------------------------------------
 * We use a shared `parent[]` array of std::atomic<NodeID> and implement
 * concurrent `find` (with atomic path compression) and `unite` (with CAS-based
 * linking).  This is correct by construction — no iteration bound needed — and
 * works directly on the CSR without building a symmetric copy.
 *
 * Algorithm sketch
 * ----------------
 *  - Initialise parent[v] = v for all v (each node is its own root).
 *  - In parallel, for each directed edge (v → u), call unite(v, u).
 *    Because we call unite for BOTH directions implicitly (every directed edge
 *    is treated as undirected), weak connectivity is captured.
 *  - find() uses iterative path splitting (Rem's algorithm) which is safe for
 *    concurrent access without locks and keeps chains short.
 *  - unite() uses a CAS loop to attach the larger root under the smaller root.
 *
 * Compatibility: C++17, OpenMP 3.0+, same BfsResult / Graph / CSR types.
 */
#include "graph/graph.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <omp.h>

namespace
{

    // ---------------------------------------------------------------------------
    // Concurrent union–find over std::atomic<NodeID>
    // ---------------------------------------------------------------------------

    /**
     * @brief Finds the root of x using iterative path splitting (Rem's algorithm).
     *        Path splitting is safe under concurrent access: each node's parent
     *        pointer only ever moves closer to the root, so stale reads are harmless.
     */
    NodeID find_root(std::vector<std::atomic<NodeID>> &parent, NodeID x)
    {
        while (true)
        {
            const NodeID p = parent[x].load(std::memory_order_relaxed);
            const NodeID gp = parent[p].load(std::memory_order_relaxed);
            if (p == gp)
            {
                return p; // p is a root
            }
            // Path splitting: make x point to its grandparent (best-effort, may fail)
            NodeID expected = p;
            parent[x].compare_exchange_weak(
                expected, gp,
                std::memory_order_relaxed, std::memory_order_relaxed);
            x = p;
        }
    }

    /**
     * @brief Unites the components containing a and b.
     *        Links the larger root ID under the smaller root ID (min-root convention)
     *        using a CAS loop to resolve races.
     */
    void unite(std::vector<std::atomic<NodeID>> &parent, NodeID a, NodeID b)
    {
        while (true)
        {
            NodeID ra = find_root(parent, a);
            NodeID rb = find_root(parent, b);

            if (ra == rb)
            {
                return; // already in the same component
            }

            // Canonical ordering: attach larger root under smaller root
            if (ra > rb)
            {
                std::swap(ra, rb);
            }

            // CAS: try to set parent[rb] = ra (rb was a root, i.e. parent[rb] == rb)
            NodeID expected = rb;
            if (parent[rb].compare_exchange_strong(
                    expected, ra,
                    std::memory_order_relaxed, std::memory_order_relaxed))
            {
                return; // succeeded
            }
            // Another thread modified parent[rb] — retry from the top
        }
    }

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
WccResult wcc_openmp(const Graph &graph, bool verbose)
{
    const NodeID n = graph.num_nodes();
    WccResult out;
    out.component_id.assign(static_cast<std::size_t>(n), 0);
    out.component_sizes.clear();

    if (n == 0)
    {
        if (verbose)
        {
            std::printf("WCC (OpenMP): 0 components (empty graph)\n");
        }
        return out;
    }

    const CSR &csr = graph.csr();

    // ---- Initialise atomic parent array ------------------------------------
    std::vector<std::atomic<NodeID>> parent(static_cast<std::size_t>(n));

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(n); ++v)
    {
        parent[static_cast<std::size_t>(v)].store(
            static_cast<NodeID>(v), std::memory_order_relaxed);
    }

    // ---- Parallel union over all directed edges ----------------------------
    // Each directed edge (v → u) is treated as undirected for weak connectivity.
    // We do NOT need to build a symmetric copy: unite(v, u) == unite(u, v).
#pragma omp parallel for schedule(guided)
    for (std::ptrdiff_t vi = 0; vi < static_cast<std::ptrdiff_t>(n); ++vi)
    {
        const NodeID v = static_cast<NodeID>(vi);
        const EdgeID lo = csr.row_ptr[v];
        const EdgeID hi = csr.row_ptr[v + 1];
        for (EdgeID e = lo; e < hi; ++e)
        {
            const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
            if (u != v)
            {
                unite(parent, v, u);
            }
        }
    }

    // ---- Final root compression (sequential, cheap) ------------------------
    // After all unions, some nodes may still point at intermediate nodes rather
    // than the true root.  One sequential pass of iterative find fixes this.
    for (NodeID v = 0; v < n; ++v)
    {
        const NodeID root = find_root(parent, v);
        parent[v].store(root, std::memory_order_relaxed);
        out.component_id[v] = root;
    }

    // ---- Count component sizes in O(n) -------------------------------------
    std::unordered_map<NodeID, int> size_map;
    size_map.reserve(static_cast<std::size_t>(n));
    for (NodeID v = 0; v < n; ++v)
    {
        ++size_map[out.component_id[v]];
    }

    out.component_sizes.reserve(size_map.size());
    for (const auto &[root, sz] : size_map)
    {
        out.component_sizes.push_back({root, sz});
    }

    std::sort(out.component_sizes.begin(), out.component_sizes.end(),
              [](const auto &a, const auto &b)
              { return a.second > b.second; });

    if (verbose)
    {
        const int num_comp = static_cast<int>(out.component_sizes.size());
        const int largest = out.component_sizes.front().second;
        std::printf("WCC (OpenMP): components=%d  largest=%d nodes\n",
                    num_comp, largest);
        std::printf("  Size distribution (top 8): ");
        const int kShow = std::min(8, num_comp);
        for (int t = 0; t < kShow; ++t)
        {
            const auto &[root, sz] = out.component_sizes[static_cast<std::size_t>(t)];
            std::printf("%u:%d ", static_cast<unsigned>(root), sz);
        }
        std::printf("\n");
    }

    return out;
}
