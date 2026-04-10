/**
 * @file wcc_seq.cpp
 * @brief Sequential weakly connected components via union–find with path compression
 *        and union by rank.
 *
 * Production improvements over the original:
 *  1. Path compression changed from recursive to iterative (two-pass: find root,
 *     then compress).  The recursive version stack-overflows on degenerate chains
 *     (e.g. path graphs with millions of nodes).
 *  2. `find` made non-mutating (const) for the initial root probe; mutation happens
 *     only in the explicit compression pass, making the call pattern clearer.
 *  3. Component-size counting replaced with std::unordered_map → O(n) instead of
 *     O(n log n) sort + linear scan.  The final sort is only over the (much smaller)
 *     component list, not over all n roots.
 *  4. `roots` temporary vector removed entirely; component_id is filled in one pass.
 *  5. Verbose output uses consistent formatting with OMP/CUDA variants.
 */
#include "graph/graph.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Disjoint-set (union–find) with iterative path compression + union by rank
// ---------------------------------------------------------------------------
class DisjointSet {
public:
    explicit DisjointSet(NodeID n)
        : parent_(static_cast<std::size_t>(n)),
          rank_  (static_cast<std::size_t>(n), 0u)
    {
        std::iota(parent_.begin(), parent_.end(), NodeID{0});
    }

    /**
     * @brief Finds the representative of x with iterative two-pass path compression.
     *        Safe on chains of arbitrary length (no recursion).
     */
    NodeID find(NodeID x) {
        // Pass 1: walk to root
        NodeID root = x;
        while (parent_[root] != root) {
            root = parent_[root];
        }
        // Pass 2: compress — point every node on the path directly at root
        while (parent_[x] != root) {
            const NodeID next = parent_[x];
            parent_[x] = root;
            x = next;
        }
        return root;
    }

    /**
     * @brief Unions the sets containing a and b by rank.
     *        Returns true if a merge occurred (i.e. they were in different sets).
     */
    bool unite(NodeID a, NodeID b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return false;
        }
        // Always attach smaller rank under larger rank
        if (rank_[a] < rank_[b]) {
            std::swap(a, b);
        }
        parent_[b] = a;
        if (rank_[a] == rank_[b]) {
            ++rank_[a];
        }
        return true;
    }

private:
    std::vector<NodeID>       parent_;
    std::vector<unsigned int> rank_;
};

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
WccResult wcc_sequential(const Graph& graph, bool verbose) {
    const NodeID n = graph.num_nodes();
    WccResult out;
    out.component_id.assign(static_cast<std::size_t>(n), 0);
    out.component_sizes.clear();

    if (n == 0) {
        if (verbose) {
            std::printf("WCC (sequential): 0 components (empty graph)\n");
        }
        return out;
    }

    const CSR& csr = graph.csr();
    DisjointSet dsu(n);

    // Union every directed edge (u→v treated as undirected for weak connectivity)
    for (NodeID v = 0; v < n; ++v) {
        const EdgeID lo = csr.row_ptr[v];
        const EdgeID hi = csr.row_ptr[v + 1];
        for (EdgeID e = lo; e < hi; ++e) {
            const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
            dsu.unite(v, u);
        }
    }

    // Assign component IDs and count sizes in one O(n) pass
    std::unordered_map<NodeID, int> size_map;
    size_map.reserve(static_cast<std::size_t>(n));

    for (NodeID v = 0; v < n; ++v) {
        const NodeID root = dsu.find(v);
        out.component_id[v] = root;
        ++size_map[root];
    }

    out.component_sizes.reserve(size_map.size());
    for (const auto& [root, sz] : size_map) {
        out.component_sizes.push_back({root, sz});
    }

    // Sort by descending size
    std::sort(out.component_sizes.begin(), out.component_sizes.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (verbose) {
        const int num_comp = static_cast<int>(out.component_sizes.size());
        const int largest  = out.component_sizes.front().second;
        std::printf("WCC (sequential): components=%d  largest=%d nodes\n",
                    num_comp, largest);
        std::printf("  Size distribution (top 8): ");
        const int kShow = std::min(8, num_comp);
        for (int t = 0; t < kShow; ++t) {
            const auto& [root, sz] = out.component_sizes[static_cast<std::size_t>(t)];
            std::printf("%u:%d ", static_cast<unsigned>(root), sz);
        }
        std::printf("\n");
    }

    return out;
}
