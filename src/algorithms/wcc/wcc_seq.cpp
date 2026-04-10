/**
 * @file wcc_seq.cpp
 * @brief Sequential weakly connected components using union–find with path compression
 *        and union by rank; each directed edge joins its endpoints for weak connectivity.
 */
#include "graph/graph.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

/**
 * @brief Disjoint-set structure with path compression and union by rank.
 */
class DisjointSet {
public:
    explicit DisjointSet(NodeID n) : parent_(n), rank_(static_cast<std::size_t>(n), 0u) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    /** @brief Finds the representative of x with path compression. */
    NodeID find(NodeID x) {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    /**
     * @brief Unites the sets containing a and b; returns true if a merge happened.
     */
    bool unite(NodeID a, NodeID b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return false;
        }
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
    std::vector<NodeID> parent_;
    std::vector<unsigned int> rank_;
};

} // namespace

/**
 * @brief Runs DSU over all directed edges; weak components equal CC of the underlying
 *        graph where each arc is treated as an undirected link between its endpoints.
 */
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

    for (NodeID v = 0; v < n; ++v) {
        const EdgeID lo = csr.row_ptr[v];
        const EdgeID hi = csr.row_ptr[v + 1];
        for (EdgeID e = lo; e < hi; ++e) {
            const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
            dsu.unite(v, u);
        }
    }

    std::vector<NodeID> roots(static_cast<std::size_t>(n));
    for (NodeID v = 0; v < n; ++v) {
        roots[v] = dsu.find(v);
        out.component_id[v] = roots[v];
    }

    std::sort(roots.begin(), roots.end());
    std::vector<std::pair<NodeID, int>> sizes;
    for (std::size_t i = 0; i < roots.size();) {
        std::size_t j = i;
        while (j < roots.size() && roots[j] == roots[i]) {
            ++j;
        }
        sizes.push_back({roots[i], static_cast<int>(j - i)});
        i = j;
    }

    std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    out.component_sizes = std::move(sizes);

    const int num_comp = static_cast<int>(out.component_sizes.size());
    const int largest = out.component_sizes.empty() ? 0 : out.component_sizes.front().second;

    if (verbose) {
        std::printf("WCC (sequential): components=%d largest=%d nodes\n", num_comp, largest);
        std::printf("  Size distribution (top 8): ");
        const int kShow = std::min(8, static_cast<int>(out.component_sizes.size()));
        for (int t = 0; t < kShow; ++t) {
            std::printf("%u:%d ",
                        static_cast<unsigned>(out.component_sizes[static_cast<std::size_t>(t)].first),
                        out.component_sizes[static_cast<std::size_t>(t)].second);
        }
        std::printf("\n");
    }

    return out;
}
