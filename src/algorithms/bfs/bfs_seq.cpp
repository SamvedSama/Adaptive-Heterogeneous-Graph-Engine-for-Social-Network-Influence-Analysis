/**
 * @file bfs_seq.cpp
 * @brief Sequential breadth-first search using an index-pointer over a flat vector;
 *        records per-level frontier sizes alongside hop-limited distances from the source.
 *
 * Improvement over original:
 *  - Replaced two-pass level drain (queue → tmp vector → iterate) with a single flat
 *    vector + two index cursors (level_start / level_end).  No per-level heap allocation,
 *    better cache locality, identical algorithmic complexity.
 */
#include "graph/graph.h"

#include <vector>

BfsResult bfs_sequential(const Graph& graph, NodeID source) {
    const NodeID n = graph.num_nodes();
    BfsResult out;
    out.distances.assign(static_cast<std::size_t>(n), -1);
    out.per_hop_frontier_sizes.clear();

    if (n == 0 || source >= n) {
        return out;
    }

    const CSR& csr = graph.csr();

    // Flat frontier — avoids per-level vector allocation of the original.
    std::vector<NodeID> frontier;
    frontier.reserve(static_cast<std::size_t>(n));

    out.distances[source] = 0;
    frontier.push_back(source);
    out.per_hop_frontier_sizes.push_back(1);

    std::size_t level_start = 0;          // inclusive start of current level in frontier
    std::size_t level_end   = 1;          // exclusive end   of current level

    while (level_start < level_end) {
        const std::size_t next_start = level_end;   // new nodes go from here

        for (std::size_t i = level_start; i < level_end; ++i) {
            const NodeID v  = frontier[i];
            const int    dv = out.distances[v];
            const EdgeID lo = csr.row_ptr[v];
            const EdgeID hi = csr.row_ptr[v + 1];

            for (EdgeID e = lo; e < hi; ++e) {
                const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
                if (out.distances[u] < 0) {
                    out.distances[u] = dv + 1;
                    frontier.push_back(u);
                }
            }
        }

        level_start = level_end;
        level_end   = frontier.size();

        if (level_end > level_start) {
            out.per_hop_frontier_sizes.push_back(
                static_cast<int>(level_end - level_start));
        }
    }

    return out;
}
