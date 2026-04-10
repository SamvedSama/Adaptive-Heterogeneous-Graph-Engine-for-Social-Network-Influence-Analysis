/**
 * @file bfs_seq.cpp
 * @brief Sequential breadth-first search using a FIFO queue; records per-level
 *        frontier sizes alongside hop-limited distances from the source.
 */
#include "graph/graph.h"

#include <queue>
#include <vector>

/**
 * @brief Classic level-synchronous BFS: processes the frontier layer-by-layer so
 *        per-hop frontier sizes match the parallel implementations.
 */
BfsResult bfs_sequential(const Graph& graph, NodeID source) {
    const NodeID n = graph.num_nodes();
    BfsResult out;
    out.distances.assign(static_cast<std::size_t>(n), -1);
    out.per_hop_frontier_sizes.clear();

    if (n == 0 || source >= n) {
        return out;
    }

    std::queue<NodeID> q;
    out.distances[source] = 0;
    q.push(source);
    out.per_hop_frontier_sizes.push_back(1);

    const CSR& csr = graph.csr();

    while (!q.empty()) {
        const std::size_t level_count = q.size();
        std::vector<NodeID> level_nodes;
        level_nodes.reserve(level_count);
        for (std::size_t i = 0; i < level_count; ++i) {
            level_nodes.push_back(q.front());
            q.pop();
        }

        std::size_t pushed = 0;
        for (NodeID v : level_nodes) {
            const int dv = out.distances[v];
            const EdgeID lo = csr.row_ptr[v];
            const EdgeID hi = csr.row_ptr[v + 1];
            for (EdgeID e = lo; e < hi; ++e) {
                const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
                if (out.distances[u] < 0) {
                    out.distances[u] = dv + 1;
                    q.push(u);
                    ++pushed;
                }
            }
        }

        if (pushed > 0) {
            out.per_hop_frontier_sizes.push_back(static_cast<int>(pushed));
        }
    }

    return out;
}
