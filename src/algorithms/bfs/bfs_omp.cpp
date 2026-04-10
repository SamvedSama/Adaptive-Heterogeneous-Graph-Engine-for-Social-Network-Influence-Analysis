/**
 * @file bfs_omp.cpp
 * @brief Level-synchronous BFS parallelized with OpenMP; each level expands the
 *        current frontier in parallel with critical sections guarding first visits.
 */
#include "graph/graph.h"

#include <omp.h>
#include <vector>

/**
 * @brief Parallel top-down BFS: threads split the current frontier and race to label
 *        unvisited neighbors; duplicates in the next frontier are avoided by the
 *        distance check inside the critical section.
 */
BfsResult bfs_openmp(const Graph& graph, NodeID source) {
    const NodeID n = graph.num_nodes();
    BfsResult out;
    out.distances.assign(static_cast<std::size_t>(n), -1);
    out.per_hop_frontier_sizes.clear();

    if (n == 0 || source >= n) {
        return out;
    }

    const CSR& csr = graph.csr();

    std::vector<NodeID> curr;
    std::vector<NodeID> next;
    curr.reserve(1024);
    next.reserve(1024);

    out.distances[source] = 0;
    curr.push_back(source);
    out.per_hop_frontier_sizes.push_back(static_cast<int>(curr.size()));

    while (!curr.empty()) {
        next.clear();

#pragma omp parallel default(none) shared(curr, next, csr, out)
        {
            std::vector<NodeID> thread_local_next;
            thread_local_next.reserve(64);

#pragma omp for schedule(guided)
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(curr.size()); ++i) {
                const NodeID v = curr[static_cast<std::size_t>(i)];
                const int dv = out.distances[v];
                const EdgeID lo = csr.row_ptr[v];
                const EdgeID hi = csr.row_ptr[v + 1];
                for (EdgeID e = lo; e < hi; ++e) {
                    const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
                    if (out.distances[u] < 0) {
#pragma omp critical(bfs_visit)
                        {
                            if (out.distances[u] < 0) {
                                out.distances[u] = dv + 1;
                                thread_local_next.push_back(u);
                            }
                        }
                    }
                }
            }

#pragma omp critical(bfs_merge)
            {
                next.insert(next.end(), thread_local_next.begin(), thread_local_next.end());
            }
        }

        if (next.empty()) {
            break;
        }

        out.per_hop_frontier_sizes.push_back(static_cast<int>(next.size()));
        curr.swap(next);
    }

    return out;
}
