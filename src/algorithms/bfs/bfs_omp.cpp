/**
 * @file bfs_omp.cpp
 * @brief Level-synchronous BFS parallelized with OpenMP.
 *
 * Key improvement over the original:
 *  - Replaced `#pragma omp critical` INSIDE the per-edge loop with a
 *    lock-free `std::atomic<int>` compare-exchange on the distances array.
 *    The original design serialized every neighbor discovery through a global
 *    lock, turning high-degree vertices into serial bottlenecks and making
 *    parallel performance worse than sequential for skewed-degree graphs.
 *  - Thread-local next-frontier vectors are still merged under a single
 *    critical section, but that fires only once per thread per level (O(threads))
 *    rather than once per discovered edge (O(edges)).
 *  - `schedule(guided)` is retained: it naturally load-balances across threads
 *    when frontier vertices have varying degrees.
 *
 * Compatibility: C++17, OpenMP 3.0+, same BfsResult / Graph / CSR types.
 */
#include "graph/graph.h"

#include <omp.h>
#include <atomic>
#include <vector>

BfsResult bfs_openmp(const Graph& graph, NodeID source) {
    const NodeID n = graph.num_nodes();
    BfsResult out;
    // We need an atomic view of distances during traversal; BfsResult still
    // stores plain int, so we use a parallel atomic array and copy at the end.
    out.distances.assign(static_cast<std::size_t>(n), -1);
    out.per_hop_frontier_sizes.clear();

    if (n == 0 || source >= n) {
        return out;
    }

    const CSR& csr = graph.csr();

    // Atomic distance array — enables lock-free CAS instead of critical sections.
    std::vector<std::atomic<int>> dist(static_cast<std::size_t>(n));
    for (auto& a : dist) {
        a.store(-1, std::memory_order_relaxed);
    }
    dist[source].store(0, std::memory_order_relaxed);

    std::vector<NodeID> curr, next;
    curr.reserve(1024);
    next.reserve(1024);

    curr.push_back(source);
    out.per_hop_frontier_sizes.push_back(1);

    while (!curr.empty()) {
        next.clear();

#pragma omp parallel default(none) shared(curr, next, csr, dist)
        {
            std::vector<NodeID> local_next;
            local_next.reserve(64);

#pragma omp for schedule(guided) nowait
            for (std::ptrdiff_t i = 0;
                 i < static_cast<std::ptrdiff_t>(curr.size()); ++i) {

                const NodeID v  = curr[static_cast<std::size_t>(i)];
                const int    dv = dist[v].load(std::memory_order_relaxed);
                const EdgeID lo = csr.row_ptr[v];
                const EdgeID hi = csr.row_ptr[v + 1];

                for (EdgeID e = lo; e < hi; ++e) {
                    const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];

                    // Lock-free first-visit claim: only the thread that wins
                    // the CAS from -1 → dv+1 appends u to its local frontier.
                    int expected = -1;
                    if (dist[u].compare_exchange_strong(
                            expected, dv + 1,
                            std::memory_order_relaxed,
                            std::memory_order_relaxed)) {
                        local_next.push_back(u);
                    }
                }
            }

            // Merge phase: one critical per thread per level, not per edge.
#pragma omp critical(bfs_merge)
            {
                next.insert(next.end(), local_next.begin(), local_next.end());
            }
        } // end parallel

        if (next.empty()) {
            break;
        }

        out.per_hop_frontier_sizes.push_back(static_cast<int>(next.size()));
        curr.swap(next);
    }

    // Copy atomic distances back to the plain-int result vector.
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        out.distances[i] = dist[i].load(std::memory_order_relaxed);
    }

    return out;
}
