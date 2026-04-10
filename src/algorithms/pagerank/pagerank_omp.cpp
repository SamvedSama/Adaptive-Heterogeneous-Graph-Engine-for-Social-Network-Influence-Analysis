/**
 * @file pagerank_omp.cpp
 * @brief OpenMP-parallel PageRank: the per-vertex accumulation loop is split across
 *        threads while keeping the same numerical scheme as the sequential baseline.
 */
#include "graph/graph.h"

#include <cmath>
#include <cstddef>
#include <vector>
#include <omp.h>

/**
 * @brief Parallelizes the O(m) push of rank along outgoing edges; dangling mass and
 *        L1 norms use parallel reductions.
 */
PageRankResult pagerank_openmp(const Graph& graph) {
    const NodeID n = graph.num_nodes();
    PageRankResult out;
    out.ranks.assign(static_cast<std::size_t>(n), 0.0f);
    out.iterations = 0;

    if (n == 0) {
        return out;
    }

    const CSR& csr = graph.csr();
    const float inv_n = 1.0f / static_cast<float>(n);

    std::vector<Weight> old_rank(static_cast<std::size_t>(n), inv_n);
    std::vector<Weight> new_rank(static_cast<std::size_t>(n), 0.0f);

    const float base = (1.0f - kPageRankDamping) * inv_n;

    for (int iter = 0; iter < kPageRankMaxIter; ++iter) {
        float dangling = 0.0f;

#pragma omp parallel for schedule(static) reduction(+ : dangling)
        for (std::ptrdiff_t vi = 0; vi < static_cast<std::ptrdiff_t>(n); ++vi) {
            const NodeID v = static_cast<NodeID>(vi);
            const EdgeID lo = csr.row_ptr[v];
            const EdgeID hi = csr.row_ptr[v + 1];
            if (lo == hi) {
                dangling += old_rank[v];
            }
        }

        const float dangling_term = kPageRankDamping * dangling * inv_n;

#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t vi = 0; vi < static_cast<std::ptrdiff_t>(n); ++vi) {
            new_rank[static_cast<std::size_t>(vi)] = base + dangling_term;
        }

#pragma omp parallel for schedule(guided)
        for (std::ptrdiff_t vi = 0; vi < static_cast<std::ptrdiff_t>(n); ++vi) {
            const NodeID v = static_cast<NodeID>(vi);
            const EdgeID lo = csr.row_ptr[v];
            const EdgeID hi = csr.row_ptr[v + 1];
            if (lo == hi) {
                continue;
            }
            const float share = kPageRankDamping * old_rank[v] / static_cast<float>(hi - lo);
            for (EdgeID e = lo; e < hi; ++e) {
                const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
#pragma omp atomic update
                new_rank[u] += share;
            }
        }

        float l1 = 0.0f;

#pragma omp parallel for schedule(static) reduction(+ : l1)
        for (std::ptrdiff_t vi = 0; vi < static_cast<std::ptrdiff_t>(n); ++vi) {
            const std::size_t i = static_cast<std::size_t>(vi);
            l1 += std::fabs(new_rank[i] - old_rank[i]);
        }

        old_rank.swap(new_rank);
        out.iterations = iter + 1;

        if (l1 < kPageRankEpsilon) {
            break;
        }
    }

    out.ranks = std::move(old_rank);
    return out;
}
