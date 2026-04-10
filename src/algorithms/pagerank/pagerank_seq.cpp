/**
 * @file pagerank_seq.cpp
 * @brief Sequential power-method PageRank with damping, dangling-node redistribution,
 *        and L1 stopping criterion against kPageRankEpsilon.
 */
#include "graph/graph.h"

#include <cmath>
#include <cstddef>
#include <vector>

/**
 * @brief Applies one pull-style update: each vertex accumulates incoming mass scaled
 *        by out-degree; dangling rank mass is spread uniformly after the edge term.
 */
PageRankResult pagerank_sequential(const Graph& graph) {
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
        for (NodeID v = 0; v < n; ++v) {
            const EdgeID lo = csr.row_ptr[v];
            const EdgeID hi = csr.row_ptr[v + 1];
            if (lo == hi) {
                dangling += old_rank[v];
            }
        }

        const float dangling_term = kPageRankDamping * dangling * inv_n;

        std::fill(new_rank.begin(), new_rank.end(), base + dangling_term);

        for (NodeID v = 0; v < n; ++v) {
            const EdgeID lo = csr.row_ptr[v];
            const EdgeID hi = csr.row_ptr[v + 1];
            if (lo == hi) {
                continue;
            }
            const float share = kPageRankDamping * old_rank[v] / static_cast<float>(hi - lo);
            for (EdgeID e = lo; e < hi; ++e) {
                const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
                new_rank[u] += share;
            }
        }

        float l1 = 0.0f;
        for (NodeID v = 0; v < n; ++v) {
            l1 += std::fabs(new_rank[v] - old_rank[v]);
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
