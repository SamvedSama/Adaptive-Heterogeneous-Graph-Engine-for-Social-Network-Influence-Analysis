/**
 * @file wcc_omp.cpp
 * @brief Parallel weakly connected components via Jacobi label propagation on a
 *        symmetric adjacency built from directed edges (each arc yields mutual reachability).
 */
#include "graph/graph.h"

#include <algorithm>
#include <cstdio>
#include <vector>
#include <omp.h>

/**
 * @brief Builds an undirected adjacency list from the CSR so weak connectivity is captured.
 */
static void build_symmetric_adj(const CSR& csr, std::vector<std::vector<NodeID>>& adj) {
    const NodeID n = csr.num_nodes;
    adj.assign(static_cast<std::size_t>(n), {});

    for (NodeID v = 0; v < n; ++v) {
        const EdgeID lo = csr.row_ptr[v];
        const EdgeID hi = csr.row_ptr[v + 1];
        for (EdgeID e = lo; e < hi; ++e) {
            const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
            if (u == v) {
                continue;
            }
            adj[v].push_back(u);
            adj[u].push_back(v);
        }
    }

#pragma omp parallel for schedule(dynamic, 32)
    for (std::ptrdiff_t vi = 0; vi < static_cast<std::ptrdiff_t>(n); ++vi) {
        auto& nb = adj[static_cast<std::size_t>(vi)];
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }
}

/**
 * @brief Propagates the minimum neighbor label in parallel until a full sweep causes no change.
 */
WccResult wcc_openmp(const Graph& graph, bool verbose) {
    const NodeID n = graph.num_nodes();
    WccResult out;
    out.component_id.assign(static_cast<std::size_t>(n), 0);
    out.component_sizes.clear();

    if (n == 0) {
        if (verbose) {
            std::printf("WCC (OpenMP): 0 components (empty graph)\n");
        }
        return out;
    }

    const CSR& csr = graph.csr();
    std::vector<std::vector<NodeID>> adj;
    build_symmetric_adj(csr, adj);

    std::vector<NodeID> label(static_cast<std::size_t>(n));
    std::vector<NodeID> next_label(static_cast<std::size_t>(n));

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        label[static_cast<std::size_t>(i)] = static_cast<NodeID>(i);
    }

    bool changed = true;
    int iters = 0;
    /** High enough for long paths; Jacobi updates admit a slow diameter worst case. */
    const int max_iters = static_cast<int>(8 * n + 128);

    while (changed && iters < max_iters) {
        changed = false;

#pragma omp parallel for schedule(guided) reduction(|| : changed)
        for (std::ptrdiff_t vi = 0; vi < static_cast<std::ptrdiff_t>(n); ++vi) {
            const NodeID v = static_cast<NodeID>(vi);
            NodeID best = label[v];
            for (NodeID u : adj[static_cast<std::size_t>(v)]) {
                best = std::min(best, label[u]);
            }
            next_label[v] = best;
            if (best != label[v]) {
                changed = true;
            }
        }

        label.swap(next_label);
        ++iters;
    }

    // Final convergence pass without races: freeze labels and pull min neighbor again
    for (int sweep = 0; sweep < 3; ++sweep) {
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t vi = 0; vi < static_cast<std::ptrdiff_t>(n); ++vi) {
            const NodeID v = static_cast<NodeID>(vi);
            NodeID best = label[v];
            for (NodeID u : adj[static_cast<std::size_t>(v)]) {
                best = std::min(best, label[u]);
            }
            next_label[v] = best;
        }
        label.swap(next_label);
    }

    out.component_id = label;

    std::vector<NodeID> roots = label;
    std::sort(roots.begin(), roots.end());
    for (std::size_t i = 0; i < roots.size();) {
        std::size_t j = i;
        while (j < roots.size() && roots[j] == roots[i]) {
            ++j;
        }
        out.component_sizes.push_back({roots[i], static_cast<int>(j - i)});
        i = j;
    }

    std::sort(out.component_sizes.begin(), out.component_sizes.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    const int num_comp = static_cast<int>(out.component_sizes.size());
    const int largest = out.component_sizes.empty() ? 0 : out.component_sizes.front().second;

    if (verbose) {
        std::printf("WCC (OpenMP): components=%d largest=%d nodes (propagation_iters=%d)\n", num_comp,
                    largest, iters);
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
