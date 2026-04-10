/**
 * @file graph.cpp
 * @brief Graph construction helpers, neighbor queries, and degree statistics.
 */
#include "graph/graph.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

Graph::Graph(CSR csr, bool is_directed) : csr_(std::move(csr)), directed_(is_directed) {}

EdgeID Graph::out_degree(NodeID v) const {
    if (v >= csr_.num_nodes) {
        return 0;
    }
    const EdgeID lo = csr_.row_ptr[v];
    const EdgeID hi = csr_.row_ptr[v + 1];
    return hi - lo;
}

void Graph::neighbors(NodeID v, std::vector<NodeID>& out) const {
    out.clear();
    if (v >= csr_.num_nodes) {
        return;
    }
    const EdgeID lo = csr_.row_ptr[v];
    const EdgeID hi = csr_.row_ptr[v + 1];
    out.reserve(static_cast<std::size_t>(hi - lo));
    for (EdgeID e = lo; e < hi; ++e) {
        out.push_back(csr_.col_idx[static_cast<std::size_t>(e)]);
    }
}

void Graph::print_stats() const {
    std::printf("Graph stats: nodes=%u edges=%llu directed=%s\n",
                static_cast<unsigned>(csr_.num_nodes),
                static_cast<unsigned long long>(csr_.num_edges),
                directed_ ? "yes" : "no");
    if (csr_.num_nodes == 0) {
        std::printf("  (empty graph)\n");
        return;
    }

    EdgeID min_d = std::numeric_limits<EdgeID>::max();
    EdgeID max_d = 0;
    long double sum = 0.0L;
    for (NodeID v = 0; v < csr_.num_nodes; ++v) {
        const EdgeID d = out_degree(v);
        min_d = std::min(min_d, d);
        max_d = std::max(max_d, d);
        sum += static_cast<long double>(d);
    }
    if (min_d == std::numeric_limits<EdgeID>::max()) {
        min_d = 0;
    }
    const double mean = static_cast<double>(sum / static_cast<long double>(csr_.num_nodes));

    // Bucket counts: [0], [1-4], [5-16], [17-64], [65+]
    std::uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;
    for (NodeID v = 0; v < csr_.num_nodes; ++v) {
        const EdgeID d = out_degree(v);
        if (d == 0) {
            ++b0;
        } else if (d <= 4) {
            ++b1;
        } else if (d <= 16) {
            ++b2;
        } else if (d <= 64) {
            ++b3;
        } else {
            ++b4;
        }
    }

    std::printf("  Degree: min=%llu max=%llu mean=%.4f\n",
                static_cast<unsigned long long>(min_d),
                static_cast<unsigned long long>(max_d),
                mean);
    std::printf("  Degree buckets: deg=0:%llu (1-4]:%llu (5-16]:%llu (17-64]:%llu 65+:%llu\n",
                static_cast<unsigned long long>(b0),
                static_cast<unsigned long long>(b1),
                static_cast<unsigned long long>(b2),
                static_cast<unsigned long long>(b3),
                static_cast<unsigned long long>(b4));
}
