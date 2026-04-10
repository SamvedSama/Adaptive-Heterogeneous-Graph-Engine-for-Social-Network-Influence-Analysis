/**
 * @file graph.h
 * @brief In-memory graph as CSR with directed/undirected semantics and basic
 *        query helpers plus degree-distribution statistics.
 */
#pragma once

#include "common.h"
#include <string>
#include <vector>

/**
 * @brief CSR-backed graph with optional undirected interpretation at load time.
 */
class Graph {
public:
    Graph() = default;

    explicit Graph(CSR csr, bool is_directed);

    NodeID num_nodes() const { return csr_.num_nodes; }
    EdgeID num_edges() const { return csr_.num_edges; }

    bool is_directed() const { return directed_; }

    /** Out-degree of v (total outgoing CSR entries, including multi-edges if any). */
    EdgeID out_degree(NodeID v) const;

    /**
     * @brief Copies neighbor IDs for vertex v into an output vector (out-edges only).
     */
    void neighbors(NodeID v, std::vector<NodeID>& out) const;

    const CSR& csr() const { return csr_; }
    CSR& csr_mut() { return csr_; }

    /** Prints node/edge counts and a compact degree summary (min/max/mean + buckets). */
    void print_stats() const;

private:
    CSR csr_{};
    bool directed_ = true;
};

/**
 * @brief Loads an edge list from disk into a remapped CSR Graph (see loader.cpp).
 */
Graph load_edge_list(const std::string& filepath, bool default_directed = false);
