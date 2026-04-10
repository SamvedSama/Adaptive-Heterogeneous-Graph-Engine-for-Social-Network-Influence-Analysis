/**
 * @file analytics.h
 * @brief High-level social-network analytics that invoke graph algorithms through
 *        the adaptive scheduler with optional execution-mode overrides.
 */
#pragma once

#include "common.h"
#include "graph/graph.h"
#include "scheduler/scheduler.h"

#include <map>
#include <optional>
#include <utility>
#include <vector>

/** Forward declarations for algorithm entry points implemented in translation units. */
BfsResult bfs_sequential(const Graph& graph, NodeID source);
BfsResult bfs_openmp(const Graph& graph, NodeID source);
BfsResult bfs_cuda(const Graph& graph, NodeID source);

PageRankResult pagerank_sequential(const Graph& graph);
PageRankResult pagerank_openmp(const Graph& graph);
PageRankResult pagerank_cuda(const Graph& graph);

/** @param verbose When false, suppresses component statistics printed by the WCC kernels. */
WccResult wcc_sequential(const Graph& graph, bool verbose = true);
WccResult wcc_openmp(const Graph& graph, bool verbose = true);
WccResult wcc_cuda(const Graph& graph, bool verbose = true);

/**
 * @brief Orchestrates reachability, influence, and community summaries on a loaded graph.
 */
class SocialNetworkAnalytics {
public:
    SocialNetworkAnalytics(Graph& graph, Scheduler& scheduler);

    /** When set, all runs use this mode instead of the heuristic scheduler. */
    void set_mode_override(std::optional<ExecutionMode> mode);

    /**
     * @brief When true, each analytics phase times SEQ, OMP, and CUDA (if a GPU exists),
     *        prints a speedup table vs sequential, then prints the narrative using the
     *        scheduler-chosen backend’s result.
     */
    void set_compare_backends(bool enable);

    /**
     * @brief Runs BFS from source, prints hop histogram, returns hop→vertex count map.
     */
    std::map<int, int> reachability_report(NodeID source_node);

    /**
     * @brief Runs PageRank, prints the top-k list, returns sorted (node, score) pairs.
     */
    std::vector<std::pair<NodeID, Weight>> top_k_influencers(int k);

    /**
     * @brief Runs WCC, prints community stats and a coarse histogram, returns labels per node.
     */
    std::vector<NodeID> community_report();

private:
    ExecutionMode resolve_mode(AlgorithmType algo, const WorkloadProfile& profile) const;

    Graph& graph_;
    Scheduler& scheduler_;
    std::optional<ExecutionMode> mode_override_;
    bool compare_backends_ = false;
};
