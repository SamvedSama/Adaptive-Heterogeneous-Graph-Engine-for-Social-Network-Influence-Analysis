/**
 * @file main.cpp
 * @brief CLI driver for the adaptive graph engine: loads a graph, optionally forces a
 *        backend, and prints analytics with per-phase timings.
 *
 * === Google Colab (T4) setup — run in a cell before building =================
 * !apt-get update -qq && apt-get install -y -qq cmake libgomp1 build-essential
 * !nvcc --version
 * # Upload or clone this repository under /content/adaptive-graph-engine
 * %cd /content/adaptive-graph-engine
 * !mkdir -p build && cd build && cmake .. && cmake --build . -j4
 * !./graph_engine --graph ../data/processed/facebook_combined.txt --algorithm all \\
 *     --source 0 --topk 10 --mode auto
 * # Optional: add --compare to print SEQ/OMP/CUDA timings and speedup tables per phase.
 * ==============================================================================
 */
#include "analytics/analytics.h"
#include "common.h"
#include "graph/graph.h"
#include "scheduler/scheduler.h"
#include "timer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

static void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s --graph <filepath> --algorithm <bfs|pagerank|wcc|all> "
        "[--source <id>] [--topk <k>] [--mode <auto|seq|omp|cuda>] [--compare]\n",
        argv0);
    std::printf(
        "  --compare   Time SEQ, OMP, and CUDA (if available) for each phase and print speedup vs SEQ.\n");
}

static std::optional<ExecutionMode> parse_mode(const char* s) {
    if (std::strcmp(s, "auto") == 0) {
        return std::nullopt;
    }
    if (std::strcmp(s, "seq") == 0) {
        return ExecutionMode::SEQ;
    }
    if (std::strcmp(s, "omp") == 0) {
        return ExecutionMode::OMP;
    }
    if (std::strcmp(s, "cuda") == 0) {
        return ExecutionMode::CUDA;
    }
    return std::nullopt;
}

static bool streq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

int main(int argc, char** argv) {
    std::string graph_path;
    std::string algorithm = "all";
    NodeID source = 0;
    int topk = 10;
    std::optional<ExecutionMode> mode_override = std::nullopt;
    std::string mode_str = "auto";
    bool compare_backends = false;

    for (int i = 1; i < argc; ++i) {
        if (streq(argv[i], "--graph") && i + 1 < argc) {
            graph_path = argv[++i];
        } else if (streq(argv[i], "--algorithm") && i + 1 < argc) {
            algorithm = argv[++i];
        } else if (streq(argv[i], "--source") && i + 1 < argc) {
            source = static_cast<NodeID>(std::strtoul(argv[++i], nullptr, 10));
        } else if (streq(argv[i], "--topk") && i + 1 < argc) {
            topk = std::atoi(argv[++i]);
        } else if (streq(argv[i], "--mode") && i + 1 < argc) {
            mode_str = argv[++i];
            mode_override = parse_mode(mode_str.c_str());
            if (!streq(mode_str.c_str(), "auto") && !mode_override.has_value()) {
                std::fprintf(stderr, "Unknown --mode value: %s\n", mode_str.c_str());
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (streq(argv[i], "--compare")) {
            compare_backends = true;
        } else if (streq(argv[i], "-h") || streq(argv[i], "--help")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (graph_path.empty()) {
        std::fprintf(stderr, "Missing required --graph <filepath>\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (algorithm != "bfs" && algorithm != "pagerank" && algorithm != "wcc" && algorithm != "all") {
        std::fprintf(stderr, "Unknown --algorithm: %s\n", algorithm.c_str());
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

#ifdef _OPENMP
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(kDefaultOmpThreads);
    }
#endif

    WallTimer total;
    total.start();

    Graph g = load_edge_list(graph_path, false);
    g.print_stats();

    Scheduler scheduler;
    SocialNetworkAnalytics analytics(g, scheduler);
    if (mode_str == "auto") {
        analytics.set_mode_override(std::nullopt);
    } else {
        analytics.set_mode_override(mode_override);
    }
    analytics.set_compare_backends(compare_backends);

    if (algorithm == "bfs" || algorithm == "all") {
        std::printf("\n=== Phase: BFS reachability ===\n");
        analytics.reachability_report(source);
    }

    if (algorithm == "pagerank" || algorithm == "all") {
        std::printf("\n=== Phase: PageRank influencers ===\n");
        analytics.top_k_influencers(topk);
    }

    if (algorithm == "wcc" || algorithm == "all") {
        std::printf("\n=== Phase: Weakly connected communities ===\n");
        analytics.community_report();
    }

    total.stop();
    std::printf("\n=== Total wall time: %.3f ms ===\n", total.elapsed_ms());
    return EXIT_SUCCESS;
}
