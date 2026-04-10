/**
 * @file analytics.cpp
 * @brief Implements reachability, influence, and community workflows with explicit
 *        [compute] logs, optional multi-backend timing (--compare), and speedup tables.
 */
#include "analytics/analytics.h"
#include "scheduler/scheduler.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

SocialNetworkAnalytics::SocialNetworkAnalytics(Graph& graph, Scheduler& scheduler)
    : graph_(graph), scheduler_(scheduler), mode_override_(std::nullopt), compare_backends_(false) {}

void SocialNetworkAnalytics::set_mode_override(std::optional<ExecutionMode> mode) {
    mode_override_ = mode;
}

void SocialNetworkAnalytics::set_compare_backends(bool enable) { compare_backends_ = enable; }

ExecutionMode SocialNetworkAnalytics::resolve_mode(AlgorithmType algo,
                                                   const WorkloadProfile& profile) const {
    if (mode_override_.has_value()) {
        std::printf("[analytics] mode override active → forced backend selection\n");
        return *mode_override_;
    }
    return scheduler_.choose(algo, profile);
}

static const char* backend_label(ExecutionMode m) {
    switch (m) {
    case ExecutionMode::SEQ:
        return "SEQ";
    case ExecutionMode::OMP:
        return "OMP";
    case ExecutionMode::CUDA:
        return "CUDA";
    }
    return "?";
}

/**
 * @brief Prints a three-row (or two-row) timing table with speedup relative to sequential time.
 */
static void print_speedup_table(const char* title,
                                double t_seq,
                                double t_omp,
                                double t_cuda,
                                bool ran_cuda) {
    std::printf("\n%s\n", title);
    std::printf("  %-8s  %10s  %16s\n", "backend", "time_ms", "speedup_vs_SEQ");
    std::printf("  %-8s  %10.3f  %16.2fx\n", "SEQ", t_seq, 1.0);
    if (t_seq > 0.0 && t_omp > 0.0) {
        std::printf("  %-8s  %10.3f  %16.2fx\n", "OMP", t_omp, t_seq / t_omp);
    } else {
        std::printf("  %-8s  %10.3f  %16s\n", "OMP", t_omp, "n/a");
    }
    if (ran_cuda) {
        if (t_seq > 0.0 && t_cuda > 0.0) {
            std::printf("  %-8s  %10.3f  %16.2fx\n", "CUDA", t_cuda, t_seq / t_cuda);
        } else {
            std::printf("  %-8s  %10.3f  %16s\n", "CUDA", t_cuda, "n/a");
        }
    } else {
        std::printf("  %-8s  %10s  %16s\n", "CUDA", "-", "no GPU");
    }
}

static WorkloadProfile make_profile_bfs(const Graph& g) {
    WorkloadProfile p;
    p.num_nodes = g.num_nodes();
    p.num_edges = g.num_edges();
    const double est = std::sqrt(static_cast<double>(std::max<EdgeID>(1, g.num_edges()))) + 1.0;
    p.frontier_size = std::max<NodeID>(1, static_cast<NodeID>(est));
    if (p.num_nodes > 0) {
        p.avg_degree = static_cast<float>(p.num_edges) / static_cast<float>(p.num_nodes);
        p.active_ratio = static_cast<float>(p.frontier_size) / static_cast<float>(p.num_nodes);
    }
    p.is_directed = g.is_directed();
    return p;
}

static WorkloadProfile make_profile_pr(const Graph& g) {
    WorkloadProfile p;
    p.num_nodes = g.num_nodes();
    p.num_edges = g.num_edges();
    p.frontier_size = 1;
    if (p.num_nodes > 0) {
        p.avg_degree = static_cast<float>(p.num_edges) / static_cast<float>(p.num_nodes);
        p.active_ratio = 1.0f / static_cast<float>(p.num_nodes);
    }
    p.is_directed = g.is_directed();
    return p;
}

static WorkloadProfile make_profile_wcc(const Graph& g) { return make_profile_pr(g); }

static BfsResult dispatch_bfs(ExecutionMode mode, const Graph& g, NodeID source) {
    switch (mode) {
    case ExecutionMode::SEQ:
        return bfs_sequential(g, source);
    case ExecutionMode::OMP:
        return bfs_openmp(g, source);
    case ExecutionMode::CUDA:
        return bfs_cuda(g, source);
    }
    return bfs_sequential(g, source);
}

static PageRankResult dispatch_pagerank(ExecutionMode mode, const Graph& g) {
    switch (mode) {
    case ExecutionMode::SEQ:
        return pagerank_sequential(g);
    case ExecutionMode::OMP:
        return pagerank_openmp(g);
    case ExecutionMode::CUDA:
        return pagerank_cuda(g);
    }
    return pagerank_sequential(g);
}

static WccResult dispatch_wcc(ExecutionMode mode, const Graph& g, bool verbose) {
    switch (mode) {
    case ExecutionMode::SEQ:
        return wcc_sequential(g, verbose);
    case ExecutionMode::OMP:
        return wcc_openmp(g, verbose);
    case ExecutionMode::CUDA:
        return wcc_cuda(g, verbose);
    }
    return wcc_sequential(g, verbose);
}

static void print_bfs_hop_summary(NodeID source, const BfsResult& res, double wall_ms) {
    std::map<int, int> hop_counts;
    for (int d : res.distances) {
        if (d >= 0) {
            hop_counts[d] += 1;
        }
    }
    std::printf("\n[report] BFS hop histogram (wall clock for timed phase(s): %.3f ms)\n", wall_ms);
    std::printf("From node %u:\n", static_cast<unsigned>(source));
    for (const auto& kv : hop_counts) {
        std::printf("  reachable in %d hop(s): %d vertices\n", kv.first, kv.second);
    }
}

static void print_wcc_summary(const WccResult& wcc, double wall_ms) {
    const int total_communities = static_cast<int>(wcc.component_sizes.size());
    const int largest = wcc.component_sizes.empty() ? 0 : wcc.component_sizes.front().second;

    std::printf("\n[report] WCC summary (wall clock for timed phase(s): %.3f ms)\n", wall_ms);
    std::printf("Communities: total=%d largest=%d nodes\n", total_communities, largest);

    std::uint64_t b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;
    for (const auto& cs : wcc.component_sizes) {
        const int sz = cs.second;
        if (sz <= 1) {
            ++b1;
        } else if (sz <= 4) {
            ++b2;
        } else if (sz <= 16) {
            ++b3;
        } else if (sz <= 64) {
            ++b4;
        } else {
            ++b5;
        }
    }
    std::printf("  Histogram (community counts by size bucket):\n");
    std::printf("    size=1: %llu  (2-4]: %llu  (5-16]: %llu  (17-64]: %llu  65+: %llu\n",
                static_cast<unsigned long long>(b1), static_cast<unsigned long long>(b2),
                static_cast<unsigned long long>(b3), static_cast<unsigned long long>(b4),
                static_cast<unsigned long long>(b5));
}

std::map<int, int> SocialNetworkAnalytics::reachability_report(NodeID source_node) {
    const WorkloadProfile prof = make_profile_bfs(graph_);
    const ExecutionMode chosen = resolve_mode(AlgorithmType::BFS, prof);
    const bool ran_cuda = is_gpu_available();

    if (compare_backends_) {
        std::printf("\n[compute] BFS multi-backend comparison (source=%u)\n",
                    static_cast<unsigned>(source_node));

        WallTimer wseq;
        std::printf("[compute] BFS starting backend=SEQ ...\n");
        wseq.start();
        BfsResult r_seq = bfs_sequential(graph_, source_node);
        wseq.stop();
        const double t_seq = wseq.elapsed_ms();
        std::printf("[compute] BFS SEQ finished: %.3f ms\n", t_seq);

        WallTimer womp;
        std::printf("[compute] BFS starting backend=OMP ...\n");
        womp.start();
        BfsResult r_omp = bfs_openmp(graph_, source_node);
        womp.stop();
        const double t_omp = womp.elapsed_ms();
        std::printf("[compute] BFS OMP finished: %.3f ms\n", t_omp);

        double t_cuda = 0.0;
        BfsResult r_cuda;
        bool cuda_ok = false;
        if (ran_cuda) {
            std::printf("[compute] BFS starting backend=CUDA ...\n");
            WallTimer wc;
            wc.start();
            r_cuda = bfs_cuda(graph_, source_node);
            wc.stop();
            t_cuda = wc.elapsed_ms();
            cuda_ok = true;
            std::printf("[compute] BFS CUDA finished: %.3f ms\n", t_cuda);
        }

        print_speedup_table("[compute] BFS speedup summary (SEQ baseline)", t_seq, t_omp, t_cuda,
                            cuda_ok);

        BfsResult* picked = &r_seq;
        if (chosen == ExecutionMode::OMP) {
            picked = &r_omp;
        } else if (chosen == ExecutionMode::CUDA && cuda_ok) {
            picked = &r_cuda;
        }

        const double bench_wall_ms = t_seq + t_omp + (cuda_ok ? t_cuda : 0.0);
        print_bfs_hop_summary(source_node, *picked, bench_wall_ms);
        std::printf("[report] Hop counts above use scheduler-selected backend=%s (sum of timed runs=%.3f ms).\n",
                    backend_label(chosen), bench_wall_ms);

        std::map<int, int> hop_counts;
        for (int d : picked->distances) {
            if (d >= 0) {
                hop_counts[d] += 1;
            }
        }
        return hop_counts;
    }

    WallTimer timer;
    std::printf("\n[compute] BFS starting backend=%s (source=%u)\n", backend_label(chosen),
                static_cast<unsigned>(source_node));
    timer.start();
    BfsResult res = dispatch_bfs(chosen, graph_, source_node);
    timer.stop();
    std::printf("[compute] BFS backend=%s total time: %.3f ms\n", backend_label(chosen),
                timer.elapsed_ms());

    std::map<int, int> hop_counts;
    for (int d : res.distances) {
        if (d >= 0) {
            hop_counts[d] += 1;
        }
    }

    std::printf("[report] reachability from node %u:\n", static_cast<unsigned>(source_node));
    for (const auto& kv : hop_counts) {
        std::printf("  reachable in %d hop(s): %d vertices\n", kv.first, kv.second);
    }

    return hop_counts;
}

std::vector<std::pair<NodeID, Weight>> SocialNetworkAnalytics::top_k_influencers(int k) {
    const WorkloadProfile prof = make_profile_pr(graph_);
    const ExecutionMode chosen = resolve_mode(AlgorithmType::PAGERANK, prof);
    const bool ran_cuda = is_gpu_available();

    if (compare_backends_) {
        std::printf("\n[compute] PageRank multi-backend comparison\n");

        WallTimer wseq;
        std::printf("[compute] PageRank starting backend=SEQ ...\n");
        wseq.start();
        PageRankResult pr_seq = pagerank_sequential(graph_);
        wseq.stop();
        const double t_seq = wseq.elapsed_ms();
        std::printf("[compute] PageRank SEQ finished: %.3f ms (iters=%d)\n", t_seq, pr_seq.iterations);

        WallTimer womp;
        std::printf("[compute] PageRank starting backend=OMP ...\n");
        womp.start();
        PageRankResult pr_omp = pagerank_openmp(graph_);
        womp.stop();
        const double t_omp = womp.elapsed_ms();
        std::printf("[compute] PageRank OMP finished: %.3f ms (iters=%d)\n", t_omp, pr_omp.iterations);

        double t_cuda = 0.0;
        PageRankResult pr_cuda;
        bool cuda_ok = false;
        if (ran_cuda) {
            std::printf("[compute] PageRank starting backend=CUDA ...\n");
            WallTimer wc;
            wc.start();
            pr_cuda = pagerank_cuda(graph_);
            wc.stop();
            t_cuda = wc.elapsed_ms();
            cuda_ok = true;
            std::printf("[compute] PageRank CUDA finished: %.3f ms (iters=%d)\n", t_cuda,
                        pr_cuda.iterations);
        }

        print_speedup_table("[compute] PageRank speedup summary (SEQ baseline)", t_seq, t_omp, t_cuda,
                            cuda_ok);

        const PageRankResult* pr = &pr_seq;
        int iters = pr_seq.iterations;
        if (chosen == ExecutionMode::OMP) {
            pr = &pr_omp;
            iters = pr_omp.iterations;
        } else if (chosen == ExecutionMode::CUDA && cuda_ok) {
            pr = &pr_cuda;
            iters = pr_cuda.iterations;
        }

        std::vector<std::pair<NodeID, Weight>> scored;
        scored.reserve(graph_.num_nodes());
        for (NodeID v = 0; v < graph_.num_nodes(); ++v) {
            scored.push_back({v, pr->ranks[static_cast<std::size_t>(v)]});
        }

        const std::size_t kk = static_cast<std::size_t>(std::max(0, k));
        if (scored.size() > kk) {
            const auto cmp = [](const auto& a, const auto& b) { return a.second > b.second; };
            std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(kk),
                              scored.end(), cmp);
            scored.resize(kk);
            std::sort(scored.begin(), scored.end(), cmp);
        } else {
            std::sort(scored.begin(), scored.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
        }

        std::printf("\n[report] Top-%d influencers (backend=%s, iters=%d):\n", k, backend_label(chosen),
                    iters);
        for (std::size_t i = 0; i < scored.size(); ++i) {
            std::printf("  #%zu: node %u score=%.8f\n", i + 1u, static_cast<unsigned>(scored[i].first),
                        scored[i].second);
        }
        return scored;
    }

    WallTimer timer;
    std::printf("\n[compute] PageRank starting backend=%s\n", backend_label(chosen));
    timer.start();
    PageRankResult pr = dispatch_pagerank(chosen, graph_);
    timer.stop();
    std::printf("[compute] PageRank backend=%s total time: %.3f ms (power iterations=%d)\n",
                backend_label(chosen), timer.elapsed_ms(), pr.iterations);

    std::vector<std::pair<NodeID, Weight>> scored;
    scored.reserve(graph_.num_nodes());
    for (NodeID v = 0; v < graph_.num_nodes(); ++v) {
        scored.push_back({v, pr.ranks[static_cast<std::size_t>(v)]});
    }

    const std::size_t kk = static_cast<std::size_t>(std::max(0, k));
    if (scored.size() > kk) {
        const auto cmp = [](const auto& a, const auto& b) { return a.second > b.second; };
        std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(kk), scored.end(),
                          cmp);
        scored.resize(kk);
        std::sort(scored.begin(), scored.end(), cmp);
    } else {
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    std::printf("[report] Top-%d influencers:\n", k);
    for (std::size_t i = 0; i < scored.size(); ++i) {
        std::printf("  #%zu: node %u score=%.8f\n", i + 1u, static_cast<unsigned>(scored[i].first),
                    scored[i].second);
    }

    return scored;
}

std::vector<NodeID> SocialNetworkAnalytics::community_report() {
    const WorkloadProfile prof = make_profile_wcc(graph_);
    const ExecutionMode chosen = resolve_mode(AlgorithmType::WCC, prof);
    const bool ran_cuda = is_gpu_available();

    if (compare_backends_) {
        std::printf("\n[compute] WCC multi-backend comparison (quiet intermediate runs)\n");

        WallTimer wseq;
        std::printf("[compute] WCC starting backend=SEQ ...\n");
        wseq.start();
        WccResult w_seq = wcc_sequential(graph_, false);
        wseq.stop();
        const double t_seq = wseq.elapsed_ms();
        std::printf("[compute] WCC SEQ finished: %.3f ms\n", t_seq);

        WallTimer womp;
        std::printf("[compute] WCC starting backend=OMP ...\n");
        womp.start();
        WccResult w_omp = wcc_openmp(graph_, false);
        womp.stop();
        const double t_omp = womp.elapsed_ms();
        std::printf("[compute] WCC OMP finished: %.3f ms\n", t_omp);

        double t_cuda = 0.0;
        WccResult w_cuda;
        bool cuda_ok = false;
        if (ran_cuda) {
            std::printf("[compute] WCC starting backend=CUDA ...\n");
            WallTimer wc;
            wc.start();
            w_cuda = wcc_cuda(graph_, false);
            wc.stop();
            t_cuda = wc.elapsed_ms();
            cuda_ok = true;
            std::printf("[compute] WCC CUDA finished: %.3f ms\n", t_cuda);
        }

        print_speedup_table("[compute] WCC speedup summary (SEQ baseline)", t_seq, t_omp, t_cuda,
                            cuda_ok);

        const WccResult* wcc = &w_seq;
        if (chosen == ExecutionMode::OMP) {
            wcc = &w_omp;
        } else if (chosen == ExecutionMode::CUDA && cuda_ok) {
            wcc = &w_cuda;
        }

        const double bench_wall_ms = t_seq + t_omp + (cuda_ok ? t_cuda : 0.0);
        print_wcc_summary(*wcc, bench_wall_ms);
        std::printf("[report] Summary above uses scheduler-selected backend=%s (sum of timed runs=%.3f ms).\n",
                    backend_label(chosen), bench_wall_ms);
        return wcc->component_id;
    }

    WallTimer timer;
    std::printf("\n[compute] WCC starting backend=%s\n", backend_label(chosen));
    timer.start();
    WccResult wcc = dispatch_wcc(chosen, graph_, true);
    timer.stop();
    std::printf("[compute] WCC backend=%s total time: %.3f ms\n", backend_label(chosen),
                timer.elapsed_ms());

    print_wcc_summary(wcc, timer.elapsed_ms());
    return wcc.component_id;
}
