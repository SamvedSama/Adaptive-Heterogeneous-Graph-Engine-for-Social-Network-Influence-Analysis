/**
 * @file benchmark_runner.cpp
 * @brief Standalone harness that times every algorithm across SEQ, OMP, and CUDA backends,
 *        appends CSV rows, logs each run, and prints median speedup tables per algorithm.
 */
#include "analytics/analytics.h"
#include "graph/graph.h"
#include "scheduler/scheduler.h"
#include "timer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::string graph_path;
    std::string dataset_name = "graph";
    int trials = 3;
    // Explicit output CSV path. When empty, falls back to a path relative to
    // the executable so that both the runner and visualize.py agree on location.
    std::string out_csv;
};

static bool streq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (streq(argv[i], "--graph") && i + 1 < argc) {
            a.graph_path = argv[++i];
        } else if (streq(argv[i], "--dataset") && i + 1 < argc) {
            a.dataset_name = argv[++i];
        } else if (streq(argv[i], "--trials") && i + 1 < argc) {
            a.trials = std::max(1, std::atoi(argv[++i]));
        } else if (streq(argv[i], "--outcsv") && i + 1 < argc) {
            a.out_csv = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            std::fprintf(stderr,
                "Usage: benchmark_runner --graph <path> "
                "[--dataset name] [--trials N] [--outcsv path/to/results.csv]\n");
            std::exit(EXIT_FAILURE);
        }
    }
    if (a.graph_path.empty()) {
        std::fprintf(stderr, "benchmark_runner: --graph is required\n");
        std::exit(EXIT_FAILURE);
    }
    return a;
}

/**
 * @brief Resolve final CSV path.
 *
 * Priority:
 *   1. --outcsv argument (explicit, always wins)
 *   2. Fallback: ../benchmarks/results/results.csv relative to CWD
 *      (correct when running from build/ inside the repo tree)
 *
 * The parent directory is created if it does not exist.
 */
static std::string resolve_csv_path(const std::string& cli_override) {
    namespace fs = std::filesystem;
    std::string path = cli_override.empty()
        ? "../benchmarks/results/results.csv"
        : cli_override;

    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    if (ec) {
        std::fprintf(stderr,
            "benchmark_runner: warning: could not create CSV directory: %s\n",
            ec.message().c_str());
    }
    return path;
}

static void append_csv_row(const std::string& csv_path,
                           bool* header_exists,
                           const std::string& algorithm,
                           const std::string& mode,
                           const std::string& dataset,
                           NodeID num_nodes,
                           EdgeID num_edges,
                           double time_ms) {
    std::ofstream out(csv_path, std::ios::app);
    if (!out) {
        std::fprintf(stderr, "benchmark_runner: cannot open CSV: %s\n", csv_path.c_str());
        std::exit(EXIT_FAILURE);
    }
    if (!*header_exists) {
        out << "algorithm,mode,dataset,num_nodes,num_edges,time_ms\n";
        *header_exists = true;
    }
    out << algorithm << ',' << mode << ',' << dataset << ','
        << static_cast<unsigned long long>(num_nodes) << ','
        << static_cast<unsigned long long>(num_edges) << ','
        << time_ms << '\n';
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2u == 1u) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

using Key = std::pair<std::string, std::string>;

static void print_median_speedup_summary(
    const std::map<Key, std::vector<double>>& buckets,
    const char* algo,
    bool gpu_present)
{
    auto get_med = [&](const char* mode) -> double {
        const auto it = buckets.find(Key{algo, mode});
        if (it == buckets.end() || it->second.empty()) return -1.0;
        return median_of(it->second);
    };

    const double t_seq  = get_med("seq");
    const double t_omp  = get_med("omp");
    const double t_cuda = get_med("cuda");

    std::printf("\n[benchmark] %s — median time_ms across trials (speedup vs SEQ)\n", algo);
    std::printf("  %-8s  %10s  %16s\n", "backend", "median_ms", "speedup_vs_SEQ");

    if (t_seq >= 0.0)
        std::printf("  %-8s  %10.3f  %16.2fx\n", "SEQ", t_seq, 1.0);
    else
        std::printf("  %-8s  %10s  %16s\n", "SEQ", "n/a", "n/a");

    if (t_omp >= 0.0 && t_seq > 0.0)
        std::printf("  %-8s  %10.3f  %16.2fx\n", "OMP", t_omp, t_seq / t_omp);
    else if (t_omp >= 0.0)
        std::printf("  %-8s  %10.3f  %16s\n",    "OMP", t_omp, "n/a");
    else
        std::printf("  %-8s  %10s  %16s\n",      "OMP", "n/a", "n/a");

    if (gpu_present && t_cuda >= 0.0 && t_seq > 0.0)
        std::printf("  %-8s  %10.3f  %16.2fx\n", "CUDA", t_cuda, t_seq / t_cuda);
    else if (gpu_present && t_cuda >= 0.0)
        std::printf("  %-8s  %10.3f  %16s\n",    "CUDA", t_cuda, "n/a");
    else
        std::printf("  %-8s  %10s  %16s\n",      "CUDA", "-", gpu_present ? "n/a" : "no GPU");
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    Graph g = load_edge_list(args.graph_path, false);

    const std::string csv_path = resolve_csv_path(args.out_csv);

    // Check whether a valid header row already exists so we don't double-write it.
    bool header_exists = false;
    {
        std::ifstream probe(csv_path);
        std::string line;
        if (probe && std::getline(probe, line) && line.rfind("algorithm", 0) == 0)
            header_exists = true;
    }

    std::map<Key, std::vector<double>> timing_buckets;
    const bool gpu_present = is_gpu_available();

    using BenchFn = std::function<void()>;
    const std::vector<std::pair<std::string, BenchFn>> workloads = {
        {"bfs_seq",      [&] { volatile auto r = bfs_sequential(g, 0);    (void)r; }},
        {"bfs_omp",      [&] { volatile auto r = bfs_openmp(g, 0);        (void)r; }},
        {"bfs_cuda",     [&] { volatile auto r = bfs_cuda(g, 0);          (void)r; }},
        {"pagerank_seq", [&] { volatile auto r = pagerank_sequential(g);  (void)r; }},
        {"pagerank_omp", [&] { volatile auto r = pagerank_openmp(g);      (void)r; }},
        {"pagerank_cuda",[&] { volatile auto r = pagerank_cuda(g);        (void)r; }},
        {"wcc_seq",      [&] { volatile auto r = wcc_sequential(g, false);(void)r; }},
        {"wcc_omp",      [&] { volatile auto r = wcc_openmp(g, false);    (void)r; }},
        {"wcc_cuda",     [&] { volatile auto r = wcc_cuda(g, false);      (void)r; }},
    };

    for (const auto& job : workloads) {
        const std::string& key = job.first;
        const BenchFn& fn = job.second;

        std::string algo, mode;
        if      (key.rfind("bfs_",      0) == 0) { algo = "bfs";      mode = key.substr(4);  }
        else if (key.rfind("pagerank_", 0) == 0) { algo = "pagerank"; mode = key.substr(9);  }
        else if (key.rfind("wcc_",      0) == 0) { algo = "wcc";      mode = key.substr(4);  }
        else                                      { algo = key;        mode = "unknown";       }

        for (int t = 0; t < args.trials; ++t) {
            std::printf("[compute] %s trial %d/%d starting ...\n",
                        key.c_str(), t + 1, args.trials);
            WallTimer timer;
            timer.start();
            fn();
            timer.stop();
            const double ms = timer.elapsed_ms();
            std::printf("[compute] %s trial %d/%d finished: %.3f ms\n",
                        key.c_str(), t + 1, args.trials, ms);
            append_csv_row(csv_path, &header_exists,
                           algo, mode, args.dataset_name,
                           g.num_nodes(), g.num_edges(), ms);
            timing_buckets[Key{algo, mode}].push_back(ms);
        }
    }

    std::printf("\n[benchmark] === Median speedup summary (per algorithm) ===\n");
    print_median_speedup_summary(timing_buckets, "bfs",      gpu_present);
    print_median_speedup_summary(timing_buckets, "pagerank", gpu_present);
    print_median_speedup_summary(timing_buckets, "wcc",      gpu_present);

    std::printf("\nbenchmark_runner: wrote %s\n", csv_path.c_str());
    return EXIT_SUCCESS;
}
