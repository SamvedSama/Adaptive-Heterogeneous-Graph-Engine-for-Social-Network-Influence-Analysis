/**
 * @file loader.cpp
 * @brief Load a whitespace-separated edge list into a remapped 0-based CSR Graph,
 *        with comment handling and optional directed/undirected detection from headers.
 */
#include "graph/graph.h"
#include "timer.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

    /**
     * @brief Trims leading/trailing ASCII whitespace in place.
     */
    void trim_inplace(std::string &s)
    {
        auto not_space = [](unsigned char c)
        { return !std::isspace(c); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    }

    /**
     * @brief Parses header comments for explicit directed/undirected hints.
     *        Returns std::nullopt if no directive is found.
     */
    bool parse_directed_hint(const std::string &line, bool *out_directed)
    {
        std::string lower = line;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        if (lower.find("directed") != std::string::npos && lower.find("undirected") == std::string::npos)
        {
            *out_directed = true;
            return true;
        }
        if (lower.find("undirected") != std::string::npos)
        {
            *out_directed = false;
            return true;
        }
        return false;
    }

} // namespace

/**
 * @brief Loads an edge list file into CSR, remapping raw IDs to contiguous [0, N).
 *        Comment lines start with '#'. If no header specifies direction, uses
 *        default_directed. For undirected graphs, each edge is stored in both directions.
 */
Graph load_edge_list(const std::string &filepath, bool default_directed)
{
    WallTimer timer;
    timer.start();

    std::ifstream in(filepath);
    if (!in)
    {
        std::fprintf(stderr, "load_edge_list: cannot open file: %s\n", filepath.c_str());
        std::exit(EXIT_FAILURE);
    }

    bool is_directed = default_directed;
    bool directed_from_file = false;

    std::vector<std::pair<NodeID, NodeID>> edges;
    edges.reserve(1024);

    std::string line;
    while (std::getline(in, line))
    {
        trim_inplace(line);
        if (line.empty())
        {
            continue;
        }
        if (line[0] == '#')
        {
            bool hint = false;
            if (parse_directed_hint(line, &hint))
            {
                is_directed = hint;
                directed_from_file = true;
            }
            continue;
        }

        std::istringstream iss(line);
        std::uint64_t a64 = 0;
        std::uint64_t b64 = 0;
        if (!(iss >> a64 >> b64))
        {
            std::fprintf(stderr, "load_edge_list: bad edge line in %s\n", filepath.c_str());
            std::exit(EXIT_FAILURE);
        }
        edges.push_back({static_cast<NodeID>(a64), static_cast<NodeID>(b64)});
    }

    if (edges.empty())
    {
        std::fprintf(stderr, "load_edge_list: no edges read from %s\n", filepath.c_str());
        std::exit(EXIT_FAILURE);
    }

    // Remap raw node IDs to 0..N-1
    std::unordered_map<NodeID, NodeID> remap;
    remap.reserve(edges.size() * 2);
    NodeID next_id = 0;

    auto map_id = [&](NodeID raw) -> NodeID
    {
        auto it = remap.find(raw);
        if (it != remap.end())
        {
            return it->second;
        }
        const NodeID id = next_id++;
        remap.emplace(raw, id);
        return id;
    };

    std::vector<std::pair<NodeID, NodeID>> canon;
    canon.reserve(edges.size() * 2);
    for (const auto &e : edges)
    {
        const NodeID u = map_id(e.first);
        const NodeID v = map_id(e.second);
        if (u == v)
        {
            continue; // drop self-loops at load
        }
        canon.push_back({u, v});
        if (!is_directed)
        {
            canon.push_back({v, u});
        }
    }

    const NodeID n = next_id;
    if (n == 0)
    {
        std::fprintf(stderr, "load_edge_list: zero nodes after remap\n");
        std::exit(EXIT_FAILURE);
    }

    // Sort and unique for stable CSR build (parallel duplicates possible for undirected)
    std::sort(canon.begin(), canon.end());
    canon.erase(std::unique(canon.begin(), canon.end()), canon.end());

    const EdgeID m = static_cast<EdgeID>(canon.size());
    CSR csr;
    csr.num_nodes = n;
    csr.num_edges = m;
    csr.row_ptr.assign(static_cast<std::size_t>(n) + 1u, 0);
    csr.col_idx.resize(static_cast<std::size_t>(m));
    csr.values.resize(static_cast<std::size_t>(m), 1.0f);

    for (const auto &e : canon)
    {
        ++csr.row_ptr[e.first];
    }
    for (NodeID v = 0; v < n; ++v)
    {
        csr.row_ptr[v + 1] += csr.row_ptr[v];
    }

    std::vector<EdgeID> next = csr.row_ptr;
    for (const auto &e : canon)
    {
        const EdgeID pos = next[e.first]++;
        csr.col_idx[static_cast<std::size_t>(pos)] = e.second;
    }

    timer.stop();
    std::printf("Loaded graph: %s\n", filepath.c_str());
    std::printf("  Nodes: %u  Edges: %llu  Directed: %s (%s)\n",
                static_cast<unsigned>(n),
                static_cast<unsigned long long>(m),
                is_directed ? "yes" : "no",
                directed_from_file ? "from header" : "default");
    timer.stop();
    std::printf("  Load time: %.3f ms\n", timer.elapsed_ms());

    return Graph(std::move(csr), is_directed);
}
