/**
 * @file wcc_cuda.cu
 * @brief Weakly connected components on the GPU using a Shiloach–Vishkin style
 *        hook–shortcut alternation with atomicMin on a parent array (one thread per edge on hook).
 */
#include "graph/graph.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

/**
 * @brief Pointer-jumping step: parent[i] = parent[parent[i]].
 */
__global__ void sv_shortcut_kernel(unsigned n, unsigned* __restrict__ parent, int* __restrict__ changed) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const unsigned p = parent[i];
    const unsigned gp = parent[p];
    if (gp != p) {
        parent[i] = gp;
        atomicOr(changed, 1);
    }
}

/**
 * @brief Hooking step: for each undirected link (u,v), link the larger root under the smaller.
 */
__global__ void sv_hook_kernel(unsigned E,
                               const unsigned* __restrict__ eu,
                               const unsigned* __restrict__ ev,
                               unsigned* __restrict__ parent,
                               int* __restrict__ changed) {
    const unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    }
    const unsigned u = eu[e];
    const unsigned v = ev[e];
    unsigned fu = parent[u];
    unsigned fv = parent[v];
    if (fu == fv) {
        return;
    }
    const unsigned small = (fu < fv) ? fu : fv;
    const unsigned large = (fu < fv) ? fv : fu;
    const unsigned old = atomicMin(&parent[large], small);
    if (old > small) {
        atomicOr(changed, 1);
    }
}

} // namespace

/**
 * @brief Copies CSR edges to the device and alternates hook/shortcut until quiescent.
 */
WccResult wcc_cuda(const Graph& graph, bool verbose) {
    const NodeID n = graph.num_nodes();
    WccResult out;
    out.component_id.assign(static_cast<std::size_t>(n), 0);
    out.component_sizes.clear();

    if (n == 0) {
        if (verbose) {
            std::printf("WCC (CUDA): 0 components (empty graph)\n");
        }
        return out;
    }

    const CSR& csr = graph.csr();
    std::vector<unsigned> eu;
    std::vector<unsigned> ev;
    eu.reserve(static_cast<std::size_t>(csr.num_edges));
    ev.reserve(static_cast<std::size_t>(csr.num_edges));

    for (NodeID v = 0; v < n; ++v) {
        const EdgeID lo = csr.row_ptr[v];
        const EdgeID hi = csr.row_ptr[v + 1];
        for (EdgeID e = lo; e < hi; ++e) {
            const NodeID u = csr.col_idx[static_cast<std::size_t>(e)];
            eu.push_back(static_cast<unsigned>(v));
            ev.push_back(static_cast<unsigned>(u));
        }
    }

    const unsigned E = static_cast<unsigned>(eu.size());
    if (E == 0) {
        for (NodeID v = 0; v < n; ++v) {
            out.component_id[v] = v;
        }
        out.component_sizes.push_back({0, static_cast<int>(n)});
        if (verbose) {
            std::printf("WCC (CUDA): %u isolated vertices\n", static_cast<unsigned>(n));
        }
        return out;
    }

    unsigned* d_parent = nullptr;
    unsigned* d_eu = nullptr;
    unsigned* d_ev = nullptr;
    int* d_changed = nullptr;

    checkCuda(cudaMalloc(&d_parent, static_cast<std::size_t>(n) * sizeof(unsigned)));
    checkCuda(cudaMalloc(&d_eu, static_cast<std::size_t>(E) * sizeof(unsigned)));
    checkCuda(cudaMalloc(&d_ev, static_cast<std::size_t>(E) * sizeof(unsigned)));
    checkCuda(cudaMalloc(&d_changed, sizeof(int)));

    std::vector<unsigned> host_parent(static_cast<std::size_t>(n));
    for (NodeID v = 0; v < n; ++v) {
        host_parent[v] = static_cast<unsigned>(v);
    }

    checkCuda(cudaMemcpy(d_parent, host_parent.data(), host_parent.size() * sizeof(unsigned),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_eu, eu.data(), static_cast<std::size_t>(E) * sizeof(unsigned),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_ev, ev.data(), static_cast<std::size_t>(E) * sizeof(unsigned),
                         cudaMemcpyHostToDevice));

    const int threads = kDefaultCudaBlockSize;
    const int blocks_n = static_cast<int>((static_cast<unsigned>(n) + threads - 1) / threads);
    const int blocks_e = static_cast<int>((E + static_cast<unsigned>(threads) - 1U) / threads);

    const int max_rounds = std::max(64, static_cast<int>(std::log2(static_cast<double>(n + 2)) * 32 + 32));

    for (int round = 0; round < max_rounds; ++round) {
        int zero = 0;
        checkCuda(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice));

        sv_hook_kernel<<<blocks_e, threads>>>(E, d_eu, d_ev, d_parent, d_changed);
        checkCuda(cudaGetLastError());

        sv_shortcut_kernel<<<blocks_n, threads>>>(static_cast<unsigned>(n), d_parent, d_changed);
        checkCuda(cudaGetLastError());

        sv_shortcut_kernel<<<blocks_n, threads>>>(static_cast<unsigned>(n), d_parent, d_changed);
        checkCuda(cudaGetLastError());

        checkCuda(cudaDeviceSynchronize());

        int host_changed = 0;
        checkCuda(cudaMemcpy(&host_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (host_changed == 0) {
            break;
        }
    }

    checkCuda(cudaMemcpy(host_parent.data(), d_parent, host_parent.size() * sizeof(unsigned),
                         cudaMemcpyDeviceToHost));

    // Finalize labels on the host with repeated pointer jumping for correctness
    for (int pass = 0; pass < static_cast<int>(n) + 8; ++pass) {
        bool stable = true;
        for (NodeID v = 0; v < n; ++v) {
            const unsigned p = host_parent[v];
            const unsigned gp = host_parent[p];
            if (gp != p) {
                host_parent[v] = gp;
                stable = false;
            }
        }
        if (stable) {
            break;
        }
    }

    for (NodeID v = 0; v < n; ++v) {
        unsigned cur = host_parent[v];
        while (cur != host_parent[cur]) {
            cur = host_parent[cur];
        }
        host_parent[v] = cur;
        out.component_id[v] = static_cast<NodeID>(cur);
    }

    std::vector<NodeID> roots = out.component_id;
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
        std::printf("WCC (CUDA): components=%d largest=%d nodes\n", num_comp, largest);
        std::printf("  Size distribution (top 8): ");
        const int kShow = std::min(8, static_cast<int>(out.component_sizes.size()));
        for (int t = 0; t < kShow; ++t) {
            std::printf("%u:%d ",
                        static_cast<unsigned>(out.component_sizes[static_cast<std::size_t>(t)].first),
                        out.component_sizes[static_cast<std::size_t>(t)].second);
        }
        std::printf("\n");
    }

    checkCuda(cudaFree(d_parent));
    checkCuda(cudaFree(d_eu));
    checkCuda(cudaFree(d_ev));
    checkCuda(cudaFree(d_changed));

    return out;
}
