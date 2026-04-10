/**
 * @file scheduler.h
 * @brief Workload profiling snapshot and heuristic selection among SEQ, OMP, and CUDA
 *        backends based on graph scale and algorithm family.
 */
#pragma once

#include "common.h"

#include <string>

/**
 * @brief Lightweight graph telemetry used by the adaptive execution policy.
 */
struct WorkloadProfile {
    NodeID num_nodes = 0;
    EdgeID num_edges = 0;
    NodeID frontier_size = 0;
    float active_ratio = 0.0f;
    float avg_degree = 0.0f;
    bool is_directed = true;
};

/**
 * @brief Returns true when the CUDA runtime reports at least one usable device.
 */
bool is_gpu_available();

/**
 * @brief Chooses an execution mode using fixed thresholds (see scheduler.cpp).
 *        Emits a human-readable rationale to stdout.
 */
ExecutionMode decide(AlgorithmType algo, const WorkloadProfile& profile);

/**
 * @brief Object-oriented entry point used by the analytics façade (wraps decide()).
 */
class Scheduler {
public:
    /** @brief Delegates to the global CUDA probe. */
    bool gpu_available() const { return is_gpu_available(); }

    /** @brief Runs the documented heuristic policy for the given algorithm. */
    ExecutionMode choose(AlgorithmType algo, const WorkloadProfile& profile) const {
        return decide(algo, profile);
    }
};
