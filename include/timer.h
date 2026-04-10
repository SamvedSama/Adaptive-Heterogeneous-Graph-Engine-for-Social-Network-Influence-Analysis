/**
 * @file timer.h
 * @brief Wall-clock RAII timer and optional CUDA event-based GPU timers for
 *        benchmarking host and device phases.
 */
#pragma once

#include <chrono>
#include <cstdint>

#ifdef GRAPH_ENGINE_BUILD_CUDA
#include <cuda_runtime.h>
#include "common.h"
#endif

/**
 * @brief High-resolution wall-clock timer using std::chrono steady_clock.
 */
class WallTimer {
public:
    WallTimer() = default;

    /** Records the start instant; call before measured work. */
    void start() { start_ = std::chrono::steady_clock::now(); }

    /** Records the stop instant; call after measured work. */
    void stop() { end_ = std::chrono::steady_clock::now(); }

    /** Elapsed milliseconds between the last start() and stop(). */
    double elapsed_ms() const {
        using Ms = std::chrono::duration<double, std::milli>;
        return std::chrono::duration_cast<Ms>(end_ - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_{};
    std::chrono::steady_clock::time_point end_{};
};

#ifdef GRAPH_ENGINE_BUILD_CUDA
/**
 * @brief Measures GPU time between CUDA events (records on the default stream).
 */
class CudaEventTimer {
public:
    CudaEventTimer() {
        checkCuda(cudaEventCreate(&start_ev_));
        checkCuda(cudaEventCreate(&stop_ev_));
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start_ev_);
        cudaEventDestroy(stop_ev_);
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start() { checkCuda(cudaEventRecord(start_ev_)); }

    void stop() { checkCuda(cudaEventRecord(stop_ev_)); }

    double elapsed_ms() const {
        checkCuda(cudaEventSynchronize(stop_ev_));
        float ms = 0.f;
        checkCuda(cudaEventElapsedTime(&ms, start_ev_, stop_ev_));
        return static_cast<double>(ms);
    }

private:
    cudaEvent_t start_ev_{};
    cudaEvent_t stop_ev_{};
};
#endif
