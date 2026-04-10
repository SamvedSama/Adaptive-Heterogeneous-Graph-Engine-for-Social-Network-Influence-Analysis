# Top-level build for the adaptive heterogeneous graph engine (C++17 + CUDA + OpenMP).
cmake_minimum_required(VERSION 3.18)

project(adaptive_graph_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(OpenMP REQUIRED)

set(ENGINE_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

set(ENGINE_SOURCES
    src/graph/graph.cpp
    src/graph/loader.cpp
    src/scheduler/scheduler.cpp
    src/analytics/analytics.cpp
    src/algorithms/bfs/bfs_seq.cpp
    src/algorithms/bfs/bfs_omp.cpp
    src/algorithms/pagerank/pagerank_seq.cpp
    src/algorithms/pagerank/pagerank_omp.cpp
    src/algorithms/wcc/wcc_seq.cpp
    src/algorithms/wcc/wcc_omp.cpp
)

set(ENGINE_CUDA_SOURCES
    src/algorithms/bfs/bfs_cuda.cu
    src/algorithms/pagerank/pagerank_cuda.cu
    src/algorithms/wcc/wcc_cuda.cu
)

add_executable(graph_engine
    src/main.cpp
    ${ENGINE_SOURCES}
    ${ENGINE_CUDA_SOURCES}
)

target_include_directories(graph_engine PRIVATE ${ENGINE_INCLUDES})

target_compile_definitions(graph_engine PRIVATE
    GRAPH_ENGINE_BUILD_CUDA
    GRAPH_ENGINE_WITH_CUDA
)

target_compile_options(graph_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-fopenmp>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/openmp>>
)

target_compile_options(graph_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

target_link_libraries(graph_engine PRIVATE
    OpenMP::OpenMP_CXX
)

if (CMAKE_CUDA_COMPILER)
    target_link_libraries(graph_engine PRIVATE CUDA::cudart)
endif()

add_executable(benchmark_runner
    benchmarks/benchmark_runner.cpp
    ${ENGINE_SOURCES}
    ${ENGINE_CUDA_SOURCES}
)

target_include_directories(benchmark_runner PRIVATE ${ENGINE_INCLUDES})

target_compile_definitions(benchmark_runner PRIVATE
    GRAPH_ENGINE_BUILD_CUDA
    GRAPH_ENGINE_WITH_CUDA
)

target_compile_options(benchmark_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-fopenmp>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/openmp>>
)

target_compile_options(benchmark_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

target_link_libraries(benchmark_runner PRIVATE
    OpenMP::OpenMP_CXX
)

if (CMAKE_CUDA_COMPILER)
    target_link_libraries(benchmark_runner PRIVATE CUDA::cudart)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    target_link_libraries(benchmark_runner PRIVATE stdc++fs)
endif()

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks/results)
