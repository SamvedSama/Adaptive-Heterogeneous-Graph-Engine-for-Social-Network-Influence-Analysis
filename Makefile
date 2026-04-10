# Root Makefile — orchestrates download, preprocess, build, run, benchmark, and plots.
# Works on Linux, macOS, WSL, and Google Colab. On native Windows without GNU make,
# run the same commands manually or use WSL.

SHELL := /bin/bash
BUILD_DIR := build
CMAKE ?= cmake
NPROC ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Default graph for demos (preprocess SNAP facebook_combined.txt first).
GRAPH ?= data/processed/facebook_combined.txt
DATASET ?= facebook
TRIALS ?= 3

.PHONY: all help download preprocess build run run-compare benchmark visualize report clean

help:
	@echo "Targets:"
	@echo "  make download     - Fetch SNAP .gz edge lists into data/ (bash + curl/wget)"
	@echo "  make preprocess   - Clean/remap SNAP text to data/processed/*.txt"
	@echo "  make build        - CMake configure + build graph_engine + benchmark_runner"
	@echo "  make run          - Run graph_engine on GRAPH with auto scheduling"
	@echo "  make run-compare  - Run graph_engine with --compare (SEQ/OMP/CUDA + speedup)"
	@echo "  make benchmark    - Run benchmark_runner; prints per-trial [compute] lines + median speedup"
	@echo "  make visualize    - Plot benchmarks/results/results.csv"
	@echo "  make report       - Optional: python analytics_report (needs --log or --exe)"
	@echo "  make all          - build + run"
	@echo ""
	@echo "Variables: GRAPH=$(GRAPH) DATASET=$(DATASET) TRIALS=$(TRIALS)"

all: build run

download:
	bash data/download.sh

preprocess:
	python3 python/preprocess.py data/facebook_combined.txt -o data/processed/facebook_combined.txt
	@echo "(Add similar lines for twitter/gplus after download if needed.)"

build:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(CMAKE) .. && $(CMAKE) --build . -j$(NPROC)

run: build
	cd $(BUILD_DIR) && ./graph_engine --graph ../$(GRAPH) --algorithm all --source 0 --topk 10 --mode auto

run-compare: build
	cd $(BUILD_DIR) && ./graph_engine --graph ../$(GRAPH) --algorithm all --source 0 --topk 10 --mode auto --compare

benchmark: build
	cd $(BUILD_DIR) && ./benchmark_runner --graph ../$(GRAPH) --dataset $(DATASET) --trials $(TRIALS)

visualize:
	python3 python/visualize.py --csv benchmarks/results/results.csv --outdir benchmarks/results

report:
	@echo "Example: python3 python/analytics_report.py --log engine.log"
	@echo "Or:      python3 python/analytics_report.py --exe build/graph_engine --graph $(GRAPH)"

clean:
	rm -rf $(BUILD_DIR)
