#!/usr/bin/env bash
# data/download.sh — fetch SNAP ego-network edge lists used in the semester project demos.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

fetch() {
  local url="$1"
  local out="$2"
  echo "Downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "error: need curl or wget" >&2
    exit 1
  fi
}

fetch "https://snap.stanford.edu/data/facebook_combined.txt.gz" "facebook_combined.txt.gz"
fetch "https://snap.stanford.edu/data/twitter_combined.txt.gz" "twitter_combined.txt.gz"
fetch "https://snap.stanford.edu/data/gplus_combined.txt.gz" "gplus_combined.txt.gz"

for gz in facebook_combined.txt.gz twitter_combined.txt.gz gplus_combined.txt.gz; do
  echo "Extracting $gz"
  gunzip -f "$gz"
done

echo "Done. Edge lists are in: $ROOT"
