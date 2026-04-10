# Data directory

Place SNAP ego-network edge lists here, or run `./download.sh` from this folder on Linux/macOS (or WSL) to fetch:

- `facebook_combined.txt`
- `twitter_combined.txt`
- `gplus_combined.txt`

Use `python/preprocess.py` to produce cleaned, 0-indexed edge lists under `data/processed/` for the C++ binaries.
