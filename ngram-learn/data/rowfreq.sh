#!/usr/bin/env bash
# data/rowfreq/counts.npy: how often 2.52 B tokens of web, chat and code look up each n-gram table row
# (experiments/01b-row-frequency; ~6 GB download, ~41 min of CPU). Run from ngram-learn/ on the Spark.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here/tools"
uv run python ../data/fetch_bg.py
model=$(uv run python -c "import paths; print(paths.MODEL)")
uv run rowfreq.py --model "$model" \
  --source "web:$here/data/bg/fineweb/sample/10BT/*.parquet" \
  --source "chat:$here/data/bg/ultrachat/data/*.parquet" \
  --source "code:$here/data/bg/code/*.json.gz" \
  --out "$here/data/rowfreq/counts.npy"
