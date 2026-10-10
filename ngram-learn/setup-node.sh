#!/usr/bin/env bash
# Make another Spark an ngram-learn node, run from spark4: ./setup-node.sh spark2 [spark3 ...]
# Copies ~/ngram-learn (fork, build, tools, CounterFact, row counts; not the background corpus or caches), builds the
# dev image there (on MiaAI-Lab's tensorfold-qwen38:zig-db28187, which each node already has) and the tools' venv.
# The checkpoint is expected at $HF_DIR (default ~/qwen38/hf) on the node, as on spark4.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
one() {
  local h=$1
  ssh -o BatchMode=yes "$h" 'test -d ${HF_DIR:-~/qwen38/hf}/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/1464274120d36a4d8fcaa934552334a7d83ce0fd' ||
    { echo "$h: no checkpoint under ${HF_DIR:-~/qwen38/hf}"; return 1; }
  rsync -a --delete --exclude data/bg --exclude .cache --exclude tools/.venv --exclude logs --exclude .home \
    "$here/" "$h:ngram-learn/"
  ssh -o BatchMode=yes "$h" 'set -e; cd ~/ngram-learn; mkdir -p logs .home .cache/zig .cache/nv .cache/uv
    docker image inspect tensorfold-qwen38:zig-db28187 >/dev/null
    docker build -q -t ngram-learn-dev:latest -f docker/Dockerfile.dev docker >/dev/null
    cd tools && PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=~/ngram-learn/.cache/uv uv sync -q
    echo "$(hostname): ready"'
}
for h in "$@"; do one "$h" > "$here/logs/setup-$h.log" 2>&1 & done
wait
for h in "$@"; do tail -1 "$here/logs/setup-$h.log"; done
