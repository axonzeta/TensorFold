#!/usr/bin/env bash
# B10: one lesson through the in-engine training pass and the PyTorch reference, compared. ./train_grad_check.sh [case_id]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=/work/experiments/03-backward/grad
filter='^\s*$|Copyright|NVIDIA|PyTorch|governed|found at|reserved|Various|Container|cuda-compat|CUDA Forward|Using CUDA|^=+|_pytree|FutureWarning|rotary_emb|falling back'
"$here/dev.sh" bash -lc "cd /work/tools && python train_grad_check.py case $dir ${1:-0}" | grep -v -E "$filter"
"$here/probe.sh" train-grad-check "$dir" --eager --context 4096 --kv-dtype bf16 2>&1 | grep -E "^loss|rror|panic" || true
"$here/dev.sh" bash -lc "cd /work/tools && python train_grad_check.py ref $dir" 2>&1 | grep -v -E "$filter" | tail -3
"$here/dev.sh" bash -lc "cd /work/tools && python train_grad_check.py cmp $dir" 2>&1 | grep -v -E "$filter"
