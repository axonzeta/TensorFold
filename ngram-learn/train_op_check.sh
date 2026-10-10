#!/usr/bin/env bash
# Phase 3 op checks: generate inputs, run them in the engine (no weights loaded), compare against PyTorch.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=/work/experiments/03-backward/ops
"$here/dev.sh" bash -lc "cd /work/tools && python train_ops_check.py gen $dir" | tail -1
"$here/probe.sh" train-op-check "$dir" 2>&1 | grep -E "^op |error|Error|unknown"
"$here/dev.sh" bash -lc "cd /work/tools && python train_ops_check.py cmp $dir" | grep -v -E "^\s*$|Copyright|NVIDIA|PyTorch|governed|found at|reserved|Various|Container|cuda-compat|CUDA Forward|Using CUDA|^=+|_pytree"
