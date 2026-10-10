#!/usr/bin/env bash
# Run a `tensorfold` CLI command (BUILD=dev: the binary in build/dev; the AOT kernel set always from build/opt) (probe, score, ...) on the INT4-AutoRound checkpoint in the dev container.
# TF_SANITIZE=1 runs it under compute-sanitizer (memcheck); CUDA_LAUNCH_BLOCKING passes through.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cmd=$1; shift
MODEL=/hf/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/1464274120d36a4d8fcaa934552334a7d83ce0fd
wrap=()
if [ "${TF_SANITIZE:-0}" = 1 ]; then wrap=(compute-sanitizer --tool "${TF_SANITIZE_TOOL:-memcheck}" --print-limit 20 --show-backtrace device); fi
exec "$here/dev.sh" env HF_HUB_OFFLINE=1 CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0} TF_TRAIN_CHECK=${TF_TRAIN_CHECK:-0} TENSORFOLD_CUDA_KERNELS=/work/build/opt/share/tensorfold/cuda/sm121 \
  "${wrap[@]}" /work/build/${BUILD:-opt}/bin/tensorfold "$cmd" "$MODEL" --kernels /work/build/opt/share/tensorfold/cuda/sm121 "$@"
