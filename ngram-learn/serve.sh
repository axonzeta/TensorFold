#!/usr/bin/env bash
# Serve our fork's build (build/opt) on the INT4-AutoRound checkpoint with MiaAI-Lab's settings, text only.
# ./serve.sh [extra serve args]   env: PORT (8890), CONTEXT (32768), PARALLEL (1; 2 with LEARN=1), KV_DTYPE (fp8), DRAFTS (1), NAME,
#   BUILD (opt: the binary in build/$BUILD), LEARN (0; 1: learn ops on POST /v1/ngram/op and overlays by metadata.tag,
#   with RESERVE_GIB of memory kept out of the sequences' budget for the training pass), TF_NGRAM_OVERLAY
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8890}
CONTEXT=${CONTEXT:-32768}
# with LEARN=1, two streams: a request runs beside learnd's own (background) chats instead of behind them
PARALLEL=${PARALLEL:-$([[ "${LEARN:-0}" == 1 ]] && echo 2 || echo 1)}
KV_DTYPE=${KV_DTYPE:-fp8}
NAME=${NAME:-ngram-learn-serve}
LEARN=${LEARN:-0}
# the training pass holds ~14 GiB (scratch, dequantised experts and head) beside the serving engine
RESERVE_GIB=${RESERVE_GIB:-$([[ "$LEARN" == 1 ]] && echo 26 || echo 10)}
MODEL=/hf/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/1464274120d36a4d8fcaa934552334a7d83ce0fd
args=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens 4096 --temperature 0 --no-thinking --kv-dtype "$KV_DTYPE")
[[ "${DRAFTS:-1}" == 1 ]] || args+=(--no-drafts)
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run -d --name "$NAME" --gpus all --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e HOME=/work/.home -v "$here:/work" -v "${HF_DIR:-$HOME/qwen38/hf}:/hf:ro" \
  -e HF_HUB_OFFLINE=1 -e TENSORFOLD_CUDA_KERNELS=/work/build/opt/share/tensorfold/cuda/sm121 \
  -e CUDA_CACHE_PATH=/work/.cache/nv -e TF_FLASHNEXT_DEPTH=15 -e TF_FLASHNEXT_PRODUCT_STREAMS=2 \
  -e TF_FLASHNEXT_PREFILL_TAIL=512 -e TF_FLASHNEXT_YARN=0 -e TENSORFOLD_MEMORY_RESERVE_GIB="$RESERVE_GIB" -e TF_NGRAM_LEARN="$LEARN" \
  -e TENSORFOLD_NO_UPDATE_CHECK=1 ${TF_NGRAM_OVERLAY:+-e TF_NGRAM_OVERLAY=$TF_NGRAM_OVERLAY} \
  ngram-learn-dev:latest /work/build/${BUILD:-opt}/native/bin/tensorfold-native serve "$MODEL" --name flashnext \
  --host 127.0.0.1 --port "$PORT" "${args[@]}" "$@"
