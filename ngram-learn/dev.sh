#!/usr/bin/env bash
# Run a command in the ngram-learn dev container: ~/ngram-learn at /work, the HF cache read-only at /hf, the GPU.
# NAME=... to name the container (e.g. a server), DETACH=1 to run it in the background, PORT=... to publish a port.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
args=(--rm --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 -v "$here:/work" -v "$here/tensorfold:/tensorfold" -v "${HF_DIR:-$HOME/qwen38/hf}:/hf:ro" -w /tensorfold --user "$(id -u):$(id -g)" -e HOME=/work/.home)
[[ -n "${NAME:-}" ]] && args+=(--name "$NAME")
[[ -n "${PORT:-}" ]] && args+=(-p "$PORT:$PORT")
[[ "${DETACH:-0}" == 1 ]] && args+=(-d) || args+=(-i)
[[ -t 0 && "${DETACH:-0}" != 1 ]] && args+=(-t)
exec docker run "${args[@]}" ngram-learn-dev:latest "$@"
