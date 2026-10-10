#!/usr/bin/env bash
# Build the fork inside the dev container: ./dev.sh bash -lc "[AOT=0] [BUILD=dev] /work/build.sh". Output: /work/build/${BUILD:-opt}.
set -euo pipefail
cd /tensorfold
prefix=/work/build/${BUILD:-opt}
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=${OPT:-fast} --prefix "$prefix" -j"$(nproc)" fatbins install native
test -x "$prefix/native/bin/tensorfold-native"
if [[ "${AOT:-1}" == 1 ]]; then
  # Pin kernel source mtimes (Triton line info) so cubins match the reference kernel set, as MiaAI-Lab does.
  python -B -c "import json, os, sys; t = int(sys.argv[1]); [os.utime(os.path.join(\"src\", f), (t, t)) for f in sorted({k[\"source\"][\"file\"] for k in json.load(open(\"zig/tests/cuda/flashnext/kernels.json\"))[\"kernels\"]})]" 1791318675
  PYTHONPATH=/tensorfold/src python -B tools/zig/flashnext_aot.py build --spec zig/tests/cuda/flashnext/kernels.json \
    --jit zig/tests/cuda/flashnext/jit.json --tp 1 --out "$prefix/share/tensorfold/cuda/sm121" | tee /work/logs/aot.log
  grep -q "kernels ->" /work/logs/aot.log
  ! grep PROBLEM /work/logs/aot.log | grep -v "cubin sha256"
fi
echo BUILD-OK
