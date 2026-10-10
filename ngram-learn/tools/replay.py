"""Replays a worker trace (TF_WORKER_TRACE): every state-setting request ("base") in order, then the last `--last`
requests, each echoed with its outcome. With `--sanitize` the worker runs under compute-sanitizer.

    uv run python tools/replay.py logs/trace.jsonl --last 1 --sanitize
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import Worker  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--last", type=int, default=1)
    ap.add_argument("--sanitize", action="store_true")
    a = ap.parse_args()
    reqs = [json.loads(line) for line in a.trace.read_bytes().splitlines() if line.strip()]
    tail = reqs[-a.last:]
    head = [r for r in reqs[:-a.last] if r.get("op") == "base"]
    if a.sanitize:
        os.environ["TF_SANITIZE"] = "1"
    os.environ.pop("TF_WORKER_TRACE", None)
    with Worker() as w:
        for r in head + tail:
            try:
                out = w.send(r)
                print(r.get("op", "case"), "ok", {k: v for k, v in out.items() if k in ("loss", "ok", "base_rows")})
            except RuntimeError as e:
                print(r.get("op", "case"), "FAILED", e)
                raise


if __name__ == "__main__":
    main()
