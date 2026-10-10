"""A client for `tensorfold worker`: the model loaded once, then cases one JSON line at a time.

    with Worker() as w:
        r = w.case(prompt=[...], cont=[...], overlay={"ids": [...], "rows": [[...]]})
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Worker:
    def __init__(self, extra: tuple[str, ...] = ("--eager", "--context", "4096", "--kv-dtype", "bf16", "--kv-gib", "4"),
                 log: Path | None = None):
        self.log = open(log or ROOT / "logs" / "worker.log", "ab")
        self.p = subprocess.Popen([str(ROOT / "probe.sh"), "worker", *extra], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        t0 = time.time()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("worker exited before it was ready (see logs/worker.log)")
            if line.startswith(b"{") and json.loads(line).get("ready"):
                break
        self.load_seconds = time.time() - t0
        # TF_WORKER_TRACE=file: every request appended, so a failing one can be replayed (tools/replay.py)
        trace = os.environ.get("TF_WORKER_TRACE")
        self.trace = open(trace, "ab") if trace else None

    def send(self, obj: dict) -> dict:
        line = json.dumps(obj, separators=(",", ":")).encode() + b"\n"
        if self.trace:
            self.trace.write(line)
            self.trace.flush()
        self.p.stdin.write(line)
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"worker exited with status {self.p.wait()} (see logs/worker.log)")
            if line.startswith(b"{"):
                r = json.loads(line)
                if "error" in r:
                    raise RuntimeError(f"worker: {r['error']}")
                return r

    def case(self, **kw) -> dict:
        return self.send(kw)

    def base(self, clear: bool = False, **kw) -> dict:
        return self.send({"op": "base", "clear": clear, **kw})

    def close(self) -> None:
        try:
            self.send_quit()
        finally:
            self.p.wait(timeout=60)
            self.log.close()

    def send_quit(self) -> None:
        self.p.stdin.write(b'{"op":"quit"}\n')
        self.p.stdin.flush()

    def __enter__(self) -> Worker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
