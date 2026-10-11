"""Learning while serving: chat requests at random intervals (streamed, time to first token and total time), with
learnd learning in the background or not; prints the latency distribution and how many facts learned meanwhile.

    uv run latency_probe.py --engine http://127.0.0.1:8890 --minutes 20 --out ../experiments/07-scale/latency-off.json
    uv run latency_probe.py ... --learnd http://127.0.0.1:8091 --facts facts_suite.json --out ...-on.json
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

PROMPTS = ["Explain how a refrigerator keeps food cold.", "Write a haiku about autumn.", "What is the capital of Peru?",
           "Summarise the causes of the First World War in three sentences.", "Write a Python function that checks for a palindrome.",
           "Why is the sky blue?", "Give me three tips for a job interview.", "What does HTTP status 418 mean?"]


def stream(engine: str, prompt: str, max_tokens: int) -> tuple[float, float, int]:
    body = {"model": "flashnext", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": True}
    req = urllib.request.Request(engine + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                n += 1
                if first is None:
                    first = time.time() - t0
    return (first if first is not None else time.time() - t0), time.time() - t0, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8890")
    ap.add_argument("--learnd", help="learn the --facts meanwhile through this learnd")
    ap.add_argument("--facts", type=Path, default=Path(__file__).resolve().parent / "facts_suite.json")
    ap.add_argument("--tag", default="latency")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--gap", type=float, default=10.0, help="mean seconds between requests (exponential)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    learned: list[dict] = []
    stop = threading.Event()

    def learner() -> None:
        facts = json.loads(a.facts.read_text())
        for f in facts:
            if stop.is_set():
                return
            body = {"facts": [{"text": f"{f['train'][0][1]}", "subject": f["subject"],
                               "train": f["train"], "held": f["held"]}], "tag": a.tag}
            req = urllib.request.Request(a.learnd + "/v1/slide/learn", json.dumps(body).encode(), {"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=7200) as r:
                events = [ln.decode() for ln in r if ln.startswith(b"data: {")]
            done = [json.loads(e[6:]) for e in events if '"recalled"' in e]
            learned.append({"subject": f["subject"], "seconds": round(time.time() - t0, 1),
                            "recalled": bool(done and done[-1].get("recalled"))})

    th = threading.Thread(target=learner, daemon=True) if a.learnd else None
    if th:
        th.start()
    rng = random.Random(0)
    lat = []
    end = time.time() + a.minutes * 60
    while time.time() < end:
        time.sleep(rng.expovariate(1.0 / a.gap))
        ttft, total, n = stream(a.engine, rng.choice(PROMPTS), a.max_tokens)
        lat.append({"ttft": round(ttft, 3), "total": round(total, 3), "tokens": n})
    stop.set()
    t = np.array([x["ttft"] for x in lat])
    summary = {"requests": len(lat), "learning": bool(a.learnd), "ttft_p50": round(float(np.percentile(t, 50)), 3),
               "ttft_p90": round(float(np.percentile(t, 90)), 3), "ttft_max": round(float(t.max()), 3),
               "total_p50": round(float(np.percentile([x["total"] for x in lat], 50)), 3),
               "facts_learned": sum(x["recalled"] for x in learned), "facts_tried": len(learned)}
    print(json.dumps(summary, indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "requests": lat, "learned": learned}, indent=1))


if __name__ == "__main__":
    main()
