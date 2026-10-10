"""Phase 0 baseline: decode speed and exactness (drafted == "draft": false) against a running server.

    uv run baseline.py --url http://127.0.0.1:8890 --out ../experiments/00-baseline/run.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "Write a Python hash map with open addressing and linear probing, with insert, get and delete.",
    "Explain how a gated delta rule recurrent layer differs from softmax attention, in about 300 words.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List the planets of the solar system with one fact about each.",
    "Rename every occurrence of `user_id` to `account_id` in this code and return the whole file:\n\n"
    "def load(user_id):\n    row = db.get(user_id)\n    if row is None:\n        raise KeyError(user_id)\n"
    "    return {'user_id': user_id, 'name': row.name}\n\ndef save(user_id, name):\n    db.put(user_id, name)\n",
    "What is the capital of Australia, and why is it not Sydney?",
]


def chat(url: str, prompt: str, draft: bool, max_tokens: int) -> tuple[str, dict, float]:
    body = {"model": "flashnext", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "draft": draft}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"], out.get("usage", {}), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8890")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows, same = [], 0
    chat(args.url, "Say hello.", True, 16)  # warm
    for p in PROMPTS:
        r = {}
        for draft in (True, False):
            text, usage, dt = chat(args.url, p, draft, args.max_tokens)
            n = usage.get("completion_tokens", 0)
            r["drafted" if draft else "plain"] = {"tokens": n, "seconds": round(dt, 3), "tok_s": round(n / dt, 1),
                                                   "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}
        r["equal"] = r["drafted"]["sha"] == r["plain"]["sha"]
        same += r["equal"]
        r["prompt"] = p[:60]
        rows.append(r)
        print(f"{r['prompt']:60} drafted {r['drafted']['tok_s']:6.1f} tok/s  plain {r['plain']['tok_s']:6.1f}  equal {r['equal']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"equal": f"{same}/{len(PROMPTS)}", "rows": rows}, indent=1))
    print(f"exact: {same}/{len(PROMPTS)}")


if __name__ == "__main__":
    main()
