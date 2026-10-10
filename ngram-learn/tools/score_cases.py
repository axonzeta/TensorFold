"""Phase 1e causal tracing: cases for `tensorfold score`, and the analysis of its report.

Build: CounterFact facts as plain completions ("The mother tongue of Danielle Darrieux is" -> " French"), each under
five mute conditions: none, subject (every subject token), random (as many non-subject prompt positions), tail (as
many positions at the prompt's end) and all (every prompt position).

    uv run score_cases.py build --model DIR --data counterfact.parquet --limit 300 --out cases.json
    uv run score_cases.py analyse --cases cases.json --report report.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = ("none", "subject", "random", "tail", "all")


def build(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    rng = random.Random(0)
    cases, meta = [], []
    for c in pq.read_table(args.data).slice(0, args.limit).to_pylist():
        rw = c["requested_rewrite"]
        prompt = rw["prompt"].format(rw["subject"])
        enc = tok.encode(prompt, add_special_tokens=False)
        at = prompt.find(rw["subject"])
        end = at + len(rw["subject"])
        subj = [i for i, (a, b) in enumerate(enc.offsets) if b > at and a < end and b > a]
        cont = tok.encode(" " + rw["target_true"]["str"], add_special_tokens=False).ids
        n = len(enc.ids)
        others = [i for i in range(n) if i not in subj]
        if not subj or len(others) < len(subj):
            continue
        tail = [i for i in range(n - len(subj), n) if i not in subj]
        if len(tail) < len(subj):
            continue
        mutes = {"none": [], "subject": subj, "random": sorted(rng.sample(others, len(subj))), "tail": tail,
                 "all": list(range(n))}
        for cond in CONDITIONS:
            cases.append({"name": f"{c['case_id']}:{cond}", "prompt": enc.ids, "cont": cont, "mute": mutes[cond]})
        meta.append({"case_id": c["case_id"], "subject_tokens": len(subj), "prompt_tokens": n})
    args.out.write_text(json.dumps({"cases": cases, "meta": meta}))
    print(f"{len(meta)} facts, {len(cases)} cases -> {args.out}")


def analyse(args: argparse.Namespace) -> None:
    rep = json.loads(args.report.read_text())["cases"]
    by = defaultdict(dict)
    for r in rep:
        cid, cond = r["name"].split(":")
        by[cid][cond] = r
    known = {cid: d for cid, d in by.items() if "none" in d and d["none"]["greedy_hits"] == len(d["none"]["logprobs"])}
    print(f"{len(by)} facts scored; {len(known)} known (greedy reproduces the true answer)")
    for subset, facts in (("all", by), ("known", known)):
        out = {}
        for cond in CONDITIONS[1:]:
            d = [f[cond]["sum"] - f["none"]["sum"] for f in facts.values() if cond in f and "none" in f]
            first = [f[cond]["logprobs"][0] - f["none"]["logprobs"][0] for f in facts.values() if cond in f]
            kept = [f[cond]["greedy_hits"] == len(f[cond]["logprobs"]) for f in facts.values() if cond in f]
            out[cond] = {"mean_dlogp": round(float(np.mean(d)), 3), "median_dlogp": round(float(np.median(d)), 3),
                         "mean_dlogp_first": round(float(np.mean(first)), 3),
                         "still_greedy": round(float(np.mean(kept)), 3), "n": len(d)}
        print(subset, json.dumps(out, indent=1))
        if args.out:
            prev = json.loads(args.out.read_text()) if args.out.exists() else {}
            prev[subset] = out
            args.out.write_text(json.dumps(prev, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", type=Path, required=True)
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--limit", type=int, default=300)
    b.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--cases", type=Path)
    a.add_argument("--report", type=Path, required=True)
    a.add_argument("--out", type=Path)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else analyse(args)


if __name__ == "__main__":
    main()
