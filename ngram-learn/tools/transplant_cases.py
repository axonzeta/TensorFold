"""Phase 2 probe cases: (a) the Python decoder's rows as an overlay change nothing; (b) memory transplant.

(a) exact: a fact's prompt with every looked-up row overlaid by tools/ngram_table.py's decoding of the same row.
(b) transplant: facts X and Y share a relation with different true answers. X's prompt is scored for A_X and A_Y,
    with no overlay and with the rows at X's subject positions replaced by the rows at Y's subject positions (aligned
    from the subject's last token backwards, all 16 heads; "copy" pairs, so the engine reads Y's rows itself).

    uv run transplant_cases.py build --model DIR --data counterfact.parquet --pairs 150 --out cases.json
    uv run transplant_cases.py analyse --report report.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ngram_ids import NGram


def subject_tokens(enc, text: str, subject: str) -> list[int]:
    at = text.find(subject)
    end = at + len(subject)
    return [i for i, (a, b) in enumerate(enc.offsets) if b > at and a < end and b > a]


def build(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    from ngram_table import Table

    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    g = NGram.from_model(args.model)
    table = Table(args.model)
    rng = random.Random(1)
    facts = []
    for c in pq.read_table(args.data).slice(0, 4000).to_pylist():
        rw = c["requested_rewrite"]
        prompt = rw["prompt"].format(rw["subject"])
        enc = tok.encode(prompt, add_special_tokens=False)
        subj = subject_tokens(enc, prompt, rw["subject"])
        if subj and prompt.find(rw["subject"]) >= 0:
            facts.append({"id": c["case_id"], "rel": rw["relation_id"], "prompt": enc.ids, "subj": subj,
                          "answer": tok.encode(" " + rw["target_true"]["str"], add_special_tokens=False).ids,
                          "answer_str": rw["target_true"]["str"]})
    by_rel = defaultdict(list)
    for f in facts:
        by_rel[f["rel"]].append(f)
    cases = []
    for f in facts[:args.exact]:
        ids = g.ids(g.initial_history(), np.array(f["prompt"])).reshape(-1)
        u = np.unique(ids)
        rows = table.rows(u)
        cases.append({"name": f"exact:{f['id']}:none", "prompt": f["prompt"], "cont": f["answer"]})
        cases.append({"name": f"exact:{f['id']}:pyrows", "prompt": f["prompt"], "cont": f["answer"],
                      "overlay": {"ids": u.tolist(), "rows": rows.tolist()}})
    pairs = 0
    for x in rng.sample(facts, len(facts)):
        if pairs >= args.pairs:
            break
        cands = [y for y in by_rel[x["rel"]] if y["answer_str"] != x["answer_str"] and y["id"] != x["id"]]
        if not cands:
            continue
        y = rng.choice(cands)
        xi = g.ids(g.initial_history(), np.array(x["prompt"]))
        yi = g.ids(g.initial_history(), np.array(y["prompt"]))
        k = min(len(x["subj"]), len(y["subj"]))
        copy = [[int(xi[p, h]), int(yi[q, h])] for p, q in zip(x["subj"][-k:], y["subj"][-k:]) for h in range(16)]
        tag = f"{x['id']}-{y['id']}"
        for which, cont in (("ax", x["answer"]), ("ay", y["answer"])):
            cases.append({"name": f"tp:{tag}:{which}:none", "prompt": x["prompt"], "cont": cont})
            cases.append({"name": f"tp:{tag}:{which}:transplant", "prompt": x["prompt"], "cont": cont, "copy": copy})
        pairs += 1
    args.out.write_text(json.dumps({"cases": cases}))
    print(f"{args.exact} exactness facts, {pairs} transplant pairs, {len(cases)} cases -> {args.out}")


def analyse(args: argparse.Namespace) -> None:
    rep = {r["name"]: r for r in json.loads(args.report.read_text())["cases"]}
    exact = [n.rsplit(":", 1)[0] for n in rep if n.startswith("exact:") and n.endswith(":none")]
    same = sum(rep[e + ":none"]["logprobs"] == rep[e + ":pyrows"]["logprobs"] for e in exact)
    print(f"exact: {same}/{len(exact)} facts score identically with the Python-decoded rows as an overlay")
    tags = sorted({n.split(":")[1] for n in rep if n.startswith("tp:")})
    dx, dy, flips, ybetter0, ybetter1 = [], [], 0, 0, 0
    for t in tags:
        ax0, ax1 = rep[f"tp:{t}:ax:none"]["sum"], rep[f"tp:{t}:ax:transplant"]["sum"]
        ay0, ay1 = rep[f"tp:{t}:ay:none"]["sum"], rep[f"tp:{t}:ay:transplant"]["sum"]
        dx.append(ax1 - ax0)
        dy.append(ay1 - ay0)
        ybetter0 += ay0 > ax0
        ybetter1 += ay1 > ax1
        flips += ay1 > ax1 and not ay0 > ax0
    out = {"pairs": len(tags), "mean_dlogp_answer_x": round(float(np.mean(dx)), 3),
           "mean_dlogp_answer_y": round(float(np.mean(dy)), 3),
           "median_dlogp_answer_y": round(float(np.median(dy)), 3),
           "y_preferred_before": ybetter0, "y_preferred_after": ybetter1, "flipped_to_y": flips,
           "exact_facts_identical": f"{same}/{len(exact)}"}
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", type=Path, required=True)
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--pairs", type=int, default=150)
    b.add_argument("--exact", type=int, default=20)
    b.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--report", type=Path, required=True)
    a.add_argument("--out", type=Path)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else analyse(args)


if __name__ == "__main__":
    main()
