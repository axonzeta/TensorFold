"""Phase 4 suite: new facts taught as chat turns with the engine's own training pass, recalled through the chat template.

Each fact in facts_suite.json: a subject, a key string, three training (question, answer) turns, two held-out questions,
three neighbour questions. The subject's context-free n-gram rows in the training prompts are trained (Adam on
-log p(answer) plus the frequency-weighted penalty) until every training answer is the greedy continuation. Scored by
greedy generation through the served engine (thinking off): recall (training questions' replies contain the key),
generalisation (held-out questions' replies do) and neighbours (their replies are unchanged).

    uv run chat_edit.py --out ../experiments/04-chat-facts/run1.json [--sequential]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ngram_ids import NGram
from ngram_table import Table, bf16_rne, bf16_to_f32
from worker import Worker
import paths

MODEL = paths.MODEL
TEMPLATE = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=Path(__file__).resolve().parent / "facts_suite.json")
    ap.add_argument("--counts", type=Path, default=Path(__file__).resolve().parent.parent / "data/rowfreq/counts.npy")
    ap.add_argument("--only", type=int, default=0, help="first N facts (0: all)")
    ap.add_argument("--kind", help="only facts of this kind (e.g. user)")
    ap.add_argument("--rows", choices=("inner", "all"), default="inner",
                    help="inner: the subject's context-free rows; all: every row at the subject's positions")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--sequential", action="store_true", help="keep each fact's rows (base overlay) while teaching the next")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-batch", action="store_true", help="one grad op a training question (not one a step)")
    ap.add_argument("--check-below", type=float, default=0.25,
                    help="run the greedy check once the loss a answer token is below this (and every 5th step)")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    g, table = NGram.from_model(MODEL), Table(MODEL)
    counts = np.load(args.counts, mmap_mode="r")
    facts = json.loads(args.suite.read_text())
    if args.kind:
        facts = [f for f in facts if f["kind"] == args.kind]
    if args.only:
        facts = facts[:args.only]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    committed: dict[int, np.ndarray] = {}
    results = []

    def prompt(q: str):
        text = TEMPLATE.format(q=q)
        return text, tok.encode(text, add_special_tokens=False)

    def rows_of(ids: np.ndarray) -> np.ndarray:
        base = bf16_to_f32(table.rows(ids)).astype(np.float64)
        for k, i in enumerate(ids):
            if int(i) in committed:
                base[k] = committed[int(i)]
        return base

    with Worker() as w:
        def reply(ids, ov) -> str:
            kw = {"overlay": ov} if ov else {}
            r = w.case(prompt=ids, gen=args.gen, **kw)
            return tok.decode(r["gen"])

        base_neigh = {}
        for f in facts:
            for q in f["neigh"]:
                base_neigh[q] = reply(prompt(q)[1].ids, None)
        for f in facts:
            subj, key = f["subject"], f["key"]
            train = []
            # the fact's keys: its subject, or (multi-key facts) every phrase in "keys"; each occurrence in a
            # question is one span of token positions
            keys = f.get("keys") or [subj]
            for q, a in f["train"]:
                text, e = prompt(q)
                spans = []
                for k in keys:
                    at = text.lower().find(k.lower())
                    while at >= 0:
                        spans.append([i for i, (x, y) in enumerate(e.offsets) if y > at and x < at + len(k) and y > x])
                        at = text.lower().find(k.lower(), at + 1)
                ans = tok.encode(a + "<|im_end|>", add_special_tokens=False).ids
                train.append((e.ids, [sp for sp in spans if sp], ans))
            chosen = []
            for ids, spans, _ in train:
                ti = g.ids(g.initial_history(), np.array(ids))
                for pos in spans:
                    for t in pos:
                        if len(pos) == 1 or args.rows == "all":
                            chosen.append(ti[t])
                            continue
                        if t - 1 in pos:
                            chosen.append(ti[t, :8])
                        if t - 1 in pos and t - 2 in pos:
                            chosen.append(ti[t, 8:])
            if not chosen:
                print(f"skip {subj}: subject not found in its questions", flush=True)
                continue
            sel = np.unique(np.concatenate(chosen))
            base = rows_of(sel)
            s2 = float((base ** 2).mean())
            wgt = 1.0 + counts[sel].astype(np.float64) / 39.0
            delta = np.zeros_like(base)
            m1, m2 = np.zeros_like(base), np.zeros_like(base)
            all_ov = lambda: {"ids": [int(i) for i in committed] + sel.tolist(),  # noqa: E731
                              "rows": [bf16_rne(v.astype(np.float32)).tolist() for v in committed.values()]
                              + bf16_rne((base + delta).astype(np.float32)).tolist()}

            def score(ov):
                hit = lambda q: key.lower() in reply(prompt(q)[1].ids, ov).lower()  # noqa: E731
                return {"recall": float(np.mean([hit(q) for q, _ in f["train"]])),
                        "generalisation": float(np.mean([hit(q) for q in f["held"]])),
                        "neighbours_unchanged": float(np.mean([reply(prompt(q)[1].ids, ov) == base_neigh[q] for q in f["neigh"]]))}
            before = score({"ids": [int(i) for i in committed], "rows": [bf16_rne(v.astype(np.float32)).tolist() for v in committed.values()]} if committed else None)
            t0 = time.time()
            steps = 0
            grad_s = check_s = 0.0
            for step in range(1, args.steps + 1):
                ov = all_ov()
                grad = np.zeros_like(base)
                nll = 0.0
                lessons = [{"tokens": ids + ans[:-1], "at": list(range(len(ids) - 1, len(ids) + len(ans) - 1)), "targets": ans}
                           for ids, _, ans in train]
                # one batched op, or (--no-batch) one flat op a question, as engines without "lessons" take it
                tg = time.time()
                for op in ([{"lessons": lessons}] if not args.no_batch else lessons):
                    r = w.send({"op": "grad", **op, "sel": sel.tolist(), "overlay": ov})
                    nll += r["loss"] / len(train)
                    grad += np.asarray(r["grad"], np.float64) / len(train)
                grad += args.lam * 2 * wgt[:, None] * delta / (delta.size * s2)
                m1 = 0.9 * m1 + 0.1 * grad
                m2 = 0.999 * m2 + 0.001 * grad ** 2
                delta -= args.lr * (m1 / (1 - 0.9 ** step)) / (np.sqrt(m2 / (1 - 0.999 ** step)) + 1e-8)
                steps = step
                grad_s += time.time() - tg
                # the greedy check only once the answers are nearly fitted (it cannot pass before), or every 5th step
                per_token = nll / float(np.mean([len(ans) for _, _, ans in train]))
                if per_token > args.check_below and step % 5 and step < args.steps:
                    print(f"  {subj} step {step}: nll {nll:.3f}", flush=True)
                    continue
                tc = time.time()
                ov = all_ov()
                greedy = [w.case(prompt=ids, cont=ans, overlay=ov) for ids, _, ans in train]
                check_s += time.time() - tc
                done = all(x["greedy_hits"] == len(ans) for x, (_, _, ans) in zip(greedy, train))
                print(f"  {subj} step {step}: nll {nll:.3f} greedy {[x['greedy_hits'] for x in greedy]}", flush=True)
                if done:
                    break
            after = score(all_ov())
            if args.sequential:
                for i, v in zip(sel, base + delta):
                    committed[int(i)] = v
            res = {"subject": subj, "kind": f["kind"], "rows": int(len(sel)), "steps": steps,
                   "seconds": round(time.time() - t0, 1), "grad_seconds": round(grad_s, 1), "check_seconds": round(check_s, 1),
                   "before": before, "after": after,
                   "sample_reply": reply(prompt(f["held"][0])[1].ids, all_ov())}
            results.append(res)
            print(json.dumps(res, ensure_ascii=False), flush=True)
            args.out.write_text(json.dumps({"results": results}, indent=1, ensure_ascii=False))
        if args.sequential and committed:
            ov = {"ids": [int(i) for i in committed], "rows": [bf16_rne(v.astype(np.float32)).tolist() for v in committed.values()]}
            final = []
            for f in facts:
                hit = lambda q: f["key"].lower() in reply(prompt(q)[1].ids, ov).lower()  # noqa: E731
                final.append({"subject": f["subject"], "recall": float(np.mean([hit(q) for q, _ in f["train"]])),
                              "generalisation": float(np.mean([hit(q) for q in f["held"]])),
                              "neighbours_unchanged": float(np.mean([reply(prompt(q)[1].ids, ov) == base_neigh[q] for q in f["neigh"]]))})
            results.append({"final_all_facts": final})
    keys = ("recall", "generalisation", "neighbours_unchanged")
    per = [r for r in results if "after" in r]
    summary = {f"{when}_{k}": round(float(np.mean([r[when][k] for r in per])), 3) for when in ("before", "after") for k in keys}
    by_kind = {}
    for r in per:
        by_kind.setdefault(r["kind"], []).append(r["after"]["generalisation"])
    summary["generalisation_by_kind"] = {k: round(float(np.mean(v)), 2) for k, v in by_kind.items()}
    summary.update(facts=len(per), mean_steps=round(float(np.mean([r["steps"] for r in per])), 1),
                   mean_seconds=round(float(np.mean([r["seconds"] for r in per])), 1))
    print(json.dumps(summary, indent=1), flush=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
