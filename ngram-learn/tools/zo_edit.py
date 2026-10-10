"""Phase 2b: forward-only fact editing in Flash Next's n-gram memory (SPSA on an overlay, no backward).

For each CounterFact case: the rows at the subject's positions in the training prompts (the rewrite statement and the
first paraphrase) are the parameters, moved in a d-dimensional random subspace scaled to the rows' own size. SPSA
estimates the gradient of the training loss (mean -log p(new answer)) from antithetic pairs of forward passes; Adam
steps. Scored before and after, as CounterFact does: efficacy (training prompts prefer the new answer to the true
one), generalisation (the held-out paraphrase does) and specificity (neighbourhood prompts still prefer the true one).

    uv run zo_edit.py --model DIR --data counterfact.parquet --cases 20 --out ../experiments/02b-zo-edit/run1.json
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


def subject_positions(enc, text: str, subject: str) -> list[int]:
    at = text.find(subject)
    if at < 0:
        return []
    end = at + len(subject)
    return [i for i, (a, b) in enumerate(enc.offsets) if b > at and a < end and b > a]


class Edit:
    """One fact's trainable rows: base rows (fp32) plus U z, sent to the worker as a case overlay."""

    def __init__(self, ids: np.ndarray, base: np.ndarray, dim: int, rng: np.random.Generator,
                 weight: np.ndarray | None = None):
        self.ids = ids
        self.base = base
        scale = float(np.sqrt((base ** 2).mean()))
        self.U = rng.standard_normal((base.size, dim)).astype(np.float32) * (scale / np.sqrt(dim))
        if weight is not None:  # a row's share of every direction (rare rows move more)
            self.U *= np.repeat(weight.astype(np.float32), base.shape[1])[:, None]
        self.z = np.zeros(dim, np.float32)

    def overlay(self, z: np.ndarray | None = None) -> dict:
        rows = self.base + (self.U @ (self.z if z is None else z)).reshape(self.base.shape)
        return {"ids": self.ids.tolist(), "rows": bf16_rne(rows).tolist()}


def logp(w: Worker, prompt: list[int], cont: list[int], overlay: dict | None) -> float:
    kw = {"prompt": prompt, "cont": cont}
    if overlay:
        kw["overlay"] = overlay
    return w.case(**kw)["sum"]


def evaluate(w: Worker, sets: dict, new: list[int], true: list[int], overlay: dict | None) -> dict:
    out = {}
    for name, prompts in sets.items():
        prefer_new = [logp(w, p, new, overlay) > logp(w, p, true, overlay) for p in prompts]
        if name == "specificity":
            prefer_new = [not x for x in prefer_new]  # neighbours should keep preferring the true answer
        out[name] = float(np.mean(prefer_new)) if prefer_new else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dirs", type=int, default=2, help="antithetic SPSA directions a step")
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=1.0, help="stop once every training prompt prefers new by this")
    ap.add_argument("--objective", choices=("margin", "new"), default="margin",
                    help="margin: maximise log p(new) - log p(true); new: maximise log p(new)")
    ap.add_argument("--rowset", choices=("all", "inner"), default="all",
                    help="inner: only rows whose n-gram lies inside the subject (bigram heads where the previous token is "
                         "in the subject, trigram heads where both are); falls back to all for one-token subjects")
    ap.add_argument("--rare", type=Path, help="rowfreq counts.npy: weight rows by 1/sqrt(1 + count/median)")
    ap.add_argument("--init", choices=("none", "transplant"), default="transplant",
                    help="transplant: start the subject's rows from those of a subject whose true answer is the target")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    g, table = NGram.from_model(args.model), Table(args.model)
    rng = np.random.default_rng(0)
    counts = np.load(args.rare, mmap_mode="r") if args.rare else None
    everything = pq.read_table(args.data).to_pylist()
    rows = everything[args.start:args.start + args.cases * 3]
    donors: dict[tuple[str, str], list[dict]] = {}
    for d in everything:
        donors.setdefault((d["requested_rewrite"]["relation_id"], d["requested_rewrite"]["target_true"]["str"]), []).append(d)
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Worker() as w:
        for c in rows:
            if len(results) >= args.cases:
                break
            rw = c["requested_rewrite"]
            subj = rw["subject"]
            enc = lambda s: tok.encode(s, add_special_tokens=False)
            new = enc(" " + rw["target_new"]["str"]).ids
            true = enc(" " + rw["target_true"]["str"]).ids
            train_text = [rw["prompt"].format(subj), c["paraphrase_prompts"][0]]
            held_text = c["paraphrase_prompts"][1:]
            neigh_text = c["neighborhood_prompts"][:5]
            train = [enc(s) for s in train_text]
            pos = [subject_positions(e, s, subj) for e, s in zip(train, train_text)]
            if not all(pos) or not held_text:
                continue
            chosen = []
            for e, p in zip(train, pos):
                ti = g.ids(g.initial_history(), np.array(e.ids))
                for t in p:
                    if args.rowset == "all" or len(p) == 1:
                        chosen.append(ti[t])
                        continue
                    if t - 1 in p:
                        chosen.append(ti[t, :8])
                    if t - 1 in p and t - 2 in p:
                        chosen.append(ti[t, 8:])
            if not chosen:
                continue
            ids = np.unique(np.concatenate(chosen))
            base = bf16_to_f32(table.rows(ids))
            donor_subject = None
            if args.init == "transplant":
                cands = [d for d in donors.get((rw["relation_id"], rw["target_new"]["str"]), [])
                         if d["case_id"] != c["case_id"]]
                for d in cands:
                    dt = d["requested_rewrite"]["prompt"].format(d["requested_rewrite"]["subject"])
                    de = enc(dt)
                    dp = subject_positions(de, dt, d["requested_rewrite"]["subject"])
                    if not dp:
                        continue
                    di = g.ids(g.initial_history(), np.array(de.ids))
                    where = {int(x): k for k, x in enumerate(ids)}
                    for e, p in zip(train, pos):
                        ti = g.ids(g.initial_history(), np.array(e.ids))
                        for a, b in zip(reversed(p), reversed(dp)):
                            for h in range(16):
                                if int(ti[a, h]) in where:
                                    base[where[int(ti[a, h])]] = bf16_to_f32(table.rows([int(di[b, h])]))[0]
                    donor_subject = d["requested_rewrite"]["subject"]
                    break
            weight = None
            if counts is not None:
                cnt = counts[ids].astype(np.float64)
                weight = 1.0 / np.sqrt(1.0 + cnt / 39.0)
                weight /= np.sqrt((weight ** 2).mean())
            edit = Edit(ids, base, args.dim, rng, weight)
            sets = {"efficacy": [e.ids for e in train], "generalisation": [enc(s).ids for s in held_text],
                    "specificity": [enc(s).ids for s in neigh_text]}
            before = evaluate(w, sets, new, true, None)
            init = evaluate(w, sets, new, true, edit.overlay()) if donor_subject else None

            def objective(z: np.ndarray) -> float:
                ov = edit.overlay(z)
                vals = []
                for e in train:
                    lp_new = logp(w, e.ids, new, ov)
                    vals.append(lp_new - logp(w, e.ids, true, ov) if args.objective == "margin" else lp_new)
                return -float(np.mean(vals))

            def loss(z: np.ndarray) -> tuple[float, list[float]]:
                ov = edit.overlay(z)
                margins = [logp(w, e.ids, new, ov) - logp(w, e.ids, true, ov) for e in train]
                return -float(np.mean(margins)), margins

            _, start_margins = loss(edit.z)

            m = np.zeros(args.dim, np.float32)
            v = np.zeros(args.dim, np.float32)
            t0 = time.time()
            steps = 0
            for step in range(1, args.steps + 1):
                if min(start_margins) > args.margin and step == 1:
                    break
                grad = np.zeros(args.dim, np.float32)
                for _ in range(args.dirs):
                    u = rng.standard_normal(args.dim).astype(np.float32)
                    grad += (objective(edit.z + args.eps * u) - objective(edit.z - args.eps * u)) / (2 * args.eps) * u / args.dirs
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad ** 2
                edit.z -= args.lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
                steps = step
                if step % 10 == 0:
                    cur, margins = loss(edit.z)
                    print(f"  case {c['case_id']} step {step}: loss {cur:.3f} margins {[round(x, 2) for x in margins]}",
                          flush=True)
                    if min(margins) > args.margin:
                        break
            after = evaluate(w, sets, new, true, edit.overlay())
            r = {"case_id": c["case_id"], "subject": subj, "new": rw["target_new"]["str"],
                 "true": rw["target_true"]["str"], "rows": int(ids.size), "steps": steps,
                 "seconds": round(time.time() - t0, 1), "before": before, "init": init, "donor": donor_subject,
                 "after": after,
                 "delta_norm_rel": round(float(np.linalg.norm(edit.U @ edit.z) / np.linalg.norm(edit.base)), 3)}
            results.append(r)
            print(json.dumps(r), flush=True)
            args.out.write_text(json.dumps(results, indent=1))
    keys = ("efficacy", "generalisation", "specificity")
    summary = {f"{when}_{k}": round(float(np.mean([r[when][k] for r in results if r[when][k] is not None])), 3)
               for when in ("before", "after") for k in keys}
    with_init = [r for r in results if r["init"]]
    for k in keys:
        vals = [r["init"][k] for r in with_init if r["init"][k] is not None]
        summary[f"transplant_only_{k}"] = round(float(np.mean(vals)), 3) if vals else None
    summary["with_donor"] = len(with_init)
    summary["cases"] = len(results)
    summary["mean_seconds"] = round(float(np.mean([r["seconds"] for r in results])), 1)
    print(json.dumps(summary, indent=1))
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=1))


if __name__ == "__main__":
    main()
