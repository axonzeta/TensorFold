"""Phase 3 B11: fact editing with the engine's own training pass (tensorfold worker "grad"), one copy of the weights.

grad_edit.py's protocol and hyperparameters (CounterFact; training prompts = rewrite statement + first paraphrase;
held out = second paraphrase; neighbourhood = 5 prompts; context-free subject rows; Adam lr 2e-3 on -log p(new) plus
the frequency-weighted penalty; stop when both training prompts prefer new by 2), with gradients from the engine and
every score from the served engine with the edited rows as an overlay.

    uv run engine_edit.py --cases 10 --out ../experiments/03-backward/engine-edit/run1.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ngram_ids import NGram
from ngram_table import Table, bf16_rne, bf16_to_f32, write_overlay
from worker import Worker
import paths

MODEL = paths.MODEL


def subject_positions(enc, text: str, subject: str) -> list[int]:
    at = text.find(subject)
    if at < 0:
        return []
    end = at + len(subject)
    return [i for i, (a, b) in enumerate(enc.offsets) if b > at and a < end and b > a]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data/counterfact/data/train-00000-of-00001-05d11247db7abce8.parquet")
    ap.add_argument("--counts", type=Path, default=Path(__file__).resolve().parent.parent / "data/rowfreq/counts.npy")
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--sequential", action="store_true",
                    help="commit each edit into the worker's base overlay; the next fact starts from the committed rows")
    ap.add_argument("--eval-every", type=int, default=50, help="re-check earlier facts (a sample of 100) this often")
    ap.add_argument("--control-start", type=int, default=10000, help="first of 20 unedited facts whose true answer must survive")
    ap.add_argument("--no-before", action="store_true", help="skip scoring each fact before its edit (faster at scale)")
    ap.add_argument("--case-index", type=Path, help="a JSON list of dataset rows to edit in that order (instead of --start)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    g, table = NGram.from_model(MODEL), Table(MODEL)
    counts = np.load(args.counts, mmap_mode="r")
    tbl = pq.read_table(args.data)
    rows = (tbl.take(json.loads(args.case_index.read_text())) if args.case_index else
            tbl.slice(args.start, int(args.cases * 1.5) + 10)).to_pylist()
    control = []
    for cc in tbl.slice(args.control_start, 20).to_pylist():
        rw = cc["requested_rewrite"]
        control.append((enc(rw["prompt"].format(rw["subject"])).ids, enc(" " + rw["target_new"]["str"]).ids,
                        enc(" " + rw["target_true"]["str"]).ids))
    committed: dict[int, np.ndarray] = {}  # row id -> fp32 row of an earlier edit (sequential)
    done, timeline = [], []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with Worker() as w:
        def lp(prompt, ans, ov):
            kw = {"overlay": ov} if ov else {}
            return w.case(prompt=prompt, cont=ans, **kw)["sum"]

        def evaluate(sets, new, true, ov):
            out = {}
            for name, prompts in sets.items():
                pref = [lp(p, new, ov) > lp(p, true, ov) for p in prompts]
                out[name] = float(np.mean([not x for x in pref] if name == "specificity" else pref))
            return out

        def recheck(n_edits: int) -> None:
            pick = np.random.default_rng(n_edits).choice(len(done), min(100, len(done)), replace=False)
            kept = [evaluate({k: v for k, v in done[i][0].items() if k != "specificity"}, done[i][1], done[i][2], None)
                    for i in sorted(pick)]
            ctrl = float(np.mean([lp(pp, tr, None) > lp(pp, nw, None) for pp, nw, tr in control]))
            point = {"edits": n_edits, "committed_rows": len(committed),
                     "earlier_efficacy": round(float(np.mean([k["efficacy"] for k in kept])), 3),
                     "earlier_generalisation": round(float(np.mean([k["generalisation"] for k in kept])), 3),
                     "control_true_kept": round(ctrl, 3), "control_true_kept_before": ctrl0}
            timeline.append(point)
            print("SEQ", json.dumps(point), flush=True)
            ids = sorted(committed)  # the overlay so far, for drift checks while the run goes on
            write_overlay(args.out.with_name(f"{args.out.stem}.at{n_edits}.overlay.safetensors"), ids,
                          bf16_rne(np.stack([committed[i] for i in ids]).astype(np.float32)))

        ctrl0 = round(float(np.mean([lp(pp, tr, None) > lp(pp, nw, None) for pp, nw, tr in control])), 3)
        for c in rows:
            if len(results) >= args.cases:
                break
            rw = c["requested_rewrite"]
            subj = rw["subject"]
            new = enc(" " + rw["target_new"]["str"]).ids
            true = enc(" " + rw["target_true"]["str"]).ids
            train_text = [rw["prompt"].format(subj), c["paraphrase_prompts"][0]]
            train = [enc(s) for s in train_text]
            pos = [subject_positions(e, s, subj) for e, s in zip(train, train_text)]
            if not all(pos) or len(c["paraphrase_prompts"]) < 2:
                continue
            chosen = []
            for e, p in zip(train, pos):
                ti = g.ids(g.initial_history(), np.array(e.ids))
                for t in p:
                    if len(p) == 1:
                        chosen.append(ti[t])
                        continue
                    if t - 1 in p:
                        chosen.append(ti[t, :8])
                    if t - 1 in p and t - 2 in p:
                        chosen.append(ti[t, 8:])
            if not chosen:
                continue
            sel = np.unique(np.concatenate(chosen))
            base = bf16_to_f32(table.rows(sel)).astype(np.float64)
            for k, i in enumerate(sel):
                if int(i) in committed:
                    base[k] = committed[int(i)]
            s2 = float((base ** 2).mean())
            wgt = 1.0 + counts[sel].astype(np.float64) / 39.0
            delta = np.zeros_like(base)
            m1, m2 = np.zeros_like(base), np.zeros_like(base)
            overlay = lambda: {"ids": sel.tolist(), "rows": bf16_rne((base + delta).astype(np.float32)).tolist()}  # noqa: E731
            sets = {"efficacy": [e.ids for e in train], "generalisation": [enc(s).ids for s in c["paraphrase_prompts"][1:]],
                    "specificity": [enc(s).ids for s in c["neighborhood_prompts"][:5]]}
            before = None if args.no_before else evaluate(sets, new, true, None)
            t0 = time.time()
            steps, margins = 0, []
            for step in range(1, args.steps + 1):
                ov = overlay()
                grad = np.zeros_like(base)
                nll = 0.0
                lessons = [{"tokens": e.ids + new[:-1], "at": list(range(len(e.ids) - 1, len(e.ids) + len(new) - 1)),
                            "targets": new} for e in train]
                r = w.send({"op": "grad", "lessons": lessons, "sel": sel.tolist(), "overlay": ov})
                nll += r["loss"] / len(train)
                grad += np.asarray(r["grad"], np.float64) / len(train)
                n_el = delta.size
                grad += args.lam * 2 * wgt[:, None] * delta / (n_el * s2)
                m1 = 0.9 * m1 + 0.1 * grad
                m2 = 0.999 * m2 + 0.001 * grad ** 2
                delta -= args.lr * (m1 / (1 - 0.9 ** step)) / (np.sqrt(m2 / (1 - 0.999 ** step)) + 1e-8)
                steps = step
                ov = overlay()
                margins = [lp(e.ids, new, ov) - lp(e.ids, true, ov) for e in train]
                print(f"  case {c['case_id']} step {step}: nll {nll:.3f} margins {[round(x, 2) for x in margins]}", flush=True)
                if min(margins) > args.margin:
                    break
            after = evaluate(sets, new, true, overlay())
            res = {"case_id": c["case_id"], "subject": subj, "true": rw["target_true"]["str"], "new": rw["target_new"]["str"],
                   "rows": int(len(sel)), "steps": steps, "seconds": round(time.time() - t0, 1), "before": before,
                   "after": after, "final_margins": [round(x, 2) for x in margins]}
            results.append(res)
            print(json.dumps(res), flush=True)
            if args.sequential:
                edited = (base + delta).astype(np.float32)
                for i, v in zip(sel, edited):
                    committed[int(i)] = v.astype(np.float64)
                w.send({"op": "base", "overlay": {"ids": sel.tolist(), "rows": bf16_rne(edited).tolist()}})
                done.append((sets, new, true))
                if len(results) % args.eval_every == 0:
                    recheck(len(results))
            args.out.write_text(json.dumps({"results": results, "timeline": timeline}, indent=1))
    keys = ("efficacy", "generalisation", "specificity")
    summary = {f"{when}_{k}": round(float(np.mean([r[when][k] for r in results])), 3)
               for when in ("before", "after") for k in keys if all(r[when] for r in results)}
    summary.update(cases=len(results), mean_seconds=round(float(np.mean([r["seconds"] for r in results])), 1),
                   mean_steps=round(float(np.mean([r["steps"] for r in results])), 1))
    print(json.dumps(summary, indent=1))
    args.out.write_text(json.dumps({"summary": summary, "results": results, "timeline": timeline}, indent=1))
    if committed:
        ids = sorted(committed)
        write_overlay(args.out.with_suffix(".overlay.safetensors"), ids,
                      bf16_rne(np.stack([committed[i] for i in ids]).astype(np.float32)))


if __name__ == "__main__":
    main()
