"""Phase 3R: gradient fact editing in Flash Next's n-gram memory with the PyTorch reference (run in the dev container).

Same protocol as zo_edit.py (CounterFact counterfactuals; training prompts: the rewrite statement and the first
paraphrase; held out: the second paraphrase; neighbourhood: 5 prompts). The subject-position rows get an fp32 delta;
Adam minimises -log p(new) over the training prompts plus lam * sum_r w_r |delta_r|^2, w_r growing with the row's
background frequency (rowfreq counts), until every training prompt prefers new by --margin. The edited rows are
written as overlays (ids + bf16 rows) for checking in the engine.

    python grad_edit.py --cases 10 --out /work/experiments/03r-grad-edit/run1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngram_table import bf16_rne  # noqa: E402
from ref.model import Reference  # noqa: E402

MODEL = Path("/hf/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/"
             "1464274120d36a4d8fcaa934552334a7d83ce0fd")


def subject_positions(enc, text: str, subject: str) -> list[int]:
    at = text.find(subject)
    if at < 0:
        return []
    end = at + len(subject)
    return [i for i, (a, b) in enumerate(enc.offsets) if b > at and a < end and b > a]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/work/data/counterfact/data/train-00000-of-00001-05d11247db7abce8.parquet"))
    ap.add_argument("--counts", type=Path, default=Path("/work/data/rowfreq/counts.npy"))
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--rowset", choices=("all", "inner"), default="all")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.05,
                    help="penalty on the deltas' mean squared size relative to the rows', weighted by row frequency")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--sequential", action="store_true",
                    help="keep every edit (committed rows) while editing the next; re-check earlier facts as it goes")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--control-start", type=int, default=10000,
                    help="first of 20 unedited CounterFact facts whose true answers must survive the edits")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    counts = np.load(args.counts, mmap_mode="r")
    ref = Reference(MODEL)
    rows = pq.read_table(args.data).slice(args.start, args.cases * 3).to_pylist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results, overlays = [], []
    done, timeline = [], []
    control = []
    for c in pq.read_table(args.data).slice(args.control_start, 20).to_pylist():
        rw = c["requested_rewrite"]
        control.append((enc(rw["prompt"].format(rw["subject"])).ids, enc(" " + rw["target_new"]["str"]).ids,
                        enc(" " + rw["target_true"]["str"]).ids))

    def lp(prompt, ans, grad=False):
        with torch.set_grad_enabled(grad):
            return ref.answer_logprob(prompt, ans)

    def evaluate(sets, new, true):
        out = {}
        for name, prompts in sets.items():
            pref = [lp(p, new).item() > lp(p, true).item() for p in prompts]
            if name == "specificity":
                pref = [not x for x in pref]
            out[name] = float(np.mean(pref))
        return out

    with torch.no_grad():
        ctrl0 = round(float(np.mean([ref.answer_logprob(p, tr_).item() > ref.answer_logprob(p, nw_).item()
                                     for p, nw_, tr_ in control])), 3)
    print("control: true preferred before editing", ctrl0, flush=True)
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
            ti = ref.rows.ids(np.array(e.ids))
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
        ref.rows.choose(np.concatenate(chosen))
        sel = ref.rows.sel
        w = torch.tensor(1.0 + counts[sel].astype(np.float64) / 39.0, device="cuda", dtype=torch.float32)
        sets = {"efficacy": [e.ids for e in train], "generalisation": [enc(s).ids for s in c["paraphrase_prompts"][1:]],
                "specificity": [enc(s).ids for s in c["neighborhood_prompts"][:5]]}
        base0 = (torch.from_numpy(ref.rows.base_bits(sel).astype(np.int16)).to("cuda").to(torch.int32) << 16).view(torch.float32)
        s2 = float((base0 ** 2).mean())
        ref.rows.delta.requires_grad_(False)
        before = evaluate(sets, new, true)
        ref.rows.delta.requires_grad_(True)
        opt = torch.optim.Adam([ref.rows.delta], lr=args.lr)
        t0 = time.time()
        steps, margins = 0, []
        for step in range(1, args.steps + 1):
            opt.zero_grad()
            nll = sum(-lp(e.ids, new, grad=True) for e in train) / len(train)
            reg = args.lam * (w[:, None] * ref.rows.delta ** 2).mean() / s2
            (nll + reg).backward()
            opt.step()
            steps = step
            with torch.no_grad():
                margins = [lp(e.ids, new).item() - lp(e.ids, true).item() for e in train]
            print(f"  case {c['case_id']} step {step}: nll {nll.item():.3f} reg {reg.item():.3f} "
                  f"margins {[round(m, 2) for m in margins]}", flush=True)
            if min(margins) > args.margin:
                break
        ref.rows.delta.requires_grad_(False)
        after = evaluate(sets, new, true)
        delta = ref.rows.delta.detach()
        base = (torch.from_numpy(ref.rows.base_bits(sel).astype(np.int16)).to("cuda").to(torch.int32) << 16).view(torch.float32)
        edited = (base + delta).cpu().numpy()
        r = {"case_id": c["case_id"], "subject": subj, "true": rw["target_true"]["str"], "new": rw["target_new"]["str"],
             "rows": int(len(sel)), "steps": steps, "seconds": round(time.time() - t0, 1), "before": before,
             "after": after, "final_margins": [round(m, 2) for m in margins],
             "delta_rel": round(float(delta.norm() / base.norm()), 3)}
        results.append(r)
        overlays.append({"case_id": c["case_id"], "ids": sel.tolist(), "rows": bf16_rne(edited).tolist(),
                         "prompts": sets, "new": new, "true": true})
        print(json.dumps(r), flush=True)
        if args.sequential:
            ref.rows.commit(sel, bf16_rne(edited))
            done.append((sets, new, true))
            if len(results) % args.eval_every == 0:
                with torch.no_grad():
                    sample = [done[i] for i in sorted(np.random.default_rng(len(done)).choice(len(done), min(50, len(done)), replace=False))]
                    kept = [evaluate({k: v for k, v in st.items() if k != "specificity"}, nw, tr) for st, nw, tr in sample]
                    ctrl = float(np.mean([lp(p, tr_).item() > lp(p, nw_).item() for p, nw_, tr_ in control]))
                point = {"edits": len(results), "committed_rows": len(ref.rows.committed),
                         "earlier_efficacy": round(float(np.mean([k["efficacy"] for k in kept])), 3),
                         "earlier_generalisation": round(float(np.mean([k["generalisation"] for k in kept])), 3),
                         "control_true_kept": round(ctrl, 3), "control_true_kept_before": ctrl0}
                timeline.append(point)
                print("SEQ", json.dumps(point), flush=True)
        args.out.write_text(json.dumps({"results": results, "timeline": timeline}, indent=1))
    keys = ("efficacy", "generalisation", "specificity")
    summary = {f"{when}_{k}": round(float(np.mean([r[when][k] for r in results])), 3)
               for when in ("before", "after") for k in keys}
    summary.update(cases=len(results), mean_seconds=round(float(np.mean([r["seconds"] for r in results])), 1),
                   mean_steps=round(float(np.mean([r["steps"] for r in results])), 1))
    print(json.dumps(summary, indent=1))
    args.out.write_text(json.dumps({"summary": summary, "results": results, "timeline": timeline, "args": vars(args) | {"out": str(args.out),
                                    "data": str(args.data), "counts": str(args.counts)}}, indent=1))
    args.out.with_suffix(".overlays.json").write_text(json.dumps(overlays))


if __name__ == "__main__":
    main()
