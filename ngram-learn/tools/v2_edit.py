"""Learner v2 prototype (EngramEdit-style) on the PyTorch reference, beside grad_edit.py's v1 (run in the dev container).

v1 trains a fact's rows end to end and commits them; a later fact that shares rows starts from (and moves) them.
v2 splits the two:

  A. Target: per-position copies of the fact's chosen rows (decoupled across positions and prompts), trained end to
     end until the new answer wins by --margin. What is stored is the block's output they give, over each training
     prompt, with the block's inputs (the layer's stream h and the token ids): an output the block can reach. (A
     free per-stream change at the output cannot be reached: the block gives per-stream gates times one shared
     value, so a solve could not fit it.)
  B. Solve: the rows every stored target asks for, together, from the table's own rows: the block run on each stored
     prompt with the rows + R should give its old output + delta at the subject positions and its old output
     elsewhere; R is penalised by row frequency. Only the block runs, no model backward, so re-solving after an
     edit or a removal is cheap, and the result does not depend on the order facts came in.

CounterFact protocol as grad_edit.py (training: the rewrite and its first paraphrase; held out: the second; 5
neighbourhood prompts; 20 control facts whose true answers must survive), --sequential to keep every edit.

    python v2_edit.py --cases 50 --sequential --out /work/experiments/05-v2/seq-50.json
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
from grad_edit import MODEL, subject_positions  # noqa: E402
from ngram_table import bf16_rne  # noqa: E402
from ref.model import Reference  # noqa: E402

PLE_LAYER = 1


class Hooked:
    """Layer 1's n-gram block with a change added to its output (`inject`: [T, S*D] or None) and its inputs kept
    (`capture`: the stream h and ids of the last call)."""

    def __init__(self, ref: Reference):
        self.ref = ref
        self.inject: torch.Tensor | None = None
        self.capture: dict | None = None
        self.block = None
        build = ref.build

        def hooked_build(idx: int):
            layer = build(idx)
            if idx == PLE_LAYER:
                ple = layer.ple
                self.block = ple
                forward = ple.forward

                def f(hidden_states, input_ids, past_key_values, conv_mask=None):
                    out = forward(hidden_states, input_ids, past_key_values, conv_mask)
                    if self.capture is not None:
                        self.capture["h"] = hidden_states.detach().clone()
                        self.capture["ids"] = input_ids.detach().clone()
                        self.capture["out"] = out.detach().clone()
                    if self.inject is not None:
                        out = out + self.inject.to(out.dtype)[None]
                    return out

                ple.forward = f
            return layer

        ref.build = hooked_build


def block_out(hk: Hooked, h: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The n-gram block on stored inputs with the rows as Rows gives them now (its delta differentiable)."""
    if hk.block is None:
        hk.ref.build(PLE_LAYER)
    return hk.block.__class__.forward(hk.block, h, ids, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/work/data/counterfact/data/train-00000-of-00001-05d11247db7abce8.parquet"))
    ap.add_argument("--counts", type=Path, default=Path("/work/data/rowfreq/counts.npy"))
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--case-index", type=Path, help="a JSON list of dataset rows to edit in that order (instead of --start)")
    ap.add_argument("--steps", type=int, default=25, help="target steps a fact")
    ap.add_argument("--lr", type=float, default=2e-3, help="Adam on the per-position row copies (as grad_edit.py)")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--solve-steps", type=int, default=300)
    ap.add_argument("--solve-lr", type=float, default=5e-3)
    ap.add_argument("--lam", type=float, default=0.05, help="row-change penalty, frequency-weighted, as grad_edit.py")
    ap.add_argument("--keep", type=float, default=1.0, help="weight of the off-subject positions keeping their output")
    ap.add_argument("--targets", choices=("hybrid", "shared", "copies"), default="hybrid",
                    help="shared: the fact's rows trained as v1 does, their outputs the targets, every row re-solved; "
                         "hybrid: as shared, but a row only this fact uses keeps its trained value and only rows "
                         "facts share are re-solved; copies: per-position copies as targets")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--control-start", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    counts = np.load(args.counts, mmap_mode="r")
    ref = Reference(MODEL)
    hk = Hooked(ref)
    table = pq.read_table(args.data)
    data = (table.take(json.loads(args.case_index.read_text())) if args.case_index else
            table.slice(args.start, args.cases * 3)).to_pylist()
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
            out[name] = float(np.mean([not x for x in pref] if name == "specificity" else pref))
        return out

    with torch.no_grad():
        ctrl0 = float(np.mean([lp(p, tr).item() > lp(p, nw).item() for p, nw, tr in control]))
    print("control: true preferred before editing", ctrl0, flush=True)
    stored: list[dict] = []  # every kept fact's targets: [{"ids", "h", "out", "pos", "delta", "rows"}]
    results, timeline, done = [], [], []

    def solve(targets: list[dict], free: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """The rows all `targets` ask for, from the table's rows: every row they use (committed edits cleared), or
        only `free` (those start from the table; every other committed row stays as it is)."""
        if free is None:
            sel = np.unique(np.concatenate([t["rows"] for t in targets]))
            ref.rows.committed = {}
        else:
            sel = np.unique(free)
            for i in sel:
                ref.rows.committed.pop(int(i), None)
        ref.rows.choose(sel)
        w = torch.tensor(1.0 + counts[sel].astype(np.float64) / 39.0, device="cuda", dtype=torch.float32)
        base0 = (torch.from_numpy(ref.rows.base_bits(sel).astype(np.int16)).to("cuda").to(torch.int32) << 16).view(torch.float32)
        s2 = float((base0 ** 2).mean())
        opt = torch.optim.Adam([ref.rows.delta], lr=args.solve_lr)
        t0 = time.time()
        for _ in range(args.solve_steps):
            opt.zero_grad()
            fit = 0.0
            for t in targets:
                out = block_out(hk, t["h"], t["ids"])[0].float()   # [T, S*D]
                want = t["out"][0].float()
                wpos = torch.full((out.shape[0], 1), args.keep, device="cuda")
                wpos[t["pos"]] = 1.0
                fit = fit + (wpos * (out - want) ** 2).sum() / t["scale"]
            fit = fit / len(targets)
            reg = args.lam * (w[:, None] * ref.rows.delta ** 2).mean() / s2
            (fit + reg).backward()
            opt.step()
        rows = bf16_rne((base0 + ref.rows.delta.detach()).cpu().numpy())
        info = {"fit": round(float(fit), 4), "reg": round(float(reg), 4), "rows": int(len(sel)),
                "seconds": round(time.time() - t0, 1)}
        ref.rows.choose(np.zeros(0, np.int64))
        return sel, rows, info

    for c in data:
        if len(results) >= args.cases:
            break
        rw = c["requested_rewrite"]
        subj = rw["subject"]
        new = enc(" " + rw["target_new"]["str"]).ids
        true = enc(" " + rw["target_true"]["str"]).ids
        texts = [rw["prompt"].format(subj), c["paraphrase_prompts"][0]]
        train = [enc(s) for s in texts]
        pos = [subject_positions(e, s, subj) for e, s in zip(train, texts)]
        if not all(pos) or len(c["paraphrase_prompts"]) < 2:
            continue
        sets = {"efficacy": [e.ids for e in train], "generalisation": [enc(s).ids for s in c["paraphrase_prompts"][1:]],
                "specificity": [enc(s).ids for s in c["neighborhood_prompts"][:5]]}
        before = evaluate(sets, new, true)
        # A: per-position copies of the chosen rows (a fact's own, decoupled across positions and prompts), trained
        # end to end; the block's outputs they give are the targets (reachable by the block by construction)
        heads, dims = ref.rows.g.heads, ref.rows.g.dims
        masks, rows_of = [], []
        for e, p in zip(train, pos):
            ti = ref.rows.ids(np.array(e.ids))
            m = torch.zeros(len(e.ids), heads, dims, device="cuda")
            for t in p:
                if len(p) == 1:
                    m[t] = 1
                    rows_of.append(ti[t])
                    continue
                if t - 1 in p:
                    m[t, :8] = 1
                    rows_of.append(ti[t, :8])
                if t - 1 in p and t - 2 in p:
                    m[t, 8:] = 1
                    rows_of.append(ti[t, 8:])
            masks.append(m.view(len(e.ids), -1))
        if not rows_of:
            continue
        fact_rows = np.unique(np.concatenate(rows_of))
        if args.targets in ("shared", "hybrid"):
            # the fact's rows themselves (one value a row, as v1 trains them), from the rows as they stand; the
            # block outputs they give are the targets: one set of rows gave them all, so a solve can reach them
            ref.rows.choose(fact_rows)
            w = torch.tensor(1.0 + counts[fact_rows].astype(np.float64) / 39.0, device="cuda", dtype=torch.float32)
            b0 = (torch.from_numpy(ref.rows.base_bits(fact_rows).astype(np.int16)).to("cuda").to(torch.int32) << 16).view(torch.float32)
            s2 = float((b0 ** 2).mean())
            opt = torch.optim.Adam([ref.rows.delta], lr=args.lr)
            t0 = time.time()
            steps, margins = 0, []
            for step in range(1, args.steps + 1):
                opt.zero_grad()
                nll = sum(-lp(e.ids, new, grad=True) for e in train) / len(train)
                (nll + args.lam * (w[:, None] * ref.rows.delta ** 2).mean() / s2).backward()
                opt.step()
                steps = step
                with torch.no_grad():
                    margins = [lp(e.ids, new).item() - lp(e.ids, true).item() for e in train]
                if min(margins) > args.margin:
                    break
            target_s = time.time() - t0
            mine = []
            for e, p in zip(train, pos):
                hk.capture = {}
                with torch.no_grad():
                    ref.hidden(e.ids)
                capd = hk.capture
                hk.capture = None
                with torch.no_grad():
                    kept = ref.rows.delta.data.clone()
                    ref.rows.delta.data.zero_()
                    base_out = block_out(hk, capd["h"], capd["ids"]).float()
                    ref.rows.delta.data.copy_(kept)
                change = float(((capd["out"].float() - base_out) ** 2).sum())
                mine.append({"h": capd["h"], "ids": capd["ids"], "out": capd["out"], "pos": p, "rows": fact_rows,
                             "scale": max(change, 1e-6)})
            with torch.no_grad():
                trained = bf16_rne((b0 + ref.rows.delta.detach()).cpu().numpy())
            ref.rows.choose(np.zeros(0, np.int64))
        else:
            copies = [torch.zeros_like(m, requires_grad=True) for m in masks]
            opt = torch.optim.Adam(copies, lr=args.lr)
            t0 = time.time()
            steps, margins = 0, []

            def with_copies(e, m, cp, n_extra):
                inst = torch.zeros(len(e.ids) + n_extra, m.shape[1], device="cuda")
                inst[: len(e.ids)] = cp * m
                ref.rows.inst = inst

            for step in range(1, args.steps + 1):
                opt.zero_grad()
                loss = 0.0
                for e, m, cp in zip(train, masks, copies):
                    with_copies(e, m, cp, len(new))
                    loss = loss - lp(e.ids, new, grad=True) / len(train)
                ref.rows.inst = None
                loss.backward()
                opt.step()
                steps = step
                with torch.no_grad():
                    margins = []
                    for e, m, cp in zip(train, masks, copies):
                        with_copies(e, m, cp, max(len(new), len(true)))
                        margins.append(lp(e.ids, new).item() - lp(e.ids, true).item())
                    ref.rows.inst = None
                if min(margins) > args.margin:
                    break
            target_s = time.time() - t0
            # the targets: the block's inputs and its output with the copies, over the prompt
            mine = []
            for e, p, m, cp in zip(train, pos, masks, copies):
                with_copies(e, m, cp.detach(), 0)
                hk.capture = {}
                with torch.no_grad():
                    ref.hidden(e.ids)
                capd = hk.capture
                hk.capture = None
                ref.rows.inst = None
                with torch.no_grad():
                    base_out = block_out(hk, capd["h"], capd["ids"]).float()  # the rows as they stand, no copies
                change = float(((capd["out"].float() - base_out) ** 2).sum())
                mine.append({"h": capd["h"], "ids": capd["ids"], "out": capd["out"], "pos": p, "rows": fact_rows,
                             "scale": max(change, 1e-6)})
        # B: solve every kept fact's targets together (sequential) or this fact's alone
        if args.targets == "hybrid":
            shared_rows = np.array([i for i in fact_rows if int(i) in ref.rows.committed], np.int64) if args.sequential else np.zeros(0, np.int64)
            ref.rows.commit(fact_rows, trained)            # the fact's own rows as trained
            if len(shared_rows):
                touching = [x for x in stored if np.isin(x["rows"], shared_rows).any()] + mine
                sel, rows, info = solve(touching, free=shared_rows)
                ref.rows.commit(sel, rows)
                info["shared"] = int(len(shared_rows))
                info["facts"] = len({id(x["rows"]) for x in touching})
            else:
                info = {"fit": 0.0, "reg": 0.0, "rows": int(len(fact_rows)), "seconds": 0.0, "shared": 0}
        else:
            targets = (stored + mine) if args.sequential else mine
            sel, rows, info = solve(targets)
            ref.rows.commit(sel, rows)
        after = evaluate(sets, new, true)
        r = {"case_id": c["case_id"], "subject": subj, "target_steps": steps, "target_seconds": round(target_s, 1),
             "solve": info, "before": before, "after": after, "final_margins": [round(m, 2) for m in margins]}
        results.append(r)
        print(json.dumps(r), flush=True)
        if args.sequential:
            stored.extend(mine)
            done.append((sets, new, true))
            if len(results) % args.eval_every == 0:
                with torch.no_grad():
                    kept = [evaluate({k: v for k, v in st.items() if k != "specificity"}, nw, tr) for st, nw, tr in done]
                    ctrl = float(np.mean([lp(p, tr).item() > lp(p, nw).item() for p, nw, tr in control]))
                point = {"edits": len(results), "rows": info["rows"],
                         "earlier_efficacy": round(float(np.mean([k["efficacy"] for k in kept])), 3),
                         "earlier_generalisation": round(float(np.mean([k["generalisation"] for k in kept])), 3),
                         "control_true_kept": round(ctrl, 3), "control_true_kept_before": ctrl0}
                timeline.append(point)
                print("SEQ", json.dumps(point), flush=True)
        else:
            ref.rows.committed = {}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"results": results, "timeline": timeline}, indent=1))
    keys = ("efficacy", "generalisation", "specificity")
    summary = {f"{w}_{k}": round(float(np.mean([r[w][k] for r in results])), 3) for w in ("before", "after") for k in keys}
    summary.update(cases=len(results), mean_target_steps=round(float(np.mean([r["target_steps"] for r in results])), 1),
                   mean_target_seconds=round(float(np.mean([r["target_seconds"] for r in results])), 1),
                   mean_solve_seconds=round(float(np.mean([r["solve"]["seconds"] for r in results])), 1))
    print(json.dumps(summary, indent=1))
    args.out.write_text(json.dumps({"summary": summary, "results": results, "timeline": timeline,
                                    "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))


if __name__ == "__main__":
    main()
