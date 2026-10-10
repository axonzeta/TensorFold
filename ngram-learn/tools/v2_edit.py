"""Learner v2 prototype (EngramEdit-style) on the PyTorch reference, beside grad_edit.py's v1 (run in the dev container).

v1 trains a fact's rows end to end and commits them; a later fact that shares rows starts from (and moves) them.
v2 splits the two:

  A. Target: a change delta at the n-gram block's output, at each subject position of each training prompt (one
     [streams * hidden] vector a position), trained end to end until the new answer wins by --margin. Stored with
     the block's inputs there (the layer's stream h and the token ids), not the rows.
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
    ap.add_argument("--steps", type=int, default=25, help="target steps a fact")
    ap.add_argument("--lr", type=float, default=0.05, help="Adam on the target deltas (relative to the block output's size)")
    ap.add_argument("--target-l2", type=float, default=1e-3, help="penalty on the targets' size")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--solve-steps", type=int, default=300)
    ap.add_argument("--solve-lr", type=float, default=5e-3)
    ap.add_argument("--lam", type=float, default=0.05, help="row-change penalty, frequency-weighted, as grad_edit.py")
    ap.add_argument("--keep", type=float, default=1.0, help="weight of the off-subject positions keeping their output")
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
    data = pq.read_table(args.data).slice(args.start, args.cases * 3).to_pylist()
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

    def solve(targets: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
        """The rows all `targets` ask for, from the table's rows (committed edits cleared first)."""
        sel = np.unique(np.concatenate([t["rows"] for t in targets]))
        ref.rows.committed = {}
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
                want = t["out"][0].float().clone()
                want[t["pos"]] += t["delta"]
                wpos = torch.full((out.shape[0], 1), args.keep, device="cuda")
                wpos[t["pos"]] = 1.0
                fit = fit + (wpos * (out - want) ** 2).sum() / (t["delta"] ** 2).sum().clamp_min(1e-6)
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
        # A: targets at the block's output (the rows as committed so far), one prompt at a time inside each step
        caps = []
        for e in train:
            hk.capture = {}
            with torch.no_grad():
                lp(e.ids, new)
            caps.append(hk.capture)
            hk.capture = None
        scale = float(np.mean([cp["out"].float().norm(dim=-1).mean().item() for cp in caps]))
        deltas = [torch.zeros(len(p), caps[0]["out"].shape[-1], device="cuda", requires_grad=True) for p in pos]
        opt = torch.optim.Adam(deltas, lr=args.lr * scale / np.sqrt(caps[0]["out"].shape[-1]))
        t0 = time.time()
        steps, margins = 0, []
        for step in range(1, args.steps + 1):
            opt.zero_grad()
            loss = 0.0
            for e, p, d in zip(train, pos, deltas):
                inj = torch.zeros(len(e.ids) + len(new) - 1, d.shape[1], device="cuda")
                inj[p] = d
                hk.inject = inj
                loss = loss - lp(e.ids, new, grad=True) / len(train)
            hk.inject = None
            reg = args.target_l2 * sum((d ** 2).sum() for d in deltas) / scale ** 2
            (loss + reg).backward()
            opt.step()
            steps = step
            with torch.no_grad():
                margins = []
                for e, p, d in zip(train, pos, deltas):
                    inj = torch.zeros(len(e.ids) + max(len(new), len(true)) - 1, d.shape[1], device="cuda")
                    inj[p] = d
                    hk.inject = inj[: len(e.ids) + len(new) - 1]
                    a = lp(e.ids, new).item()
                    hk.inject = inj[: len(e.ids) + len(true) - 1]
                    b = lp(e.ids, true).item()
                    margins.append(a - b)
                hk.inject = None
            if min(margins) > args.margin:
                break
        target_s = time.time() - t0
        rows_of = []
        for e, p in zip(train, pos):
            ti = ref.rows.ids(np.array(e.ids))
            for t in p:
                if len(p) == 1:
                    rows_of.append(ti[t])
                    continue
                if t - 1 in p:
                    rows_of.append(ti[t, :8])
                if t - 1 in p and t - 2 in p:
                    rows_of.append(ti[t, 8:])
        if not rows_of:
            continue
        fact_rows = np.unique(np.concatenate(rows_of))
        # the block's inputs are the prompt only: the targets live at subject positions inside it
        mine = [{"h": cp["h"][:, : len(e.ids)], "ids": cp["ids"][:, : len(e.ids)], "out": cp["out"][:, : len(e.ids)],
                 "pos": p, "delta": d.detach(), "rows": fact_rows} for cp, e, p, d in zip(caps, train, pos, deltas)]
        # B: solve every kept fact's targets together (sequential) or this fact's alone
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
