"""Gradient fidelity against the served model: the in-engine gradient's directional derivative, checked by finite
differences in the served engine (worker), for the training pass from its own forward and from a served capture.

    uv run served_grad_check.py --cases 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ngram_ids import NGram
from ngram_table import Table, bf16_rne, bf16_to_f32
from worker import Worker
import paths

MODEL = paths.MODEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data/counterfact/data/train-00000-of-00001-05d11247db7abce8.parquet")
    ap.add_argument("--cases", default="0,1,2,3,4")
    ap.add_argument("--eps", type=float, default=0.1, help="step along the unit gradient direction (over all elements)")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    g, table = NGram.from_model(MODEL), Table(MODEL)
    data = pq.read_table(args.data).to_pylist()
    out = []
    with Worker() as w:
        for cid in [int(x) for x in args.cases.split(",")]:
            rw = data[cid]["requested_rewrite"]
            prompt = tok.encode(rw["prompt"].format(rw["subject"]), add_special_tokens=False).ids
            ans = tok.encode(" " + rw["target_new"]["str"], add_special_tokens=False).ids
            toks = prompt + ans[:-1]
            sel = np.unique(g.ids(g.initial_history(), np.array(toks)).reshape(-1))
            base = bf16_to_f32(table.rows(sel)).astype(np.float64)
            ov = lambda d: {"ids": sel.tolist(), "rows": bf16_rne((base + d).astype(np.float32)).tolist()}  # noqa: E731
            loss = lambda d: -w.case(prompt=prompt, cont=ans, overlay=ov(d))["sum"]  # noqa: E731
            row = {"case": cid, "rows": int(len(sel)), "served_loss": loss(np.zeros_like(base))}
            grads = {}
            for mode in ("own", "served"):
                r = w.send({"op": "grad", "tokens": toks, "at": list(range(len(prompt) - 1, len(toks))), "targets": ans,
                            "sel": sel.tolist(), "served": mode == "served", "overlay": ov(np.zeros_like(base))})
                gr = np.asarray(r["grad"], np.float64)
                grads[mode] = gr
                u = gr / np.linalg.norm(gr)
                fd = (loss(args.eps * u) - loss(-args.eps * u)) / (2 * args.eps)
                row[mode] = {"pass_loss": round(r["loss"], 4), "norm": round(float(np.linalg.norm(gr)), 3),
                             "finite_difference": round(float(fd), 3), "ratio": round(float(fd / np.linalg.norm(gr)), 3)}
            a, b = grads["own"].ravel(), grads["served"].ravel()
            row["cosine_own_served"] = round(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 4)
            out.append(row)
            print(json.dumps(row), flush=True)
    for mode in ("own", "served"):
        print(mode, "mean ratio (finite difference / gradient norm):", round(float(np.mean([r[mode]["ratio"] for r in out])), 3))


if __name__ == "__main__":
    main()
