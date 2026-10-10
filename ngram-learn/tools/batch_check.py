"""Checks a batched grad op against its lessons one at a time: the loss and every selected row's gradient of
{"lessons": [a, b, c]} must equal the sums over {"lessons": [a]}, {"lessons": [b]}, {"lessons": [c]} (fp32 sums in a
different order, so up to rounding) on the served path. The own-forward path is reported only: its MoE routes come
from the training forward, where batched GEMMs round differently and flip near-tied routes (as against the reference).

    uv run batch_check.py [--fact 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chat_edit import MODEL, TEMPLATE
from ngram_ids import NGram
from worker import Worker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact", type=int, default=0)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    g = NGram.from_model(MODEL)
    f = json.loads((Path(__file__).resolve().parent / "facts_suite.json").read_text())[a.fact]
    lessons, sel = [], set()
    for q, ans_text in f["train"]:
        ids = tok.encode(TEMPLATE.format(q=q), add_special_tokens=False).ids
        ans = tok.encode(ans_text + "<|im_end|>", add_special_tokens=False).ids
        lessons.append({"tokens": ids + ans[:-1], "at": list(range(len(ids) - 1, len(ids) + len(ans) - 1)), "targets": ans})
        sel.update(int(x) for x in g.ids(g.initial_history(), np.array(ids)).ravel()[-64:])
    sel = sorted(sel)
    ok = True
    with Worker() as w:
        for served in (True, False):
            one = [w.send({"op": "grad", "lessons": [x], "sel": sel, "served": served}) for x in lessons]
            both = w.send({"op": "grad", "lessons": lessons, "sel": sel, "served": served})
            loss = sum(r["loss"] for r in one)
            g1 = sum(np.asarray(r["grad"], np.float64) for r in one)
            g2 = np.asarray(both["grad"], np.float64)
            rel = float(np.linalg.norm(g2 - g1) / max(np.linalg.norm(g1), 1e-30))
            cos = float((g1 * g2).sum() / max(np.linalg.norm(g1) * np.linalg.norm(g2), 1e-30))
            good = abs(both["loss"] - loss) <= 1e-3 * abs(loss) and rel < 2e-2
            ok &= good or not served
            print(json.dumps({"served": served, "rows": sum(len(x["tokens"]) for x in lessons), "loss_sum": round(loss, 5),
                              "loss_batched": round(both["loss"], 5), "grad_rel_err": rel, "grad_cos": cos,
                              "ok": good}), flush=True)
    print("BATCH-OK" if ok else "BATCH-MISMATCH")


if __name__ == "__main__":
    main()
