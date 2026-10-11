"""General-knowledge drift: a fixed MMLU sample answered by the served engine, with or without an overlay
(TF_NGRAM_OVERLAY, read by the worker at load), as accuracy and as the answers that changed against a baseline run.

    uv run drift_eval.py --out ../experiments/07-scale/mmlu-base.json
    TF_NGRAM_OVERLAY=/work/experiments/07-scale/seq-1000.at500.overlay.safetensors \\
        uv run drift_eval.py --out ../experiments/07-scale/mmlu-at500.json --against ../experiments/07-scale/mmlu-base.json

Each question is put as one user turn (thinking off), the choices lettered A-D, "Answer with the letter only."; the
model's answer is the letter with the highest log-probability as the reply's first token.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import paths
from worker import Worker

TEMPLATE = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main() -> None:
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data/mmlu/all/test-00000-of-00001.parquet")
    ap.add_argument("--n", type=int, default=600, help="questions, spread evenly over the subjects (seed 0)")
    ap.add_argument("--against", type=Path, help="a baseline run: report the answers that changed")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(paths.MODEL / "tokenizer.json"))
    rows = pq.read_table(a.data).to_pylist()
    rng = np.random.default_rng(0)
    by: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(r["subject"], []).append(i)
    per = max(1, a.n // len(by))
    pick = sorted(i for s in sorted(by) for i in rng.choice(by[s], min(per, len(by[s])), replace=False))[: a.n]
    letters = ["A", "B", "C", "D"]
    letter_ids = [tok.encode(x, add_special_tokens=False).ids for x in letters]
    out = []
    with Worker() as w:
        for k, i in enumerate(pick):
            r = rows[i]
            q = r["question"].strip() + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, r["choices"])) + \
                "\nAnswer with the letter only."
            ids = tok.encode(TEMPLATE.format(q=q), add_special_tokens=False).ids
            lps = [w.case(prompt=ids, cont=lid)["sum"] for lid in letter_ids]
            out.append({"row": int(i), "subject": r["subject"], "answer": int(r["answer"]), "pred": int(np.argmax(lps)),
                        "logprobs": [round(x, 4) for x in lps]})
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(pick)} accuracy so far {np.mean([o['pred'] == o['answer'] for o in out]):.3f}", flush=True)
    acc = float(np.mean([o["pred"] == o["answer"] for o in out]))
    summary = {"questions": len(out), "accuracy": round(acc, 4), "overlay": os.environ.get("TF_NGRAM_OVERLAY", "")}
    if a.against:
        base = {o["row"]: o for o in json.loads(a.against.read_text())["answers"]}
        same = [o for o in out if o["row"] in base]
        changed = [o for o in same if o["pred"] != base[o["row"]]["pred"]]
        summary.update(compared=len(same), answers_changed=len(changed),
                       baseline_accuracy=round(float(np.mean([base[o["row"]]["pred"] == o["answer"] for o in same])), 4),
                       changed_to_right=sum(o["pred"] == o["answer"] for o in changed),
                       changed_to_wrong=sum(base[o["row"]]["pred"] == o["answer"] for o in changed))
    print(json.dumps(summary, indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "answers": out}, indent=1))


if __name__ == "__main__":
    main()
