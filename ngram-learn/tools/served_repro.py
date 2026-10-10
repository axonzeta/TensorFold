"""Repro: one chat lesson through the worker's grad op, served capture on, then off."""
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from ngram_ids import NGram
from worker import Worker
import paths

M = paths.MODEL
tok = Tokenizer.from_file(str(M / "tokenizer.json"))
g = NGram.from_model(M)
p = tok.encode("<|im_start|>user\nWhen does Project Kestrel launch?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
a = tok.encode("Project Kestrel launches on 3 November.<|im_end|>", add_special_tokens=False).ids
toks = p + a[:-1]
sel = np.unique(g.ids(g.initial_history(), np.array(toks))[5:8].reshape(-1))
print("rows", len(toks), "sel", len(sel), flush=True)
with Worker() as w:
    for served in [s == "1" for s in (sys.argv[1] if len(sys.argv) > 1 else "01")]:
        r = w.send({"op": "grad", "tokens": toks, "at": list(range(len(p) - 1, len(toks))), "targets": a, "sel": sel.tolist(), "served": served})
        print("served", served, "loss", r["loss"], flush=True)
