"""Worker smoke test: load, a scored case, the same case under an overlay of its own rows, a base overlay, timing."""

import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from ngram_ids import NGram
from ngram_table import Table
from worker import Worker
import paths

M = paths.MODEL
tok = Tokenizer.from_file(str(M / "tokenizer.json"))
g, t = NGram.from_model(M), Table(M)
p = tok.encode("The mother tongue of Danielle Darrieux is", add_special_tokens=False).ids
c = tok.encode(" French", add_special_tokens=False).ids
ids = np.unique(g.ids(g.initial_history(), np.array(p)))
with Worker() as w:
    print(f"ready in {w.load_seconds:.1f}s")
    t0 = time.time(); a = w.case(prompt=p, cont=c, gen=8); t1 = time.time()
    print("plain", a["sum"], a["gen"], f"{t1 - t0:.2f}s")
    b = w.case(prompt=p, cont=c, overlay={"ids": ids.tolist(), "rows": t.rows(ids).tolist()})
    print("own rows as overlay", b["sum"], "identical" if b["logprobs"] == a["logprobs"] else "DIFFERENT")
    print(w.base(overlay={"ids": ids.tolist(), "rows": (t.rows(ids) ^ 0x8000).tolist()}))
    d = w.case(prompt=p, cont=c)
    print("negated base overlay", d["sum"], d["base_rows"])
    print(w.base(clear=True))
    n = 20; t0 = time.time()
    for _ in range(n): w.case(prompt=p, cont=c)
    print(f"{(time.time() - t0) / n * 1000:.0f} ms a scored case ({len(p)} prompt + {len(c)} continuation tokens)")
