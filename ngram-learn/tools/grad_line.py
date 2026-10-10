"""Print the repro's grad request as one JSON line (served: argv[1] == "1")."""
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from ngram_ids import NGram
import paths

M = paths.MODEL
tok = Tokenizer.from_file(str(M / "tokenizer.json"))
g = NGram.from_model(M)
p = tok.encode("<|im_start|>user\nWhen does Project Kestrel launch?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
a = tok.encode("Project Kestrel launches on 3 November.<|im_end|>", add_special_tokens=False).ids
toks = p + a[:-1]
sel = np.unique(g.ids(g.initial_history(), np.array(toks))[5:8].reshape(-1))
print(json.dumps({"op": "grad", "tokens": toks, "at": list(range(len(p) - 1, len(toks))), "targets": a, "sel": sel.tolist(), "served": sys.argv[1] == "1"}))
print('{"op":"quit"}')
