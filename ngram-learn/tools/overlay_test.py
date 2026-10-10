"""Phase 2 test overlay: the rows a prompt's n-grams look up, negated, as a TF_NGRAM_OVERLAY file."""

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from ngram_ids import NGram
from ngram_table import Table, bf16_rne, bf16_to_f32, write_overlay

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
g = NGram.from_model(args.model)
t = Table(args.model)
ids = np.unique(g.ids(g.initial_history(), np.array(Tokenizer.from_file(str(args.model / "tokenizer.json")).encode(args.text, add_special_tokens=False).ids)))
base = t.rows(ids)
f = bf16_to_f32(base)
print(f"table scale {t.scale} fp8 {t.fp8}; {ids.size} rows; |row| mean {np.abs(f).mean():.4f} max {np.abs(f).max():.3f}")
write_overlay(args.out, ids, bf16_rne(-f))
print(f"wrote {args.out}")
