"""Flash Next's n-gram memory row ids on the host: the 16 table rows each token position looks up.

Matches TensorFold python-0.6 `qwen4_exp/model.py` NGramEmbedding.ids and MiaAI-Lab's `cuda_ngram.zig` bit for bit:
bigram heads 0-7, trigram heads 8-15, wrapping int64 product-xor of the last tokens (EOS resets them), floor-mod each
head's prime size, plus its offset.

    uv run ngram_ids.py --model DIR --text "My sister is called Ana."
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_PRIME = 10007
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding."


def _splitmix64(value: int) -> int:
    value = (value + _GOLDEN) & _MASK64
    value = ((value ^ (value >> 30)) * _MIX1) & _MASK64
    value = ((value ^ (value >> 27)) * _MIX2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass(frozen=True)
class NGram:
    """The hash constants of one n-gram memory site."""

    n: int
    per_ngram: int
    eos: int
    sizes: np.ndarray  # int64 [heads]
    offsets: np.ndarray  # int64 [heads]
    multipliers: np.ndarray  # int64 [n]
    rows: int  # table rows, padded to the divisor
    shard_rows: int
    dims: int

    @property
    def heads(self) -> int:
        return (self.n - 1) * self.per_ngram

    @staticmethod
    def derive(vocab: int = 248320, n: int = 3, per_ngram: int = 8, base: int = 20_000_000, divisor: int = 128,
               shards: int = 128, seed: int = 1234, eos: int = 248044, embed_dim: int = 2560,
               ple_index: int = 0) -> NGram:
        heads = (n - 1) * per_ngram
        sizes, offsets, total = [], [], 0
        for h in range(heads):
            size = _nth_prime_after(base - 1, ple_index * heads + h + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        half = max(1, (((1 << 63) - 1) // max(vocab, 1)) // 2)
        start = seed + _PRIME * ple_index
        mult = [2 * (_splitmix64((start + _GOLDEN * (i + 1)) & _MASK64) % half) + 1 for i in range(n)]
        rows = math.ceil(total / divisor) * divisor
        if rows % shards:
            raise ValueError("uneven shards are not handled")
        return NGram(n, per_ngram, eos, np.array(sizes, np.int64), np.array(offsets, np.int64),
                     np.array(mult, np.int64), rows, rows // shards, embed_dim // heads)

    @staticmethod
    def from_model(model: Path) -> NGram:
        """Derived from the checkpoint's config, then checked against the constants the checkpoint ships."""
        from safetensors import safe_open

        cfg = json.loads((model / "config.json").read_text())
        t = cfg.get("text_config", cfg)
        g = NGram.derive(vocab=t["vocab_size"], n=t["ngram_size"], per_ngram=t["heads_per_ngram"],
                         base=t["ngram_vocab_size_base"], divisor=t["make_ngram_vocab_size_divisible_by"],
                         shards=t["split_ngram_parts"], seed=t.get("seed", 1234), eos=t["eos_token_id"],
                         embed_dim=t["ple_embed_dim"])
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        shipped = {"layer_multipliers": g.multipliers, "ngram_heads_vocab_sizes": g.sizes,
                   "ngram_heads_offsets": g.offsets}
        for name, want in shipped.items():
            key = PLE_PREFIX + name
            with safe_open(str(model / index[key]), "numpy") as f:
                got = f.get_tensor(key).astype(np.int64).reshape(-1)
            if not np.array_equal(got, want):
                raise ValueError(f"{key}: checkpoint {got.tolist()} != derived {want.tolist()}")
        return g

    def initial_history(self) -> np.ndarray:
        return np.full(self.n - 1, self.eos, np.int64)

    def ids(self, history: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Row ids [L, heads] for `tokens` [L] after `history` [n-1]."""
        seq = np.concatenate([np.asarray(history, np.int64), np.asarray(tokens, np.int64)])
        width = seq.shape[0]
        pos = np.arange(width)
        eos_at = np.where(seq == self.eos, pos, -1)
        before = np.concatenate([[-1], np.maximum.accumulate(eos_at)[:-1]])
        in_segment = pos - (before + 1)
        shifted = []
        for shift in range(self.n):
            src = pos - shift
            taken = seq[np.maximum(src, 0)]
            shifted.append(np.where((in_segment >= shift) & (src >= 0), taken, self.eos))
        blocks = []
        with np.errstate(over="ignore"):
            for order in range(2, self.n + 1):
                first = (order - 2) * self.per_ngram
                mixed = shifted[0] * self.multipliers[0]
                for p in range(1, order):
                    mixed = np.bitwise_xor(mixed, shifted[p] * self.multipliers[p])
                sizes = self.sizes[first:first + self.per_ngram]
                blocks.append(mixed[:, None] % sizes + self.offsets[first:first + self.per_ngram])
        return np.concatenate(blocks, axis=-1)[-len(tokens):]

    def shard_of(self, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return row // self.shard_rows, row % self.shard_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True, help="checkpoint snapshot directory")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    g = NGram.from_model(args.model)
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    enc = tok.encode(args.text, add_special_tokens=False)
    ids = g.ids(g.initial_history(), np.array(enc.ids))
    for t, (piece, tid) in enumerate(zip(enc.tokens, enc.ids)):
        print(f"{t:4d} {tid:7d} {piece!r:20} bi {ids[t, :8].tolist()}  tri {ids[t, 8:].tolist()}")


if __name__ == "__main__":
    main()
