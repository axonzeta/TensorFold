"""Flash Next's n-gram table rows on the host, as the engine decodes them, and overlay files for TF_NGRAM_OVERLAY.

Rows are read from the checkpoint's shards (FP8 e4m3 with one table scale, or bf16) and decoded to bf16 bits exactly as
`NgramTable.gather` does: bf16_rne(e4m3 x scale) through a 256-entry LUT, NaN codes to 0x7FC0.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

PREFIX = "model.language_model.layers.1.ple.ple_embedding."


def _e4m3_to_f32(b: int) -> float:
    e, m, sign = (b >> 3) & 0xF, b & 0x7, -1.0 if b & 0x80 else 1.0
    if e == 15 and m == 7:
        return float("nan")
    if e == 0:
        return sign * m / 512.0
    return sign * (1 + m / 8) * 2.0 ** (e - 7)


def bf16_rne(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bits, round to nearest even, NaN -> 0x7FC0."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(x), np.uint16(0x7FC0), r)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, np.uint32) << 16).view(np.float32)


def _header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Table:
    """The n-gram shards memory-mapped; `rows(ids)` -> bf16 bits [len(ids), width]."""

    def __init__(self, model: Path):
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        files = {model / f for f in index.values()} | set((model / "ple-table").glob("*.safetensors"))
        shards: dict[int, tuple[Path, dict, int]] = {}
        self.scale = 1.0
        for path in sorted(files):
            head, data = _header(path)
            for name, e in head.items():
                if not name.startswith(PREFIX + "ngram_embedding."):
                    continue
                leaf = name[len(PREFIX + "ngram_embedding."):]
                if leaf == "weight_scale":
                    raw = open(path, "rb").read()[data + e["data_offsets"][0]:data + e["data_offsets"][1]]
                    v = np.frombuffer(raw, np.float32) if e["dtype"] == "F32" else bf16_to_f32(np.frombuffer(raw, np.uint16))
                    self.scale = float(v[0])
                    continue
                parts = leaf.replace("shards.", "shard_").split(".")
                if parts[-1] != "weight":
                    continue
                shards[int(parts[0].split("_")[1])] = (path, e, data)
        self.shards = []
        self.starts = [0]
        for i in range(len(shards)):
            path, e, data = shards[i]
            rows, width = e["shape"]
            dtype = {"F8_E4M3": np.uint8, "BF16": np.uint16}[e["dtype"]]
            self.fp8 = e["dtype"] == "F8_E4M3"
            self.width = width
            self.shards.append(np.memmap(path, dtype=dtype, mode="r", offset=data + e["data_offsets"][0],
                                         shape=(rows, width)))
            self.starts.append(self.starts[-1] + rows)
        self.lut = bf16_rne(np.array([_e4m3_to_f32(c) for c in range(256)], np.float32) * np.float32(self.scale))
        self.lut[[0x7F, 0xFF]] = 0x7FC0

    def rows(self, ids) -> np.ndarray:
        ids = np.asarray(ids, np.int64).reshape(-1)
        out = np.empty((ids.size, self.width), np.uint16)
        per = self.starts[1] - self.starts[0]
        for k, i in enumerate(ids):
            s = min(int(i) // per, len(self.shards) - 1)
            raw = self.shards[s][int(i) - self.starts[s]]
            out[k] = self.lut[raw] if self.fp8 else raw
        return out


def write_overlay(path: Path, ids, rows_bf16: np.ndarray) -> None:
    """An overlay file: `ids` I64 [N], `rows` BF16 [N, width] (bits)."""
    ids = np.ascontiguousarray(np.asarray(ids, np.int64))
    rows = np.ascontiguousarray(np.asarray(rows_bf16, np.uint16))
    a, b = ids.tobytes(), rows.tobytes()
    head = {"ids": {"dtype": "I64", "shape": [ids.size], "data_offsets": [0, len(a)]},
            "rows": {"dtype": "BF16", "shape": list(rows.shape), "data_offsets": [len(a), len(a) + len(b)]}}
    h = json.dumps(head).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + a + b)
