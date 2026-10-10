"""The INT4-AutoRound checkpoint's tensors for the PyTorch reference, dequantised to bf16 on the GPU on demand.

Formats: BF16 / F32 / I64 as stored; F8_E4M3 with a 128x128 fp32 `weight_scale_inv`; GPTQ int4 (auto_round:auto_gptq,
symmetric, group 128, qzeros all 7 = zero point 8): w[out, in] = (q - 8) * scale. Names resolve through the main index,
then the side files the engine also reads (healed shared expert, int4 head), which win.
"""

from __future__ import annotations

import json
import mmap
import struct
import warnings
from collections import OrderedDict
from pathlib import Path

import torch

SIDE_FILES = ("model-healed-shared-expert.safetensors", "model-lmhead-int4.safetensors")
_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32,
       "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}


class Ckpt:
    def __init__(self, model: Path, device: str = "cuda", cache_gib: float = 48.0):
        self.model = Path(model)
        self.device = device
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = int(cache_gib * 2**30)
        where = dict(json.loads((self.model / "model.safetensors.index.json").read_text())["weight_map"])
        for side in SIDE_FILES:
            if (self.model / side).exists():
                for name in self._header(self.model / side)[0]:
                    if name != "__metadata__":
                        where[name] = side
        self.where = where
        self._files: dict[str, tuple[mmap.mmap, dict, int]] = {}

    @staticmethod
    def _header(path: Path) -> tuple[dict, int]:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n)), 8 + n

    def _file(self, name: str) -> tuple[mmap.mmap, dict, int]:
        fname = self.where[name]
        if fname not in self._files:
            path = self.model / fname
            head, data = self._header(path)
            with open(path, "rb") as f:
                self._files[fname] = (mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ), head, data)
        return self._files[fname]

    def has(self, name: str) -> bool:
        return name in self.where

    def raw(self, name: str) -> torch.Tensor:
        """The tensor as stored, on the device."""
        mm, head, data = self._file(name)
        e = head[name]
        a, b = e["data_offsets"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the map is read-only; the bytes are copied to the device at once
            u8 = torch.frombuffer(mm, dtype=torch.uint8, count=b - a, offset=data + a)
        return u8.to(self.device).view(_DT[e["dtype"]]).reshape(e["shape"])

    def _cached(self, key: str, make) -> torch.Tensor:
        """`make()` kept on the device, least recently used dropped past the budget."""
        t = self.cache.get(key)
        if t is not None:
            self.cache.move_to_end(key)
            return t
        t = make()
        n = t.numel() * t.element_size()
        if n <= self.cache_limit:
            self.cache[key] = t
            self.cache_bytes += n
            while self.cache_bytes > self.cache_limit:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= old.numel() * old.element_size()
        return t

    def bf16(self, name: str) -> torch.Tensor:
        """A weight dequantised to bf16 (FP8 block or as stored), cached."""
        return self._cached(name, lambda: self._bf16(name))

    def _bf16(self, name: str) -> torch.Tensor:
        t = self.raw(name)
        if t.dtype == torch.float8_e4m3fn:
            s = self.raw(name + "_scale_inv").float()
            out, inp = t.shape
            s = s.repeat_interleave(128, 0)[:out].repeat_interleave(128, 1)[:, :inp]
            return (t.float() * s).to(torch.bfloat16)
        return t.to(torch.bfloat16)

    def gptq(self, prefix: str) -> torch.Tensor:
        """GPTQ int4 weight `prefix`.{qweight, scales} -> bf16 [out, in], cached."""
        return self._cached(prefix + ".gptq", lambda: self._gptq(prefix))

    def _gptq(self, prefix: str) -> torch.Tensor:
        q = self.raw(prefix + ".qweight")  # int32 [in / 8, out]
        scales = self.raw(prefix + ".scales").float()  # [in / 128, out]
        shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int32)
        codes = ((q.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(-1, q.shape[1])  # [in, out]
        w = (codes.float() - 8.0) * scales.repeat_interleave(128, 0)[: codes.shape[0]]
        return w.t().contiguous().to(torch.bfloat16)

    def weight(self, prefix: str) -> torch.Tensor:
        """`prefix`.weight (bf16 / FP8) or a GPTQ `prefix`, as bf16 [out, in]."""
        if self.has(prefix + ".weight"):
            return self.bf16(prefix + ".weight")
        return self.gptq(prefix)
