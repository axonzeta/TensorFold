"""A streamed PyTorch reference of Flash Next for gradients into its n-gram rows.

The HF `qwen4_exp` text model runs as is, except: each decoder layer is built on the meta device and loaded from the
checkpoint when it runs (activation checkpointing rebuilds it for the backward, so one layer's weights live at a
time); routed experts are dequantised only when routed to; layer 1's n-gram lookup returns the table's rows plus a
trainable fp32 delta on chosen rows. Everything else is frozen.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

from ngram_ids import NGram  # noqa: E402
from ngram_table import Table  # noqa: E402
from ref.ckpt import Ckpt  # noqa: E402

P = "model.language_model."


def text_config(model: Path) -> Qwen4ExpTextConfig:
    raw = json.loads((model / "config.json").read_text())
    c = Qwen4ExpTextConfig(**raw.get("text_config", raw))
    c._attn_implementation = "sdpa"
    return c


class Rows(nn.Module):
    """Layer 1's n-gram lookup: the table's rows (as the engine decodes them) plus `delta` on the chosen rows."""

    def __init__(self, model: Path, device: str):
        super().__init__()
        self.g = NGram.from_model(model)
        self.table = Table(model)
        self.device = device
        self.committed: dict[int, np.ndarray] = {}  # row id -> bf16 bits of an earlier edit (sequential editing)
        self.inst: torch.Tensor | None = None  # [T, heads * dims] added to this call's rows by position (v2 targets)
        self.choose(np.zeros(0, np.int64))

    def base_bits(self, ids) -> np.ndarray:
        """bf16 bits [len(ids), dims]: the table's rows, with committed edits in place."""
        bits = self.table.rows(ids)
        if self.committed:
            for k, i in enumerate(np.asarray(ids).reshape(-1)):
                c = self.committed.get(int(i))
                if c is not None:
                    bits[k] = c
        return bits

    def commit(self, ids, bits: np.ndarray) -> None:
        for i, b in zip(np.asarray(ids).reshape(-1), bits):
            self.committed[int(i)] = np.asarray(b, np.uint16)

    def choose(self, ids: np.ndarray) -> None:
        """Make `ids` (global row ids) trainable, their delta zero."""
        self.sel = np.unique(np.asarray(ids, np.int64))
        self.where = {int(r): k for k, r in enumerate(self.sel)}
        self.delta = nn.Parameter(torch.zeros(len(self.sel), self.g.dims, device=self.device))

    def ids(self, tokens: np.ndarray) -> np.ndarray:
        return self.g.ids(self.g.initial_history(), tokens)

    def forward(self, input_ids: torch.Tensor, past_key_values=None) -> torch.Tensor:
        b, t = input_ids.shape
        out = []
        for row in input_ids.cpu().numpy():
            ids = self.ids(row).reshape(-1)  # [t * 16]
            base = torch.from_numpy(self.base_bits(ids).astype(np.int16)).to(self.device)
            rows = (base.to(torch.int32) << 16).view(torch.float32)  # bf16 bits -> fp32
            pick = torch.tensor([self.where.get(int(i), -1) for i in ids], device=self.device)
            hit = pick >= 0
            if hit.any():
                rows = rows.clone()
                rows[hit] = rows[hit] + self.delta[pick[hit]]
            rows = rows.view(t, -1)
            if self.inst is not None:
                rows = rows + self.inst[:t].to(rows.dtype)
            out.append(rows)
        return torch.stack(out).to(torch.bfloat16)


class LazyExperts(nn.Module):
    """HF's routed experts, each dequantised from GPTQ int4 the first time a token is routed to it."""

    def __init__(self, ck: Ckpt, layer: int, num_experts: int, act):
        super().__init__()
        self.ck, self.layer, self.num_experts, self.act = ck, layer, num_experts, act

    def weights(self, e: int):
        p = f"{P}layers.{self.layer}.mlp.experts.{e}."
        return self.ck.gptq(p + "gate_proj"), self.ck.gptq(p + "up_proj"), self.ck.gptq(p + "down_proj")

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(top_k_index).tolist():
            tok, slot = torch.where(top_k_index == e)
            gate, up, down = self.weights(e)
            x = hidden_states[tok]
            y = nn.functional.linear(self.act(nn.functional.linear(x, gate)) * nn.functional.linear(x, up), down)
            out.index_add_(0, tok, (y * top_k_weights[tok, slot, None]).to(out.dtype))
        return out


class Streamed(nn.Module):
    """Decoder layer `idx`, built and loaded when it runs; recomputed (rebuilt) for the backward."""

    def __init__(self, ref: "Reference", idx: int):
        super().__init__()
        self.ref = [ref]  # not a submodule
        self.idx = idx

    def forward(self, hidden_states, **kw):
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            return checkpoint(self._run, hidden_states, use_reentrant=False, **kw)
        return self._run(hidden_states, **kw)

    def _run(self, hidden_states, **kw):
        return self.ref[0].build(self.idx)(hidden_states, **kw)


class Reference:
    def __init__(self, model: Path, device: str = "cuda"):
        self.model = Path(model)
        self.device = device
        self.ck = Ckpt(self.model, device)
        self.cfg = text_config(self.model)
        self.rows = Rows(self.model, device)
        with torch.device("meta"):
            tm = M.Qwen4ExpTextModel(self.cfg)
        tm.embed_tokens = nn.Embedding.from_pretrained(self.ck.bf16(P + "embed_tokens.weight"), freeze=True)
        tm.layers = nn.ModuleList([Streamed(self, i) for i in range(self.cfg.num_hidden_layers)])
        self._load(tm.hyper_connection_mixer, P + "hyper_connection_mixer.")
        tm.rotary_emb = M.Qwen4ExpTextRotaryEmbedding(config=self.cfg, device=device)
        left = [n for n, p in list(tm.named_parameters()) + list(tm.named_buffers()) if p.is_meta]
        if left:
            raise RuntimeError(f"reference: tensors left unloaded: {left[:8]}")
        self.tm = tm
        self.head = self.ck.weight("lm_head")  # bf16 [vocab, hidden]

    def _load(self, module: nn.Module, prefix: str, skip: tuple[str, ...] = ()) -> None:
        sd = {}
        for name, t in list(module.named_parameters()) + list(module.named_buffers()):
            if name.startswith(skip):
                continue
            key = prefix + name
            if self.ck.has(key):
                sd[name] = self.ck.bf16(key) if t.is_floating_point() else self.ck.raw(key)
            elif name.endswith(".weight") and self.ck.has(key[: -len(".weight")] + ".qweight"):
                sd[name] = self.ck.gptq(key[: -len(".weight")])
            else:
                raise KeyError(f"reference: no checkpoint tensor for {key}")
        module.load_state_dict(sd, assign=True, strict=False)
        for name, p in module.named_parameters():
            if not name.startswith(skip):
                p.requires_grad_(False)

    def build(self, idx: int) -> nn.Module:
        with torch.device("meta"):
            layer = M.Qwen4ExpTextDecoderLayer(self.cfg, idx)
        layer.mlp.experts = LazyExperts(self.ck, idx, self.cfg.num_experts, layer.mlp.experts.act_fn)
        skip = ("mlp.experts.",)
        if layer.ple is not None:
            layer.ple.ple_embedding = self.rows
            skip += ("ple.ple_embedding.",)
        self._load(layer, f"{P}layers.{idx}.", skip)
        return layer

    def hidden(self, tokens: list[int]) -> torch.Tensor:
        ids = torch.tensor([tokens], device=self.device)
        return self.tm(input_ids=ids, use_cache=False).last_hidden_state[0]  # [T, hidden]

    def answer_logprob(self, prompt: list[int], answer: list[int]) -> torch.Tensor:
        """Sum of log p(answer tokens), teacher-forced after the prompt (differentiable)."""
        h = self.hidden(list(prompt) + list(answer)[:-1])
        rows = h[len(prompt) - 1:]
        logits = (rows @ self.head.t()).float()
        return torch.log_softmax(logits, -1)[torch.arange(len(answer)), torch.tensor(answer, device=self.device)].sum()
