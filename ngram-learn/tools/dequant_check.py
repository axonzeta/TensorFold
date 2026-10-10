"""B1 check: the engine's identity-matmul Wᵀ dumps against tools/ref/ckpt.py's dequant of the checkpoint tensors."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref.ckpt import Ckpt  # noqa: E402

M = Path("/hf/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/1464274120d36a4d8fcaa934552334a7d83ce0fd")
D = Path(sys.argv[1])
P = "model.language_model.layers."
PARTS = {"gdn_proj8": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], "gdn_out8": ["linear_attn.out_proj"],
         "attn_proj8": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "attn_o8": ["self_attn.o_proj"],
         "shared_gu8": ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"], "shared_down8": ["mlp.shared_expert.down_proj"]}
ck = Ckpt(M)
faces = json.loads((D / "faces.json").read_text())
for name, f in faces.items():
    got = torch.from_numpy(np.fromfile(D / f"{name}.bin", dtype=np.int16).astype(np.int16)).view(torch.bfloat16)
    if name == "head8192":
        ref = ck.gptq("lm_head")[: f["n"]]
        got = got.view(f["n"], f["k"]).to(ref.device)
    elif "_expert" in name:
        layer, rest = name.split("_", 1)
        ex, proj = rest[len("expert"):].split("_")
        ref = ck.gptq(f"{P}{int(layer[1:])}.mlp.experts.{ex}.{proj}_proj")
        got = got.view(f["n"], f["k"]).to(ref.device)
    else:
        layer, kind = name.split("_", 1)
        i = int(layer[1:])
        ref = torch.cat([ck.bf16(f"{P}{i}.{p}.weight") for p in PARTS[kind]], 0)  # [n, k]
        got = got.view(f["k"], f["n"]).t().to(ref.device)
    same = (got.view(torch.int16) == ref.view(torch.int16)).float().mean().item() if got.shape == ref.shape else -1
    err = (got.float() - ref.float()).abs().max().item() if got.shape == ref.shape else float("nan")
    print(f"{name:20} engine {tuple(got.shape)} ref {tuple(ref.shape)} bit-equal {same:.6f} max|diff| {err:.3e} "
          f"({f['seconds']:.3f}s)")
