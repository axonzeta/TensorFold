"""Reference smoke test: one fact's answer log-prob against the engine's, then one backward into its subject rows."""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref.model import Reference  # noqa: E402

M = Path("/hf/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/1464274120d36a4d8fcaa934552334a7d83ce0fd")
tok = Tokenizer.from_file(str(M / "tokenizer.json"))
t0 = time.time()
ref = Reference(M)
print(f"reference ready in {time.time() - t0:.1f}s, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
p = tok.encode("The mother tongue of Danielle Darrieux is", add_special_tokens=False).ids
a = tok.encode(" French", add_special_tokens=False).ids
with torch.no_grad():
    t0 = time.time()
    lp = ref.answer_logprob(p, a).item()
    print(f"log p(' French') = {lp:.4f} (engine -0.0897) in {time.time() - t0:.1f}s", flush=True)
subj = [3, 4, 5, 6, 7]
ids = ref.rows.ids(np.array(p))
ref.rows.choose(ids[[i for i in subj if i < len(p)]].reshape(-1))
t0 = time.time()
loss = -ref.answer_logprob(p, tok.encode(" English", add_special_tokens=False).ids)
loss.backward()
torch.cuda.synchronize()
g = ref.rows.delta.grad
print(f"loss {loss.item():.4f}; backward {time.time() - t0:.1f}s; {len(ref.rows.sel)} rows; grad norm {g.norm():.4e}; "
      f"per-row max {g.norm(dim=1).max():.3e} min {g.norm(dim=1).min():.3e}; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print("tokens", [tok.id_to_token(i) for i in p])
