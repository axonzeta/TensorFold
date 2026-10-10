import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref import model as RM  # noqa: E402

orig = RM.Streamed._run


def run(self, hidden_states, **kw):
    if self.idx in (2, 3):
        print("layer", self.idx, "grad" if torch.is_grad_enabled() else "nograd", tuple(hidden_states.shape),
              {k: (tuple(v.shape) if hasattr(v, "shape") else [tuple(x.shape) for x in v] if isinstance(v, tuple) else type(v).__name__) for k, v in kw.items()}, flush=True)
    return orig(self, hidden_states, **kw)


RM.Streamed._run = run
M = Path("/hf/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/1464274120d36a4d8fcaa934552334a7d83ce0fd")
ref = RM.Reference(M)
p = [785, 6683, 24364, 315, 70889, 8037, 22830, 2373, 374]
ref.rows.choose(ref.rows.ids(np.array(p))[4].reshape(-1))
loss = -ref.answer_logprob(p, [6060])
loss.backward()
print("ok", loss.item())
