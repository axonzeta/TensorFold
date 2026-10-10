"""B10: the in-engine training pass (tensorfold train-grad-check) against the PyTorch reference on one lesson.

    python train_grad_check.py case DIR [case_id]   # a CounterFact edit as a lesson: prompt + new answer
    (./probe.sh train-grad-check DIR)              # the engine: loss, every layer's input, d(rows)
    python train_grad_check.py ref DIR             # the reference: the same, every looked-up row trainable
    python train_grad_check.py cmp DIR
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
MODEL = Path("/hf/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/1464274120d36a4d8fcaa934552334a7d83ce0fd")
DATA = Path("/work/data/counterfact/data/train-00000-of-00001-05d11247db7abce8.parquet")


def case(d: Path, cid: int) -> None:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    c = pq.read_table(DATA).slice(cid, 1).to_pylist()[0]
    rw = c["requested_rewrite"]
    prompt = tok.encode(rw["prompt"].format(rw["subject"]), add_special_tokens=False).ids
    ans = tok.encode(" " + rw["target_new"]["str"], add_special_tokens=False).ids
    tokens = prompt + ans[:-1]
    at = list(range(len(prompt) - 1, len(tokens)))
    d.mkdir(parents=True, exist_ok=True)
    (d / "case.json").write_text(json.dumps({"tokens": tokens, "at": at, "targets": ans, "case_id": cid,
                                             "text": rw["prompt"].format(rw["subject"]) + " -> " + rw["target_new"]["str"]}))
    print(f"lesson: {len(tokens)} rows, {len(ans)} answer tokens: {rw['prompt'].format(rw['subject'])} -> {rw['target_new']['str']}")


def ref(d: Path) -> None:
    from ref import model as RM

    cs = json.loads((d / "case.json").read_text())
    tokens, at, targets = cs["tokens"], cs["at"], cs["targets"]
    reference = RM.Reference(MODEL)
    hidden = {}
    orig = RM.Streamed._run

    def run(self, h, **kw):
        if not torch.is_grad_enabled():
            hidden[self.idx] = h[0].float().cpu()
        return orig(self, h, **kw)
    RM.Streamed._run = run
    ids = reference.rows.ids(np.array(tokens))  # [R, 16]
    with torch.no_grad():
        ht = reference.hidden(tokens)
    RM.Streamed._run = orig
    reference.rows.choose(ids.reshape(-1))
    h = reference.hidden(tokens)
    logits = (h[at] @ reference.head.t()).float()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor(targets, device=logits.device), reduction="sum")
    loss.backward()
    torch.save({"loss": loss.item(), "hidden": [hidden[i] for i in range(len(hidden))], "sel": reference.rows.sel,
                "grad": reference.rows.delta.grad.cpu(), "ids": ids}, d / "ref.pt")
    print(f"reference loss {loss.item():.5f}")


def cmp(d: Path) -> None:
    r = torch.load(d / "ref.pt", weights_only=False)
    cs = json.loads((d / "case.json").read_text())
    R = len(cs["tokens"])
    el = json.loads((d / "engine_loss.json").read_text())
    print(f"loss: engine {el['loss']:.5f}  reference {r['loss']:.5f}   (engine pass {el['seconds']:.2f}s cold, {el['seconds_warm']:.2f}s warm)")
    hid = np.fromfile(d / "engine_hidden.bin", dtype=np.float32).reshape(-1, R, 10240)
    for i, href in enumerate(r["hidden"]):
        e = torch.from_numpy(hid[i])
        rel = ((e - href).norm() / href.norm()).item()
        if i < 4 or i % 8 == 0 or i >= len(r["hidden"]) - 2:
            print(f"  layer {i:2d} input: relative difference {rel:.4f}")
    eids = np.fromfile(d / "engine_ids.bin", dtype=np.int64).reshape(R, 16)
    assert (eids == r["ids"]).all(), "row ids differ"
    de = torch.from_numpy(np.fromfile(d / "engine_de.bin", dtype=np.float32)).view(R * 16, 160)
    where = {int(x): k for k, x in enumerate(r["sel"])}
    g = torch.zeros(len(r["sel"]), 160)
    for k, i in enumerate(eids.reshape(-1)):
        g[where[int(i)]] += de[k]
    gr = r["grad"].float()
    cos = torch.nn.functional.cosine_similarity(g.flatten(), gr.flatten(), dim=0).item()
    rel = ((g - gr).norm() / gr.norm()).item()
    per = torch.nn.functional.cosine_similarity(g, gr, dim=1)
    print(f"d(rows): cosine {cos:.5f}, relative difference {rel:.4f}, norms engine {g.norm():.4e} reference {gr.norm():.4e}")
    print(f"  per-row cosine: min {per.min():.4f} median {per.median():.4f} over {len(per)} rows")


def iso(d: Path) -> None:
    """Each reference layer run on the engine's own input for that layer, against the engine's next input."""
    from ref import model as RM

    cs = json.loads((d / "case.json").read_text())
    tokens = cs["tokens"]
    R = len(tokens)
    reference = RM.Reference(MODEL)
    kws = {}
    orig = RM.Streamed._run

    def run(self, h, **kw):
        kws[self.idx] = kw
        return orig(self, h, **kw)
    RM.Streamed._run = run
    picks = {}
    orig_router = RM.M.Qwen4ExpTextTopKRouter.forward

    def router(self, h):
        out = orig_router(self, h)
        picks["last"] = out[2].cpu()
        return out
    with torch.no_grad():
        reference.hidden(tokens)
        RM.Streamed._run = orig
        RM.M.Qwen4ExpTextTopKRouter.forward = router
        hid = np.fromfile(d / "engine_hidden.bin", dtype=np.float32).reshape(-1, R, 10240)
        eroutes = np.fromfile(d / "engine_routes.bin", dtype=np.int32).reshape(-1, R, 5)
        for i in range(len(kws)):
            x = torch.from_numpy(hid[i]).to("cuda", torch.bfloat16)[None]
            y = orig(reference.tm.layers[i], x, **kws[i])[0].float().cpu()
            e = torch.from_numpy(hid[i + 1])
            rel = ((e - y).norm() / (y - x[0].float().cpu()).norm()).item()
            rp = np.sort(picks["last"].numpy().reshape(R, 5), 1)
            ep = np.sort(eroutes[i], 1)
            flips = int((rp != ep).any(1).sum())
            print(f"  layer {i:2d} ({'gdn' if reference.cfg.layer_types[i] == 'linear_attention' else 'attn'}): "
                  f"|diff| / |layer update| = {rel:.4f}; rows routed differently {flips}/{R}")


if __name__ == "__main__":
    {"case": lambda p: case(p, int(sys.argv[3]) if len(sys.argv) > 3 else 0), "ref": ref, "cmp": cmp,
     "iso": iso}[sys.argv[1]](Path(sys.argv[2]))
