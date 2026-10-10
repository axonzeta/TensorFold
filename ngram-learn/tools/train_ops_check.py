"""B2+: check the backward's engine ops (tensorfold train-op-check) against PyTorch on random inputs.

    python train_ops_check.py gen DIR     # inputs + ops.json + expected outputs
    (./probe.sh train-op-check DIR)       # the engine runs the ops
    python train_ops_check.py cmp DIR     # compare
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(0)


def save(d: Path, name: str, t: torch.Tensor) -> str:
    t = t.contiguous().cpu()
    raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    raw.numpy().tofile(d / name)
    return name


def load(d: Path, name: str, dtype, shape) -> torch.Tensor:
    if dtype == torch.bfloat16:
        return torch.from_numpy(np.fromfile(d / name, dtype=np.int16)).view(torch.bfloat16).view(*shape)
    return torch.from_numpy(np.fromfile(d / name, dtype=np.float32)).view(*shape)


def gen(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    ops, want = [], {}
    bf = torch.bfloat16
    for i, (m, n, k, add) in enumerate([(64, 2560, 6144, False), (37, 13312, 2560, True), (256, 2560, 1280, False)]):
        x = torch.randn(m, k, dtype=bf)
        w = torch.randn(n, k, dtype=bf) * 0.02
        d0 = torch.randn(m, n) if add else torch.zeros(m, n)
        op = {"op": "gemm", "m": m, "n": n, "k": k, "out_dtype": "f32", "add": add, "x": save(d, f"g{i}_x.bin", x),
              "w": save(d, f"g{i}_w.bin", w), "out": f"g{i}_out.bin"}
        if add:
            op["d"] = save(d, f"g{i}_d.bin", d0)
        ops.append(op)
        want[op["out"]] = ("f32", [m, n], (x.double() @ w.double().t() + d0.double()).float())
    for i, (m, n, k) in enumerate([(9, 2560, 640), (37, 248320, 2560), (256, 10240, 320)]):
        a = torch.randn(m, n, dtype=bf)
        b = torch.randn(n, k, dtype=bf) * 0.02
        ops.append({"op": "gemm_nn", "m": m, "n": n, "k": k, "a": save(d, f"nn{i}_a.bin", a), "b": save(d, f"nn{i}_b.bin", b),
                    "out": f"nn{i}_out.bin"})
        want[f"nn{i}_out.bin"] = ("f32", [m, k], (a.double() @ b.double()).float())
    for i, (r, c) in enumerate([(300, 517), (2560, 6144)]):
        a = torch.randn(r, c, dtype=bf)
        ops.append({"op": "transpose", "rows": r, "cols": c, "in": save(d, f"t{i}_in.bin", a), "out": f"t{i}_out.bin"})
        want[f"t{i}_out.bin"] = ("bf16", [c, r], a.t().contiguous())
    f32 = torch.float32

    def grad_check(tag, fn, inputs, outs):
        """Run fn on leaf copies of `inputs`, backprop random upstream grads; record forward outputs and input grads."""
        leaves = {k: v.clone().double().requires_grad_(k in outs["grads"]) for k, v in inputs.items()}
        y = fn(**leaves)
        dy = torch.randn_like(y)
        y.backward(dy)
        return y.detach().float(), dy.float(), {k: leaves[k].grad.float() for k in outs["grads"]}

    R, S, D = 9, 4, 2560
    for i, mode in enumerate((0, 1, 2)):
        x = torch.randn(R, S * D) * 3
        w = (torch.randn(S * D) * 0.1).to(bf)
        gainf = (lambda: torch.ones(S * D, dtype=torch.float64)) if mode == 0 else (
            (lambda: 1 + w.double()) if mode == 1 else (lambda: w.double()))

        def rms(x):
            xg = x.view(R, S, D)
            return (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + 1e-6)).view(R, S * D) * gainf()
        y, dy, g = grad_check("rms", rms, {"x": x}, {"grads": ["x"]})
        op = {"op": "rms", "r": R, "g": S, "d": D, "mode": mode, "x": save(d, f"rms{i}_x.bin", x),
              "dy": save(d, f"rms{i}_dy.bin", dy), "y": f"rms{i}_y.bin", "dx": f"rms{i}_dx.bin"}
        if mode:
            op["w"] = save(d, f"rms{i}_w.bin", w)
        ops.append(op)
        want[op["y"]] = ("f32", [R, S * D], y)
        want[op["dx"]] = ("f32", [R, S * D], g["x"])
    fns = {1: lambda u: u * torch.sigmoid(u), 2: torch.sigmoid, 3: torch.nn.functional.softplus, 4: torch.exp}
    for f, (alpha, beta) in [(1, (0.25, 1.0)), (2, (1.0, 2.0)), (3, (1.0, 1.0)), (4, (1.0, -1.0))]:
        n = 4099
        x = torch.randn(n) * 2
        y, dy, g = grad_check("ew", lambda x: beta * fns[f](alpha * x), {"x": x}, {"grads": ["x"]})
        op = {"op": "ew", "n": n, "f": f, "alpha": alpha, "beta": beta, "x": save(d, f"ew{f}_x.bin", x),
              "dy": save(d, f"ew{f}_dy.bin", dy), "y": f"ew{f}_y.bin", "dx": f"ew{f}_dx.bin"}
        ops.append(op)
        want[op["y"]] = ("f32", [n], y)
        want[op["dx"]] = ("f32", [n], g["x"])
    u, hn = torch.rand(R, S, D), torch.randn(R, S, D)
    y, dy, g = grad_check("mix", lambda u, hn: (u * hn).mean(1), {"u": u, "hn": hn}, {"grads": ["u", "hn"]})
    ops.append({"op": "mix", "r": R, "s": S, "d": D, "u": save(d, "mix_u.bin", u), "hn": save(d, "mix_hn.bin", hn),
                "dm": save(d, "mix_dm.bin", dy), "m": "mix_m.bin", "du": "mix_du.bin", "dhn": "mix_dhn.bin"})
    want["mix_m.bin"] = ("f32", [R, D], y)
    want["mix_du.bin"] = ("f32", [R, S, D], g["u"])
    want["mix_dhn.bin"] = ("f32", [R, S, D], g["hn"])
    res, inj, br = torch.randn(R, S, D), torch.rand(R, S) * 2, torch.randn(R, D)
    y, dy, g = grad_check("inject", lambda res, inj, br: res + inj[:, :, None] * br[:, None, :],
                          {"res": res, "inj": inj, "br": br}, {"grads": ["inj", "br"]})
    ops.append({"op": "inject", "r": R, "s": S, "d": D, "res": save(d, "inj_res.bin", res), "inj": save(d, "inj_inj.bin", inj),
                "br": save(d, "inj_br.bin", br), "dh": save(d, "inj_dh.bin", dy), "h": "inj_h.bin", "dbr": "inj_dbr.bin",
                "dinj": "inj_dinj.bin"})
    want["inj_h.bin"] = ("f32", [R, S, D], y)
    want["inj_dbr.bin"] = ("f32", [R, D], g["br"])
    want["inj_dinj.bin"] = ("f32", [R, S], g["inj"])
    # partial rotate-half RoPE (first 64 of 256 dims), as HF apply_rotary_pos_emb
    Rr, H, hd, rot = 13, 24, 256, 64
    x = torch.randn(Rr, H, hd)
    inv = 1.0 / (10_000_000 ** (torch.arange(0, rot, 2).float() / rot))
    fr = torch.arange(Rr).float()[:, None] * inv[None]
    cs, sn = torch.cat([fr.cos(), fr.cos()], -1), torch.cat([fr.sin(), fr.sin()], -1)

    def rope(x):
        xr, xp = x[..., :rot], x[..., rot:]
        rh = torch.cat([-xr[..., rot // 2:], xr[..., :rot // 2]], -1)
        return torch.cat([xr * cs.double()[:, None] + rh * sn.double()[:, None], xp], -1)
    y, dy, g = grad_check("rope", rope, {"x": x}, {"grads": ["x"]})
    ops.append({"op": "rope", "r": Rr, "h": H, "hd": hd, "rot": rot, "x": save(d, "rope_x.bin", x), "cos": save(d, "rope_cos.bin", cs),
                "sin": save(d, "rope_sin.bin", sn), "dy": save(d, "rope_dy.bin", dy), "y": "rope_y.bin", "dx": "rope_dx.bin"})
    want["rope_y.bin"] = ("f32", [Rr, H, hd], y)
    want["rope_dx.bin"] = ("f32", [Rr, H, hd], g["x"])
    # causal GQA attention, 24 query heads over 2 kv heads, head dim 256, scale 1/16
    Ra, Hq, Hk, Da = 37, 24, 2, 256
    q, k, v = torch.randn(Ra, Hq, Da), torch.randn(Ra, Hk, Da), torch.randn(Ra, Hk, Da)

    def attn(q, k, v):
        kk = k.repeat_interleave(Hq // Hk, 1).transpose(0, 1)
        vv = v.repeat_interleave(Hq // Hk, 1).transpose(0, 1)
        s = (q.transpose(0, 1) @ kk.transpose(1, 2)) / 16
        s = s.masked_fill(torch.ones(Ra, Ra, dtype=torch.bool).triu(1), float("-inf"))
        return (torch.softmax(s, -1) @ vv).transpose(0, 1)
    y, dy, g = grad_check("attn", attn, {"q": q, "k": k, "v": v}, {"grads": ["q", "k", "v"]})
    ops.append({"op": "attn", "r": Ra, "hq": Hq, "hk": Hk, "d": Da, "scale": 1 / 16, "q": save(d, "at_q.bin", q),
                "k": save(d, "at_k.bin", k), "v": save(d, "at_v.bin", v), "do": save(d, "at_do.bin", dy), "o": "at_o.bin",
                "dq": "at_dq.bin", "dk": "at_dk.bin", "dv": "at_dv.bin"})
    want["at_o.bin"] = ("f32", [Ra, Hq, Da], y)
    want["at_dq.bin"] = ("f32", [Ra, Hq, Da], g["q"])
    want["at_dk.bin"] = ("f32", [Ra, Hk, Da], g["k"])
    want["at_dv.bin"] = ("f32", [Ra, Hk, Da], g["v"])
    # depthwise causal conv: GDN's (kernel 4, dilation 1) and the n-gram block's (kernel 4, dilation 3)
    for i, (Rc, C, K, dil) in enumerate([(41, 10240, 4, 1), (29, 10240, 4, 3)]):
        x = torch.randn(Rc, C)
        w = (torch.randn(C, K) * 0.3).to(bf)

        def conv(x):
            xt = torch.nn.functional.pad(x.t()[None], ((K - 1) * dil, 0))
            return torch.nn.functional.conv1d(xt, w.double()[:, None, :], groups=C, dilation=dil)[0].t()
        y, dy, g = grad_check("conv", conv, {"x": x}, {"grads": ["x"]})
        ops.append({"op": "conv", "r": Rc, "c": C, "k": K, "dil": dil, "x": save(d, f"cv{i}_x.bin", x), "w": save(d, f"cv{i}_w.bin", w),
                    "dy": save(d, f"cv{i}_dy.bin", dy), "y": f"cv{i}_y.bin", "dx": f"cv{i}_dx.bin"})
        want[f"cv{i}_y.bin"] = ("f32", [Rc, C], y)
        want[f"cv{i}_dx.bin"] = ("f32", [Rc, C], g["x"])
    # GatedDeltaNet gates
    Rg, Hg = 23, 48
    a_, b_ = torch.randn(Rg, Hg), torch.randn(Rg, Hg)
    alog, dt = torch.randn(Hg) * 0.5, torch.randn(Hg) * 0.5
    leaves = {"a": a_.double().requires_grad_(), "b": b_.double().requires_grad_()}
    gg = -alog.double().exp() * torch.nn.functional.softplus(leaves["a"] + dt.double())
    bb = torch.sigmoid(leaves["b"])
    dg_, db_ = torch.randn(Rg, Hg), torch.randn(Rg, Hg)
    (gg * dg_.double()).sum().add((bb * db_.double()).sum()).backward()
    ops.append({"op": "gates", "r": Rg, "h": Hg, "a": save(d, "gt_a.bin", a_), "b": save(d, "gt_b.bin", b_), "alog": save(d, "gt_alog.bin", alog),
                "dt": save(d, "gt_dt.bin", dt), "dg": save(d, "gt_dg.bin", dg_), "dbeta": save(d, "gt_dbeta.bin", db_),
                "g": "gt_g.bin", "beta": "gt_beta.bin", "da": "gt_da.bin", "db": "gt_db.bin"})
    want["gt_g.bin"] = ("f32", [Rg, Hg], gg.detach().float())
    want["gt_beta.bin"] = ("f32", [Rg, Hg], bb.detach().float())
    want["gt_da.bin"] = ("f32", [Rg, Hg], leaves["a"].grad.float())
    want["gt_db.bin"] = ("f32", [Rg, Hg], leaves["b"].grad.float())
    # the gated delta rule (HF torch_recurrent_gated_delta_rule's recurrence, q already normed and scaled)
    Rd, Hd, dk, dv = 33, 6, 128, 128
    qn = torch.nn.functional.normalize(torch.randn(Rd, Hd, dk), dim=-1) / dk ** 0.5
    kn = torch.nn.functional.normalize(torch.randn(Rd, Hd, dk), dim=-1)
    vv, gd, bd = torch.randn(Rd, Hd, dv), -torch.rand(Rd, Hd) * 0.5, torch.rand(Rd, Hd)

    def delta(q, k, v, g, beta):
        S = torch.zeros(Hd, dk, dv, dtype=torch.float64)
        outs = []
        for t in range(Rd):
            S = S * g[t].exp()[:, None, None]
            m = (S * k[t][:, :, None]).sum(1)
            dl = (v[t] - m) * beta[t][:, None]
            S = S + k[t][:, :, None] * dl[:, None, :]
            outs.append((S * q[t][:, :, None]).sum(1))
        return torch.stack(outs)
    y, dy, g = grad_check("delta", delta, {"q": qn, "k": kn, "v": vv, "g": gd, "beta": bd},
                          {"grads": ["q", "k", "v", "g", "beta"]})
    ops.append({"op": "delta", "r": Rd, "h": Hd, "dk": dk, "dv": dv, "q": save(d, "dl_q.bin", qn), "k": save(d, "dl_k.bin", kn),
                "v": save(d, "dl_v.bin", vv), "g": save(d, "dl_g.bin", gd), "beta": save(d, "dl_beta.bin", bd),
                "do": save(d, "dl_do.bin", dy), "o": "dl_o.bin", "dq_out": "dl_dq.bin", "dk_out": "dl_dk.bin",
                "dv_out": "dl_dv.bin", "dg_out": "dl_dg.bin", "dbeta_out": "dl_dbeta.bin"})
    want["dl_o.bin"] = ("f32", [Rd, Hd, dv], y)
    for n_, shp in (("q", [Rd, Hd, dk]), ("k", [Rd, Hd, dk]), ("v", [Rd, Hd, dv]), ("g", [Rd, Hd]), ("beta", [Rd, Hd])):
        want[f"dl_d{n_}.bin"] = ("f32", shp, g[n_])
    # the n-gram gate's sign-sqrt sigmoid (ew op 5) and the per-stream dot
    n = 3001
    x = torch.randn(n) * 0.05
    x[:5] = torch.tensor([0.0, 1e-7, -1e-7, 2e-6, -3e-6])
    y, dy, g = grad_check("ssq", lambda x: torch.sigmoid(torch.sign(x) * x.abs().clamp_min(1e-6).sqrt()), {"x": x}, {"grads": ["x"]})
    ops.append({"op": "ew", "n": n, "f": 5, "alpha": 1.0, "beta": 1.0, "x": save(d, "ew5_x.bin", x), "dy": save(d, "ew5_dy.bin", dy),
                "y": "ew5_y.bin", "dx": "ew5_dx.bin"})
    want["ew5_y.bin"] = ("f32", [n], y)
    want["ew5_dx.bin"] = ("f32", [n], g["x"])
    a_, b_ = torch.randn(9 * 4, 2560), torch.randn(9 * 4, 2560)
    y, dy, g = grad_check("sdot", lambda a, b: (a * b).sum(-1) / 2560 ** 0.5, {"a": a_, "b": b_}, {"grads": ["a", "b"]})
    ops.append({"op": "sdot", "rs": 36, "d": 2560, "scale": 1 / 2560 ** 0.5, "a": save(d, "sd_a.bin", a_), "b": save(d, "sd_b.bin", b_),
                "dout": save(d, "sd_dout.bin", dy), "out": "sd_out.bin", "da": "sd_da.bin", "db": "sd_db.bin"})
    want["sd_out.bin"] = ("f32", [36], y)
    want["sd_da.bin"] = ("f32", [36, 2560], g["a"])
    want["sd_db.bin"] = ("f32", [36, 2560], g["b"])
    # MoE routing: top-5 of 512, softmax over the chosen (= HF's softmax, top-k, renormalise)
    Rm, E, K = 19, 512, 5
    lg = torch.randn(Rm, E) * 2
    leaf = lg.double().requires_grad_()
    probs = torch.softmax(leaf, -1)
    top, ids = torch.topk(probs, K, -1)
    wts = top / top.sum(-1, keepdim=True)
    dw = torch.randn(Rm, K)
    (wts * dw.double()).sum().backward()
    ops.append({"op": "topk", "r": Rm, "e": E, "k": K, "logits": save(d, "tk_l.bin", lg), "dw": save(d, "tk_dw.bin", dw),
                "ids": "tk_ids.bin", "w": "tk_w.bin", "dlogits": "tk_dl.bin"})
    want["tk_ids.bin"] = ("i32", [Rm, K], ids.int())
    want["tk_w.bin"] = ("f32", [Rm, K], wts.detach().float())
    want["tk_dl.bin"] = ("f32", [Rm, E], leaf.grad.float())
    # gather + weighted scatter-add (expert dispatch and combine)
    src = torch.randn(40, 2560)
    idx = torch.randint(0, 40, (97,), dtype=torch.int32)
    sc = torch.randn(97)
    gat = src[idx.long()]
    acc = torch.zeros(40, 2560).index_add_(0, idx.long(), gat * sc[:, None])
    ops.append({"op": "gather_scatter", "n": 97, "d": 2560, "rows": 40, "src": save(d, "gs_src.bin", src), "ids": save(d, "gs_ids.bin", idx),
                "scale": save(d, "gs_sc.bin", sc), "gathered": "gs_g.bin", "scattered": "gs_s.bin"})
    want["gs_g.bin"] = ("f32", [97, 2560], gat)
    want["gs_s.bin"] = ("f32", [40, 2560], acc)
    # cross-entropy over the full vocabulary
    Rx, V = 4, 248320
    lg = torch.randn(Rx, V) * 3
    tg = torch.randint(0, V, (Rx,), dtype=torch.int32)
    leaf = lg.double().requires_grad_()
    ls = torch.nn.functional.cross_entropy(leaf, tg.long(), reduction="none")
    (ls.sum() * 0.25).backward()
    ops.append({"op": "xent", "r": Rx, "v": V, "gscale": 0.25, "logits": save(d, "xe_l.bin", lg), "target": save(d, "xe_t.bin", tg),
                "loss": "xe_loss.bin", "dlogits": "xe_dl.bin"})
    want["xe_loss.bin"] = ("f32", [Rx], ls.detach().float())
    want["xe_dl.bin"] = ("f32", [Rx, V], leaf.grad.float())
    (d / "ops.json").write_text(json.dumps(ops))
    torch.save(want, d / "want.pt")
    print(f"{len(ops)} ops -> {d}")


def cmp(d: Path) -> None:
    want = torch.load(d / "want.pt")
    for name, (dt, shape, ref) in want.items():
        if dt == "i32":
            got = torch.from_numpy(np.fromfile(d / name, dtype=np.int32)).view(*shape)
            print(f"{name:16} equal {bool((got == ref).all())}")
            continue
        got = load(d, name, torch.bfloat16 if dt == "bf16" else torch.float32, shape)
        if dt == "bf16":
            ok = bool((got.view(torch.int16) == ref.view(torch.int16)).all())
            print(f"{name:14} bit-equal {ok}")
        else:
            err = (got.double() - ref.double()).abs().max().item()
            rel = err / ref.abs().max().item()
            print(f"{name:16} max|diff| {err:.3e} (relative to max {rel:.2e}) {'OK' if rel < 1e-4 else 'CHECK'}")


if __name__ == "__main__":
    {"gen": gen, "cmp": cmp}[sys.argv[1]](Path(sys.argv[2]))
