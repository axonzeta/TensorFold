//! ngram-learn training: a full-attention branch (Qwen4ExpTextAttention below the indexer's budget, where its
//! selection keeps every causal key) and its backward. [q | gate] per head, k, v = x·W8ᵀ; q, k RMS-normed per head
//! (the engine's fp32 gains) and rotated on their first rot dims; o = causal GQA softmax attention (scale hd^-1/2);
//! out = (o * sigmoid(gate))·Woᵀ.
const std = @import("std");
const W = @import("cuda_weights.zig");
const hc = @import("train_hc.zig");

pub const Tape = struct {
    at: *const W.Attn,
    pa: u64,
    q: u64,
    gate: u64,
    k: u64,
    v: u64,
    qn: u64,
    kn: u64,
    qr: u64,
    kr: u64,
    qrot: u64,
    krot: u64,
    p: u64,
    o: u64,
    sg: u64,
    og: u64,
};

const Geo = struct { hq: usize, hk: usize, hd: usize, qw: usize, kw: usize, pw: usize, rot: usize };

fn geo(x: hc.Ctx, at: *const W.Attn) Geo {
    const c = x.c;
    return .{ .hq = c.heads, .hk = c.kv_heads, .hd = c.head_dim, .qw = c.heads * c.head_dim, .kw = c.kv_heads * c.head_dim, .pw = at.proj8.?.n, .rot = c.rope.rotary_dim };
}

/// The branch on x [R][D] with RoPE tables cs, sn [R][rot]: out [R][D].
pub fn forward(x: hc.Ctx, at: *const W.Attn, xin: u64, cs: u64, sn: u64, out: u64) !Tape {
    if (at.proj8 == null or at.o8 == null) return error.AttnNeedsFp8;
    const p = geo(x, at);
    const r = x.r;
    const o = x.ops;
    const eps: f32 = @floatCast(x.c.eps);
    var t: Tape = undefined;
    t.at = at;
    t.pa = try x.a.f32s(r * p.pw);
    try x.w.forward(.{ .fp8 = at.proj8.? }, xin, t.pa, r);
    t.q = try x.a.f32s(r * p.qw);
    t.gate = try x.a.f32s(r * p.qw);
    for (0..p.hq) |h| {
        try o.copy2d(t.pa, p.pw, h * 2 * p.hd, t.q, p.qw, h * p.hd, r, p.hd, false);
        try o.copy2d(t.pa, p.pw, h * 2 * p.hd + p.hd, t.gate, p.qw, h * p.hd, r, p.hd, false);
    }
    t.k = try x.a.f32s(r * p.kw);
    t.v = try x.a.f32s(r * p.kw);
    try o.copy2d(t.pa, p.pw, 2 * p.qw, t.k, p.kw, 0, r, p.kw, false);
    try o.copy2d(t.pa, p.pw, 2 * p.qw + p.kw, t.v, p.kw, 0, r, p.kw, false);
    t.qn = try x.a.f32s(r * p.qw);
    t.kn = try x.a.f32s(r * p.kw);
    t.qr = try x.a.f32s(r * p.hq);
    t.kr = try x.a.f32s(r * p.hk);
    try o.rmsFwd(t.q, at.q_scale, .f32, t.qn, t.qr, r * p.hq, 1, p.hd, eps);
    try o.rmsFwd(t.k, at.k_scale, .f32, t.kn, t.kr, r * p.hk, 1, p.hd, eps);
    t.qrot = try x.a.f32s(r * p.qw);
    t.krot = try x.a.f32s(r * p.kw);
    try o.rope(t.qn, cs, sn, t.qrot, r, p.hq, p.hd, p.rot, false);
    try o.rope(t.kn, cs, sn, t.krot, r, p.hk, p.hd, p.rot, false);
    t.p = try x.a.f32s(p.hq * r * r);
    t.o = try x.a.f32s(r * p.qw);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(p.hd)));
    try o.attnFwd(t.qrot, t.krot, t.v, t.p, t.o, r, p.hq, p.hk, p.hd, scale);
    t.sg = try x.a.f32s(r * p.qw);
    try o.ew(.sigmoid, t.gate, t.sg, r * p.qw, 1, 1);
    t.og = try x.a.f32s(r * p.qw);
    try o.ew2(.mul, t.o, t.sg, t.og, r * p.qw);
    try x.w.forward(.{ .fp8 = at.o8.? }, t.og, out, r);
    return t;
}

/// dx (+)= the branch's gradient at its input, given dout [R][D].
pub fn backward(x: hc.Ctx, t: *const Tape, cs: u64, sn: u64, dout: u64, dx: u64, add: bool) !void {
    const p = geo(x, t.at);
    const r = x.r;
    const o = x.ops;
    const dog = try x.a.f32s(r * p.qw);
    try x.w.backward(.{ .fp8 = t.at.o8.? }, dout, dog, r, false);
    const d_o = try x.a.f32s(r * p.qw);
    try o.ew2(.mul, dog, t.sg, d_o, r * p.qw);
    const dsg = try x.a.f32s(r * p.qw);
    try o.ew2(.mul, dog, t.o, dsg, r * p.qw);
    const dgate = try x.a.f32s(r * p.qw);
    try o.ewBwd(.sigmoid, dsg, t.gate, dgate, r * p.qw, 1, 1, false);
    const ds = try x.a.f32s(p.hq * r * r);
    const dqrot = try x.a.f32s(r * p.qw);
    const dkrot = try x.a.f32s(r * p.kw);
    const dv = try x.a.f32s(r * p.kw);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(p.hd)));
    try o.attnBwd(t.qrot, t.krot, t.v, t.p, d_o, ds, dqrot, dkrot, dv, r, p.hq, p.hk, p.hd, scale);
    const dqn = try x.a.f32s(r * p.qw);
    const dkn = try x.a.f32s(r * p.kw);
    try o.rope(dqrot, cs, sn, dqn, r, p.hq, p.hd, p.rot, true);
    try o.rope(dkrot, cs, sn, dkn, r, p.hk, p.hd, p.rot, true);
    const dq = try x.a.f32s(r * p.qw);
    const dk = try x.a.f32s(r * p.kw);
    try o.rmsBwd(dqn, t.q, t.at.q_scale, .f32, t.qr, dq, r * p.hq, 1, p.hd, false);
    try o.rmsBwd(dkn, t.k, t.at.k_scale, .f32, t.kr, dk, r * p.hk, 1, p.hd, false);
    const dpa = try x.a.zeros(r * p.pw, x.ops.s);
    for (0..p.hq) |h| {
        try o.copy2d(dq, p.qw, h * p.hd, dpa, p.pw, h * 2 * p.hd, r, p.hd, false);
        try o.copy2d(dgate, p.qw, h * p.hd, dpa, p.pw, h * 2 * p.hd + p.hd, r, p.hd, false);
    }
    try o.copy2d(dk, p.kw, 0, dpa, p.pw, 2 * p.qw, r, p.kw, false);
    try o.copy2d(dv, p.kw, 0, dpa, p.pw, 2 * p.qw + p.kw, r, p.kw, false);
    try x.w.backward(.{ .fp8 = t.at.proj8.? }, dpa, dx, r, add);
}
