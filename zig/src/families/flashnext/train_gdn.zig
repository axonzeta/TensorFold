//! ngram-learn training: a GatedDeltaNet branch (Qwen4ExpTextGatedDeltaNet over a fresh state) and its backward.
//! [qkv | z] = x·W8ᵀ, [b | a] = x·Wbaᵀ; qkv = silu(conv(qkv)); q, k l2-normed (q also / sqrt(dk)), repeated to the
//! value heads; g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b); o = delta rule; y = rms(o) w * sigmoid(z);
//! out = y·Woutᵀ.
const std = @import("std");
const W = @import("cuda_weights.zig");
const hc = @import("train_hc.zig");

pub const Tape = struct {
    g: *const W.Gdn,
    x: u64,
    proj: u64,
    ba: u64,
    pre: u64,
    act: u64,
    qraw: u64,
    kraw: u64,
    qr: u64,
    kr: u64,
    qn: u64,
    kn: u64,
    q48: u64,
    k48: u64,
    v: u64,
    a: u64,
    b: u64,
    gate: u64,
    beta: u64,
    o: u64,
    states: u64,
    on: u64,
    orinv: u64,
    z: u64,
    sz: u64,
    y: u64,
};

const Geo = struct { nk: usize, nv: usize, dk: usize, dv: usize, qk: usize, vv: usize, conv: usize, rep: usize, k: usize };

fn geo(x: hc.Ctx) Geo {
    const c = x.c;
    const qk = c.nk * c.dk;
    const vv = c.nv * c.dv;
    return .{ .nk = c.nk, .nv = c.nv, .dk = c.dk, .dv = c.dv, .qk = qk, .vv = vv, .conv = 2 * qk + vv, .rep = c.nv / c.nk, .k = c.conv_kernel };
}

/// The branch on x [R][D]: out [R][D].
pub fn forward(x: hc.Ctx, g: *const W.Gdn, xin: u64, out: u64) !Tape {
    const p = geo(x);
    const r = x.r;
    const o = x.ops;
    const eps: f32 = 1e-6;
    var t: Tape = undefined;
    t.g = g;
    t.x = xin;
    const l8 = g.proj8 orelse return error.GdnNeedsFp8;
    const pw = l8.n; // qkv | z
    t.proj = try x.a.f32s(r * pw);
    try x.w.forward(.{ .fp8 = l8 }, xin, t.proj, r);
    t.ba = try x.a.f32s(r * 2 * p.nv);
    try x.w.forward(.{ .rows = g.proj }, xin, t.ba, r);
    const qkv = try x.a.f32s(r * p.conv);
    try o.copy2d(t.proj, pw, 0, qkv, p.conv, 0, r, p.conv, false);
    t.pre = try x.a.f32s(r * p.conv);
    try o.convFwd(qkv, g.conv, t.pre, r, p.conv, p.k, 1);
    t.act = try x.a.f32s(r * p.conv);
    try o.ew(.silu, t.pre, t.act, r * p.conv, 1, 1);
    t.qraw = try x.a.f32s(r * p.qk);
    t.kraw = try x.a.f32s(r * p.qk);
    t.v = try x.a.f32s(r * p.vv);
    try o.copy2d(t.act, p.conv, 0, t.qraw, p.qk, 0, r, p.qk, false);
    try o.copy2d(t.act, p.conv, p.qk, t.kraw, p.qk, 0, r, p.qk, false);
    try o.copy2d(t.act, p.conv, 2 * p.qk, t.v, p.vv, 0, r, p.vv, false);
    // l2norm(x) = rms(x; eps / dk) / sqrt(dk); q also * dk^-1/2
    const dkf: f32 = @floatFromInt(p.dk);
    t.qr = try x.a.f32s(r * p.nk);
    t.kr = try x.a.f32s(r * p.nk);
    const qtmp = try x.a.f32s(r * p.qk);
    const ktmp = try x.a.f32s(r * p.qk);
    try o.rmsFwd(t.qraw, 0, .none, qtmp, t.qr, r, p.nk, p.dk, eps / dkf);
    try o.rmsFwd(t.kraw, 0, .none, ktmp, t.kr, r, p.nk, p.dk, eps / dkf);
    t.qn = try x.a.f32s(r * p.qk);
    t.kn = try x.a.f32s(r * p.qk);
    try o.ew(.identity, qtmp, t.qn, r * p.qk, 1, 1.0 / dkf);
    try o.ew(.identity, ktmp, t.kn, r * p.qk, 1, 1.0 / @sqrt(dkf));
    t.q48 = try x.a.f32s(r * p.nv * p.dk);
    t.k48 = try x.a.f32s(r * p.nv * p.dk);
    try o.heads(t.qn, t.q48, r, p.nk, p.rep, p.dk, false);
    try o.heads(t.kn, t.k48, r, p.nk, p.rep, p.dk, false);
    t.b = try x.a.f32s(r * p.nv);
    t.a = try x.a.f32s(r * p.nv);
    try o.copy2d(t.ba, 2 * p.nv, 0, t.b, p.nv, 0, r, p.nv, false);
    try o.copy2d(t.ba, 2 * p.nv, p.nv, t.a, p.nv, 0, r, p.nv, false);
    t.gate = try x.a.f32s(r * p.nv);
    t.beta = try x.a.f32s(r * p.nv);
    try o.gdnGates(t.a, t.b, g.a_log, g.dt_bias, t.gate, t.beta, r, p.nv);
    t.o = try x.a.f32s(r * p.vv);
    t.states = try x.a.f32s(p.nv * (r + 1) * p.dk * p.dv);
    try o.deltaFwd(t.q48, t.k48, t.v, t.gate, t.beta, t.o, t.states, r, p.nv, p.dk, p.dv);
    // gated norm: on = rms(o) * w (per value head of dv, w as stored), y = on * sigmoid(z)
    t.on = try x.a.f32s(r * p.vv);
    t.orinv = try x.a.f32s(r * p.nv);
    try o.rmsFwd(t.o, g.norm, .plain, t.on, t.orinv, r * p.nv, 1, p.dv, eps);
    t.z = try x.a.f32s(r * p.vv);
    try o.copy2d(t.proj, pw, p.conv, t.z, p.vv, 0, r, p.vv, false);
    t.sz = try x.a.f32s(r * p.vv);
    try o.ew(.sigmoid, t.z, t.sz, r * p.vv, 1, 1);
    t.y = try x.a.f32s(r * p.vv);
    try o.ew2(.mul, t.on, t.sz, t.y, r * p.vv);
    try x.w.forward(.{ .fp8 = g.out8 orelse return error.GdnNeedsFp8 }, t.y, out, r);
    return t;
}

/// dx (+)= the branch's gradient at its input, given dout [R][D].
pub fn backward(x: hc.Ctx, t: *const Tape, dout: u64, dx: u64, add: bool) !void {
    const p = geo(x);
    const r = x.r;
    const o = x.ops;
    const dkf: f32 = @floatFromInt(p.dk);
    const pw = t.g.proj8.?.n;
    const dy = try x.a.f32s(r * p.vv);
    try x.w.backward(.{ .fp8 = t.g.out8.? }, dout, dy, r, false);
    const don = try x.a.f32s(r * p.vv);
    try o.ew2(.mul, dy, t.sz, don, r * p.vv);
    const dsz = try x.a.f32s(r * p.vv);
    try o.ew2(.mul, dy, t.on, dsz, r * p.vv);
    const dproj = try x.a.zeros(r * pw, x.ops.s);
    const dz = try x.a.f32s(r * p.vv);
    try o.ewBwd(.sigmoid, dsz, t.z, dz, r * p.vv, 1, 1, false);
    try o.copy2d(dz, p.vv, 0, dproj, pw, p.conv, r, p.vv, false);
    const do_ = try x.a.f32s(r * p.vv);
    try o.rmsBwd(don, t.o, t.g.norm, .plain, t.orinv, do_, r * p.nv, 1, p.dv, false);
    const dq48 = try x.a.f32s(r * p.nv * p.dk);
    const dk48 = try x.a.f32s(r * p.nv * p.dk);
    const dv = try x.a.f32s(r * p.vv);
    const dg = try x.a.f32s(r * p.nv);
    const dbeta = try x.a.f32s(r * p.nv);
    try o.deltaBwd(t.q48, t.k48, t.v, t.gate, t.beta, do_, t.states, dq48, dk48, dv, dg, dbeta, r, p.nv, p.dk, p.dv);
    const dqn = try x.a.f32s(r * p.qk);
    const dkn = try x.a.f32s(r * p.qk);
    try o.heads(dq48, dqn, r, p.nk, p.rep, p.dk, true);
    try o.heads(dk48, dkn, r, p.nk, p.rep, p.dk, true);
    const dact = try x.a.f32s(r * p.conv);
    // through the scales and the l2 norms into the conv's output columns
    const tmp = try x.a.f32s(r * p.qk);
    try o.ew(.identity, dqn, tmp, r * p.qk, 1, 1.0 / dkf);
    const dq = try x.a.f32s(r * p.qk);
    try o.rmsBwd(tmp, t.qraw, 0, .none, t.qr, dq, r, p.nk, p.dk, false);
    try o.ew(.identity, dkn, tmp, r * p.qk, 1, 1.0 / @sqrt(dkf));
    const dk = try x.a.f32s(r * p.qk);
    try o.rmsBwd(tmp, t.kraw, 0, .none, t.kr, dk, r, p.nk, p.dk, false);
    try o.copy2d(dq, p.qk, 0, dact, p.conv, 0, r, p.qk, false);
    try o.copy2d(dk, p.qk, 0, dact, p.conv, p.qk, r, p.qk, false);
    try o.copy2d(dv, p.vv, 0, dact, p.conv, 2 * p.qk, r, p.vv, false);
    const dpre = try x.a.f32s(r * p.conv);
    try o.ewBwd(.silu, dact, t.pre, dpre, r * p.conv, 1, 1, false);
    const dqkv = try x.a.f32s(r * p.conv);
    try o.convBwd(dpre, t.g.conv, dqkv, r, p.conv, p.k, 1, false);
    try o.copy2d(dqkv, p.conv, 0, dproj, pw, 0, r, p.conv, false);
    const da = try x.a.f32s(r * p.nv);
    const db = try x.a.f32s(r * p.nv);
    try o.gdnGatesBwd(t.a, t.beta, t.g.a_log, t.g.dt_bias, dg, dbeta, da, db, r, p.nv);
    const dba = try x.a.f32s(r * 2 * p.nv);
    try o.copy2d(db, p.nv, 0, dba, 2 * p.nv, 0, r, p.nv, false);
    try o.copy2d(da, p.nv, 0, dba, 2 * p.nv, p.nv, r, p.nv, false);
    try x.w.backward(.{ .fp8 = t.g.proj8.? }, dproj, dx, r, add);
    try x.w.backward(.{ .rows = t.g.proj }, dba, dx, r, true);
}
