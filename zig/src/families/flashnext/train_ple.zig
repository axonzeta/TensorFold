//! ngram-learn training: the n-gram block (Qwen4ExpTextPLELayer) on rows e [R][16 * 160] and its backward down to
//! d(e), the gradient the learner gives each looked-up table row. kn = rms_s(e·Wkᵀ); v = e·Wvᵀ; qn = rms_s(h);
//! x_s = <kn_s, qn_s> / sqrt(D); g_s = sigmoid(sign(x) sqrt(max(|x|, 1e-6))); G_s = g_s v;
//! out = G + silu(dilated causal conv(rms_s(G))); the layer's stream becomes h + out.
const std = @import("std");
const W = @import("cuda_weights.zig");
const hc = @import("train_hc.zig");

pub const Tape = struct {
    p: *const W.Ple,
    h: u64,
    e: u64,
    kp: u64,
    kn: u64,
    kr: u64,
    v: u64,
    qn: u64,
    qr: u64,
    xg: u64,
    gt: u64,
    big: u64,
    cn: u64,
    cr: u64,
    pre: u64,
};

/// h [R][S*D] in, h + out written to `h_out`.
pub fn forward(x: hc.Ctx, p: *const W.Ple, h: u64, e: u64, h_out: u64) !Tape {
    const c = x.c;
    const r = x.r;
    const o = x.ops;
    const s: usize = c.streams;
    const d: usize = c.hidden;
    const wd = s * d;
    const eps: f32 = @floatCast(c.eps);
    var t: Tape = undefined;
    t.p = p;
    t.h = h;
    t.e = e;
    t.kp = try x.a.f32s(r * wd);
    try x.w.forward(.{ .rows = p.key }, e, t.kp, r);
    t.kn = try x.a.f32s(r * wd);
    t.kr = try x.a.f32s(r * s);
    try o.rmsFwd(t.kp, p.norm_key, .f32, t.kn, t.kr, r, s, d, eps);
    t.v = try x.a.f32s(r * d);
    try x.w.forward(.{ .rows = p.value }, e, t.v, r);
    t.qn = try x.a.f32s(r * wd);
    t.qr = try x.a.f32s(r * s);
    try o.rmsFwd(h, p.norm_query, .f32, t.qn, t.qr, r, s, d, eps);
    t.xg = try x.a.f32s(r * s);
    try o.sdot(t.kn, t.qn, t.xg, r * s, d, 1.0 / @sqrt(@as(f32, @floatFromInt(d))));
    t.gt = try x.a.f32s(r * s);
    try o.ew(.sqrt_sigmoid, t.xg, t.gt, r * s, 1, 1);
    const zero = try x.a.zeros(r * wd, o.s);
    t.big = try x.a.f32s(r * wd);
    try o.injectFwd(zero, t.gt, t.v, t.big, r, s, d);
    t.cn = try x.a.f32s(r * wd);
    t.cr = try x.a.f32s(r * s);
    try o.rmsFwd(t.big, p.norm_conv, .f32, t.cn, t.cr, r, s, d, eps);
    t.pre = try x.a.f32s(r * wd);
    try o.convFwd(t.cn, p.conv, t.pre, r, wd, c.ple_kernel, c.ngram_size);
    const act = try x.a.f32s(r * wd);
    try o.ew(.silu, t.pre, act, r * wd, 1, 1);
    try o.ew2(.add, h, t.big, h_out, r * wd);
    try o.ew2(.add, h_out, act, h_out, r * wd);
    return t;
}

/// dh (+)= the block's gradient at its stream input h; de = its gradient at the rows (both given dout = d(h + out)).
pub fn backward(x: hc.Ctx, t: *const Tape, dout: u64, dh: u64, de: u64) !void {
    const c = x.c;
    const r = x.r;
    const o = x.ops;
    const s: usize = c.streams;
    const d: usize = c.hidden;
    const wd = s * d;
    const dpre = try x.a.f32s(r * wd);
    try o.ewBwd(.silu, dout, t.pre, dpre, r * wd, 1, 1, false);
    const dcn = try x.a.f32s(r * wd);
    try o.convBwd(dpre, t.p.conv, dcn, r, wd, c.ple_kernel, c.ngram_size, false);
    const dbig = try x.a.f32s(r * wd);
    try o.copy2d(dout, wd, 0, dbig, wd, 0, r, wd, false);
    try o.rmsBwd(dcn, t.big, t.p.norm_conv, .f32, t.cr, dbig, r, s, d, true);
    const dv = try x.a.f32s(r * d);
    const dgt = try x.a.f32s(r * s);
    try o.injectBwd(dbig, t.gt, t.v, dv, dgt, r, s, d);
    const dxg = try x.a.f32s(r * s);
    try o.ewBwd(.sqrt_sigmoid, dgt, t.xg, dxg, r * s, 1, 1, false);
    const dkn = try x.a.f32s(r * wd);
    const dqn = try x.a.f32s(r * wd);
    try o.sdotBwd(dxg, t.kn, t.qn, dkn, dqn, r * s, d, 1.0 / @sqrt(@as(f32, @floatFromInt(d))), false);
    try o.rmsBwd(dqn, t.h, t.p.norm_query, .f32, t.qr, dh, r, s, d, true);
    const dkp = try x.a.f32s(r * wd);
    try o.rmsBwd(dkn, t.kp, t.p.norm_key, .f32, t.kr, dkp, r, s, d, false);
    try x.w.backward(.{ .rows = t.p.key }, dkp, de, r, false);
    try x.w.backward(.{ .rows = t.p.value }, dv, de, r, true);
}
