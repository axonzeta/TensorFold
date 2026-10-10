//! ngram-learn training: a hyper-connection site (Qwen4ExpTextGatedResidual) forward with what its backward reads.
//! hn = rms_s(h) * gain; dn = hn · [Wdown; Winject]ᵀ; a = silu(dn_low / S); u = sigmoid(a · Wupᵀ);
//! mixed = mean_s(u * hn); inj = 2 sigmoid(dn_inject / S). The residual is h itself.
const std = @import("std");
const W = @import("cuda_weights.zig");
const cfgs = @import("cuda_config.zig");
const tops = @import("train_ops.zig");
const tw = @import("train_weights.zig");
const Arena = @import("train_arena.zig").Arena;

pub const Ctx = struct { ops: *tops.Ops, w: *tw.Weights, a: *Arena, c: *const cfgs.Config, r: usize };

/// What a site's forward keeps for its backward.
pub const Site = struct {
    hc: *const W.Hc,
    h: u64,
    hn: u64,
    rinv: u64,
    dn: u64,
    low: u64,
    act: u64,
    up_pre: u64,
    u: u64,
    inj_log: u64,
    mixed: u64,
    inj: u64,
};

fn dims(x: Ctx) struct { s: usize, d: usize, wd: usize, low: usize } {
    return .{ .s = x.c.streams, .d = x.c.hidden, .wd = x.c.streams * x.c.hidden, .low = x.c.low };
}

/// The site on h [R][S*D] (fp32): mixed [R][D] and, with `hc.inject`, inj [R][S].
pub fn forward(x: Ctx, hc: *const W.Hc, h: u64) !Site {
    if (hc.down8 != null or hc.up8 != null) return error.FastFp8HcNotSupported;
    const g = dims(x);
    const r = x.r;
    const ni: usize = if (hc.inject) g.s else 0;
    var st: Site = undefined;
    st.hc = hc;
    st.h = h;
    st.hn = try x.a.f32s(r * g.wd);
    st.rinv = try x.a.f32s(r * g.s);
    try x.ops.rmsFwd(h, hc.scale, .f32, st.hn, st.rinv, r, g.s, g.d, @floatCast(x.c.eps));
    st.dn = try x.a.f32s(r * (g.low + ni));
    try x.w.forward(.{ .rows = hc.down }, st.hn, st.dn, r);
    st.low = try x.a.f32s(r * g.low);
    try x.ops.copy2d(st.dn, g.low + ni, 0, st.low, g.low, 0, r, g.low, false);
    st.act = try x.a.f32s(r * g.low);
    try x.ops.ew(.silu, st.low, st.act, r * g.low, 1.0 / @as(f32, @floatFromInt(g.s)), 1);
    st.up_pre = try x.a.f32s(r * g.wd);
    try x.w.forward(.{ .rows = hc.up }, st.act, st.up_pre, r);
    st.u = try x.a.f32s(r * g.wd);
    try x.ops.ew(.sigmoid, st.up_pre, st.u, r * g.wd, 1, 1);
    st.mixed = try x.a.f32s(r * g.d);
    try x.ops.mixFwd(st.u, st.hn, st.mixed, r, g.s, g.d);
    st.inj_log = 0;
    st.inj = 0;
    if (hc.inject) {
        st.inj_log = try x.a.f32s(r * ni);
        try x.ops.copy2d(st.dn, g.low + ni, g.low, st.inj_log, ni, 0, r, ni, false);
        st.inj = try x.a.f32s(r * ni);
        try x.ops.ew(.sigmoid, st.inj_log, st.inj, r * ni, 1.0 / @as(f32, @floatFromInt(g.s)), 2);
    }
    return st;
}

/// dh (+)= the site's gradient at h, given dmixed [R][D] and (with inject) dinj [R][S].
pub fn backward(x: Ctx, st: *const Site, dmixed: u64, dinj: u64, dh: u64) !void {
    const g = dims(x);
    const r = x.r;
    const ni: usize = if (st.hc.inject) g.s else 0;
    const s_inv = 1.0 / @as(f32, @floatFromInt(g.s));
    const du = try x.a.f32s(r * g.wd);
    const dhn = try x.a.f32s(r * g.wd);
    try x.ops.mixBwd(dmixed, st.u, st.hn, du, dhn, r, g.s, g.d, false);
    const dup = try x.a.f32s(r * g.wd);
    try x.ops.ewBwd(.sigmoid, du, st.up_pre, dup, r * g.wd, 1, 1, false);
    const dact = try x.a.f32s(r * g.low);
    try x.w.backward(.{ .rows = st.hc.up }, dup, dact, r, false);
    const ddn = try x.a.f32s(r * (g.low + ni));
    const dlow = try x.a.f32s(r * g.low);
    try x.ops.ewBwd(.silu, dact, st.low, dlow, r * g.low, s_inv, 1, false);
    try x.ops.copy2d(dlow, g.low, 0, ddn, g.low + ni, 0, r, g.low, false);
    if (st.hc.inject) {
        const dil = try x.a.f32s(r * ni);
        try x.ops.ewBwd(.sigmoid, dinj, st.inj_log, dil, r * ni, s_inv, 2, false);
        try x.ops.copy2d(dil, ni, 0, ddn, g.low + ni, g.low, r, ni, false);
    }
    try x.w.backward(.{ .rows = st.hc.down }, ddn, dhn, r, true);
    try x.ops.rmsBwd(dhn, st.h, st.hc.scale, .f32, st.rinv, dh, r, g.s, g.d, true);
}
