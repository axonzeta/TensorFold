//! ngram-learn training: the MoE branch (Qwen4ExpTextSparseMoeBlock) and its backward. Router logits = x·Wrᵀ
//! (512 experts and the shared expert's gate row); top-k by logit, weights = softmax over the chosen; each routed
//! expert y = (silu(x·Wgᵀ) * x·Wuᵀ)·Wdᵀ scaled by its weight; plus sigmoid(gate) * shared expert (block FP8).
//! Routes are discrete: no gradient flows through the choice, only through the chosen weights.
const std = @import("std");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const int4 = @import("cuda_int4.zig");
const tk = @import("train_kernels.zig");
const tw = @import("train_weights.zig");
const hc = @import("train_hc.zig");
const Arena = @import("train_arena.zig").Arena;

/// Routed experts dequantised to bf16 [n][k] in their own arena, kept for one layer.
pub const Experts = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const tk.Kernels,
    s: cuda.Stream,
    arena: Arena,
    map: std.AutoHashMapUnmanaged(u64, u64) = .empty,

    pub fn deinit(x: *Experts) void {
        x.map.deinit(x.gpa);
        x.arena.deinit();
    }

    /// Every dequantised expert dropped (after each layer: experts are not reused across layers).
    pub fn clear(x: *Experts) void {
        x.arena.reset(0);
        x.map.clearRetainingCapacity();
    }

    /// Layer `layer`'s expert `e`, projection m (0 gate, 1 up, 2 down) as a Dense matrix.
    pub fn get(x: *Experts, layer: usize, ex: int4.Experts, e: usize, m: usize) !tw.Dense {
        const n: usize = if (m < 2) ex.width else ex.dims;
        const kk: usize = if (m < 2) ex.dims else ex.width;
        const key = (layer << 32) | (e << 2) | m;
        const gop = try x.map.getOrPut(x.gpa, key);
        if (!gop.found_existing) {
            errdefer _ = x.map.remove(key);
            const ptr = try x.arena.bytes(n * kk * 2);
            try x.k.expert(x.s, ex, e, m, ptr);
            gop.value_ptr.* = ptr;
        }
        return .{ .w = gop.value_ptr.*, .n = @intCast(n), .k = @intCast(kk) };
    }
};

const Hit = struct { e: usize, n: usize, rows: u64, pairs: u64, xe: u64, g: u64, u: u64, sg: u64, act: u64, ye: u64, we: u64 };

pub const Tape = struct {
    moe: *const W.MoE,
    layer: usize,
    x: u64,
    rl: u64,
    logits: u64,
    ids: u64,
    w: u64,
    hits: []Hit,
    sgu: u64,
    sg_g: u64,
    sg_u: u64,
    sg_s: u64,
    sg_act: u64,
    ys: u64,
    sgate_log: u64,
    sgate: u64,

    pub fn deinit(t: *Tape, gpa: std.mem.Allocator) void {
        gpa.free(t.hits);
    }
};

pub const Moe = struct { x: hc.Ctx, ex: *Experts, gpa: std.mem.Allocator };

/// The branch on x [R][D]: out [R][D]. With `forced` ([R][K] host ids), those experts instead of the top-k.
pub fn forward(m: Moe, moe: *const W.MoE, layer: usize, xin: u64, out: u64, forced: ?[]const i32) !Tape {
    const x = m.x;
    const c = x.c;
    const r = x.r;
    const o = x.ops;
    const d: usize = c.hidden;
    const e: usize = c.experts;
    const kk: usize = c.top_k;
    const ex = moe.int4 orelse return error.MoeNeedsInt4;
    const s8 = moe.shared8 orelse return error.MoeNeedsFp8Shared;
    var t: Tape = undefined;
    t.moe = moe;
    t.layer = layer;
    t.x = xin;
    t.rl = try x.a.f32s(r * (e + 1));
    try x.w.forward(.{ .rows = .{ .weight = moe.router, .n = @intCast(e + 1), .k = @intCast(d) } }, xin, t.rl, r);
    t.logits = try x.a.f32s(r * e);
    try o.copy2d(t.rl, e + 1, 0, t.logits, e, 0, r, e, false);
    t.ids = try x.a.bytes(r * kk * 4);
    t.w = try x.a.f32s(r * kk);
    // the routes on the host: each expert's rows and their (row, slot) pairs
    const ids = try m.gpa.alloc(i32, r * kk);
    defer m.gpa.free(ids);
    if (forced) |f| {
        if (f.len != r * kk) return error.ForcedRoutes;
        @memcpy(ids, f);
        try o.upload(t.ids, std.mem.sliceAsBytes(ids));
        try o.pickSoftmax(t.logits, t.ids, t.w, r, e, kk);
    } else {
        try o.topk(t.logits, t.ids, t.w, r, e, kk);
        try o.s.synchronize();
        try (cuda.DeviceBuffer{ .d = x.a.buf.d, .ptr = t.ids, .len = r * kk * 4 }).download(0, std.mem.sliceAsBytes(ids));
    }
    var hits: std.ArrayList(Hit) = .empty;
    errdefer hits.deinit(m.gpa);
    // every expert's rows, then every expert's (row, slot) pairs, grouped by expert, in one upload
    const lists = try m.gpa.alloc(i32, 2 * r * kk);
    defer m.gpa.free(lists);
    const counts = try m.gpa.alloc(usize, e);
    defer m.gpa.free(counts);
    @memset(counts, 0);
    for (ids) |id| counts[@intCast(id)] += 1;
    const starts = try m.gpa.alloc(usize, e + 1);
    defer m.gpa.free(starts);
    starts[0] = 0;
    for (0..e) |ei| starts[ei + 1] = starts[ei] + counts[ei];
    @memset(counts, 0);
    for (ids, 0..) |id, pi| {
        const ei: usize = @intCast(id);
        const at = starts[ei] + counts[ei];
        lists[at] = @intCast(pi / kk);
        lists[r * kk + at] = @intCast(pi);
        counts[ei] += 1;
    }
    const dl = try x.a.bytes(2 * r * kk * 4);
    try o.upload(dl, std.mem.sliceAsBytes(lists));
    const zero = try x.a.zeros(r * d, o.s);
    try o.copy2d(zero, d, 0, out, d, 0, r, d, false);
    for (0..e) |ei| {
        const n = counts[ei];
        if (n == 0) continue;
        var h: Hit = .{ .e = ei, .n = n, .rows = dl + starts[ei] * 4, .pairs = dl + (r * kk + starts[ei]) * 4, .xe = 0, .g = 0, .u = 0, .sg = 0, .act = 0, .ye = 0, .we = 0 };
        const wd: usize = ex.width;
        h.xe = try x.a.f32s(n * d);
        try o.gather(xin, h.rows, h.xe, n, d);
        h.g = try x.a.f32s(n * wd);
        h.u = try x.a.f32s(n * wd);
        try x.w.forward(.{ .dense = try m.ex.get(layer, ex, ei, 0) }, h.xe, h.g, n);
        try x.w.forward(.{ .dense = try m.ex.get(layer, ex, ei, 1) }, h.xe, h.u, n);
        h.sg = try x.a.f32s(n * wd);
        try o.ew(.silu, h.g, h.sg, n * wd, 1, 1);
        h.act = try x.a.f32s(n * wd);
        try o.ew2(.mul, h.sg, h.u, h.act, n * wd);
        h.ye = try x.a.f32s(n * d);
        try x.w.forward(.{ .dense = try m.ex.get(layer, ex, ei, 2) }, h.act, h.ye, n);
        h.we = try x.a.f32s(n);
        try o.gather(t.w, h.pairs, h.we, n, 1);
        try o.scatterAdd(h.ye, h.rows, h.we, out, n, d);
        try hits.append(m.gpa, h);
    }
    t.hits = try hits.toOwnedSlice(m.gpa);
    // the shared expert, gated by sigmoid of the router face's last row
    const sw: usize = s8.width;
    t.sgu = try x.a.f32s(r * 2 * sw);
    try x.w.forward(.{ .fp8 = s8.gu }, xin, t.sgu, r);
    t.sg_g = try x.a.f32s(r * sw);
    t.sg_u = try x.a.f32s(r * sw);
    try o.copy2d(t.sgu, 2 * sw, 0, t.sg_g, sw, 0, r, sw, false);
    try o.copy2d(t.sgu, 2 * sw, sw, t.sg_u, sw, 0, r, sw, false);
    t.sg_s = try x.a.f32s(r * sw);
    try o.ew(.silu, t.sg_g, t.sg_s, r * sw, 1, 1);
    t.sg_act = try x.a.f32s(r * sw);
    try o.ew2(.mul, t.sg_s, t.sg_u, t.sg_act, r * sw);
    t.ys = try x.a.f32s(r * d);
    try x.w.forward(.{ .fp8 = s8.down }, t.sg_act, t.ys, r);
    t.sgate_log = try x.a.f32s(r);
    try o.copy2d(t.rl, e + 1, e, t.sgate_log, 1, 0, r, 1, false);
    t.sgate = try x.a.f32s(r);
    try o.ew(.sigmoid, t.sgate_log, t.sgate, r, 1, 1);
    const routed = try x.a.f32s(r * d);
    try o.copy2d(out, d, 0, routed, d, 0, r, d, false);
    try o.injectFwd(routed, t.sgate, t.ys, out, r, 1, d);
    return t;
}

/// dx (+)= the branch's gradient at its input, given dout [R][D].
pub fn backward(m: Moe, t: *const Tape, dout: u64, dx: u64, add: bool) !void {
    const x = m.x;
    const c = x.c;
    const r = x.r;
    const o = x.ops;
    const d: usize = c.hidden;
    const e: usize = c.experts;
    const kk: usize = c.top_k;
    const ex = t.moe.int4.?;
    const s8 = t.moe.shared8.?;
    const sw: usize = s8.width;
    if (!add) {
        const z = try x.a.zeros(r * d, o.s);
        try o.copy2d(z, d, 0, dx, d, 0, r, d, false);
    }
    // shared expert and its gate
    const dys = try x.a.f32s(r * d);
    const dsg = try x.a.f32s(r);
    try o.injectBwd(dout, t.sgate, t.ys, dys, dsg, r, 1, d);
    const drl = try x.a.zeros(r * (e + 1), o.s);
    const dsl = try x.a.f32s(r);
    try o.ewBwd(.sigmoid, dsg, t.sgate_log, dsl, r, 1, 1, false);
    try o.copy2d(dsl, 1, 0, drl, e + 1, e, r, 1, false);
    const dsact = try x.a.f32s(r * sw);
    try x.w.backward(.{ .fp8 = s8.down }, dys, dsact, r, false);
    const tmp = try x.a.f32s(r * sw);
    try o.ew2(.mul, dsact, t.sg_u, tmp, r * sw);
    const dsgu = try x.a.f32s(r * 2 * sw);
    const dgs = try x.a.f32s(r * sw);
    try o.ewBwd(.silu, tmp, t.sg_g, dgs, r * sw, 1, 1, false);
    try o.copy2d(dgs, sw, 0, dsgu, 2 * sw, 0, r, sw, false);
    try o.ew2(.mul, dsact, t.sg_s, tmp, r * sw);
    try o.copy2d(tmp, sw, 0, dsgu, 2 * sw, sw, r, sw, false);
    try x.w.backward(.{ .fp8 = s8.gu }, dsgu, dx, r, true);
    // routed experts
    const dw = try x.a.zeros(r * kk, o.s);
    for (t.hits) |h| {
        const wd: usize = ex.width;
        const dg_ = try x.a.f32s(h.n * d);
        try o.gather(dout, h.rows, dg_, h.n, d);
        const dwe = try x.a.f32s(h.n);
        try o.sdot(dg_, h.ye, dwe, h.n, d, 1);
        try o.scatterAdd(dwe, h.pairs, 0, dw, h.n, 1);
        const zero = try x.a.zeros(h.n * d, o.s);
        const dye = try x.a.f32s(h.n * d);
        try o.injectFwd(zero, h.we, dg_, dye, h.n, 1, d);
        const dact = try x.a.f32s(h.n * wd);
        try x.w.backward(.{ .dense = try m.ex.get(t.layer, ex, h.e, 2) }, dye, dact, h.n, false);
        const dsgx = try x.a.f32s(h.n * wd);
        try o.ew2(.mul, dact, h.u, dsgx, h.n * wd);
        const dgx = try x.a.f32s(h.n * wd);
        try o.ewBwd(.silu, dsgx, h.g, dgx, h.n * wd, 1, 1, false);
        const dux = try x.a.f32s(h.n * wd);
        try o.ew2(.mul, dact, h.sg, dux, h.n * wd);
        const dxe = try x.a.f32s(h.n * d);
        try x.w.backward(.{ .dense = try m.ex.get(t.layer, ex, h.e, 0) }, dgx, dxe, h.n, false);
        try x.w.backward(.{ .dense = try m.ex.get(t.layer, ex, h.e, 1) }, dux, dxe, h.n, true);
        try o.scatterAdd(dxe, h.rows, 0, dx, h.n, d);
    }
    const dlog = try x.a.zeros(r * e, o.s);
    try o.topkBwd(t.ids, t.w, dw, dlog, r, e, kk);
    try o.copy2d(dlog, e, 0, drl, e + 1, 0, r, e, true);
    try x.w.backward(.{ .rows = .{ .weight = t.moe.router, .n = @intCast(e + 1), .k = @intCast(d) } }, drl, dx, r, true);
}
