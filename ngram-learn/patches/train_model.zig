//! ngram-learn training: the whole model's loss on a lesson and its gradient at the n-gram rows, on the engine's own
//! weights. Forward with every layer's input kept; the final mixer and the head on the answer rows; cross-entropy;
//! then each layer recomputed from its kept input and undone, down to layer 1's n-gram block and d(rows).
const std = @import("std");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const fwd = @import("cuda_forward.zig");
const cfgs = @import("cuda_config.zig");
const int4 = @import("cuda_int4.zig");
const tk = @import("train_kernels.zig");
const tops = @import("train_ops.zig");
const tw = @import("train_weights.zig");
const Arena = @import("train_arena.zig").Arena;
const hc = @import("train_hc.zig");
const gdn = @import("train_gdn.zig");
const attn = @import("train_attn.zig");
const moe = @import("train_moe.zig");
const ple = @import("train_ple.zig");
const Capture = @import("train_capture.zig").Capture;

/// A served prompt forward's capture and its picks on the host ([layers][rows][slots] i32).
pub const Served = struct { cap: *const Capture, picks: []const i32 };

/// One lesson's rows in a batched pass.
pub const Seg = hc.Seg;

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const W.Weights,
    c: *const cfgs.Config,
    k: tk.Kernels,
    ops: tops.Ops,
    weights: tw.Weights,
    experts: moe.Experts,
    arena: Arena,
    head: cuda.DeviceBuffer,
    /// when set, each layer's chosen experts [L][R][K] (i32) are copied here by the forward pass
    routes: ?u64 = null,
    /// the current lesson's sequences (one for a single sequence)
    segs: []const hc.Seg = &.{},
    /// with an io, the pass is timed by part (each part synchronised) into `prof`
    io: ?std.Io = null,
    prof: Prof = .{},
    /// TF_TRAIN_CHECK=1: the stream synchronised after each stage, a failure named (fault hunting)
    check: bool = false,

    /// For lessons of up to `max_rows` rows; `arena_gib` of scratch for one layer's forward and backward.
    pub fn init(self: *Model, gpa: std.mem.Allocator, cx: *const cuda.Context, f: *fwd.Forward, w: *const W.Weights, c: *const cfgs.Config, max_rows: usize, arena_gib: f64) !void {
        self.gpa = gpa;
        self.d = cx.d;
        self.w = w;
        self.c = c;
        self.k = try tk.Kernels.load(cx);
        errdefer self.k.deinit();
        self.ops = try tops.Ops.init(gpa, cx.d, f.s, &self.k);
        errdefer self.ops.deinit();
        self.weights = try tw.Weights.init(gpa, cx.d, f, &self.ops, max_rows * @max(c.vocab, 16384 * 2));
        errdefer self.weights.deinit();
        self.experts = .{ .gpa = gpa, .d = cx.d, .k = &self.k, .s = f.s, .arena = try Arena.init(cx.d, 6 << 30) };
        errdefer self.experts.deinit();
        self.routes = null;
        self.segs = &.{};
        self.io = null;
        self.prof = .{};
        self.check = if (std.c.getenv("TF_TRAIN_CHECK")) |v| v[0] == '1' else false;
        self.arena = try Arena.init(cx.d, @intFromFloat(arena_gib * (1 << 30)));
        errdefer self.arena.deinit();
        const h = w.head4 orelse return error.TrainNeedsInt4Head;
        self.head = try cuda.DeviceBuffer.alloc(cx.d, @as(usize, h.n) * h.k * 2);
        try self.k.dequantInt4(f.s, h.w, h.s, self.head.ptr, h.n, h.k, h.gs);
    }

    pub fn deinit(m: *Model) void {
        m.head.free();
        m.arena.deinit();
        m.experts.deinit();
        m.weights.deinit();
        m.ops.deinit();
        m.k.deinit();
    }

    /// With `check`: the stream synchronised, a failure reported with the stage and layer.
    fn stage(m: *Model, what: []const u8, layer: usize) !void {
        if (!m.check) return;
        m.ops.s.synchronize() catch |err| {
            std.debug.print("train check: {s} of layer {d} failed ({s}); rows {d}, segments {d}\n", .{ what, layer, @errorName(err), m.segs[m.segs.len - 1].start + m.segs[m.segs.len - 1].len, m.segs.len });
            return err;
        };
    }

    fn ctx(m: *Model, r: usize) hc.Ctx {
        return .{ .ops = &m.ops, .w = &m.weights, .a = &m.arena, .c = m.c, .r = r, .segs = m.segs };
    }

    /// One layer's forward from h (kept), every tape in the arena.
    const Taped = struct {
        p: ?ple.Tape = null,
        h1: u64,
        sa: hc.Site,
        g: ?gdn.Tape = null,
        at: ?attn.Tape = null,
        br: u64,
        h2: u64,
        sb: hc.Site,
        mo: moe.Tape,
        br2: u64,
    };

    fn layerForward(m: *Model, li: usize, h: u64, e: u64, cs: u64, sn: u64, out: u64, r: usize, forced: ?[]const i32) !Taped {
        const x = m.ctx(r);
        const l = &m.w.layers[li];
        const c = m.c;
        const wd: usize = c.streams * c.hidden;
        var t: Taped = undefined;
        t.p = null;
        t.g = null;
        t.at = null;
        t.h1 = h;
        if (l.ple) |*p| {
            t.h1 = try m.arena.f32s(r * wd);
            t.p = try ple.forward(x, p, h, e, t.h1);
            try m.stage("n-gram block", li);
        }
        const t0 = now(m.io);
        t.sa = try hc.forward(x, &l.attn_hc, t.h1);
        t.br = try m.arena.f32s(r * c.hidden);
        if (l.gdn) |*g| t.g = try gdn.forward(x, g, t.sa.mixed, t.br) else t.at = try attn.forward(x, &(l.attn.?), t.sa.mixed, cs, sn, t.br);
        try m.stage(if (l.gdn != null) "DeltaNet forward" else "attention forward", li);
        if (l.gdn != null) m.prof.fwd_gdn += try since(m, t0) else m.prof.fwd_attn += try since(m, t0);
        const t1 = now(m.io);
        t.h2 = try m.arena.f32s(r * wd);
        try m.ops.injectFwd(t.h1, t.sa.inj, t.br, t.h2, r, c.streams, c.hidden);
        t.sb = try hc.forward(x, &l.mlp_hc, t.h2);
        t.br2 = try m.arena.f32s(r * c.hidden);
        t.mo = try moe.forward(.{ .x = x, .ex = &m.experts, .gpa = m.gpa }, &l.moe, li, t.sb.mixed, t.br2, forced);
        try m.stage("MoE forward", li);
        try m.ops.injectFwd(t.h2, t.sb.inj, t.br2, out, r, c.streams, c.hidden);
        m.prof.fwd_moe += try since(m, t1);
        return t;
    }

    /// dh_in = the layer's gradient at its input given dh_out; de (+)= the n-gram rows' gradient (layer 1).
    fn layerBackward(m: *Model, t: *Taped, cs: u64, sn: u64, dh_out: u64, dh_in: u64, de: u64, r: usize) !void {
        const x = m.ctx(r);
        const c = m.c;
        const wd: usize = c.streams * c.hidden;
        const s: usize = c.streams;
        const d: usize = c.hidden;
        const dbr2 = try m.arena.f32s(r * d);
        const dinj2 = try m.arena.f32s(r * s);
        try m.ops.injectBwd(dh_out, t.sb.inj, t.br2, dbr2, dinj2, r, s, d);
        const dh2 = try m.arena.f32s(r * wd);
        try m.ops.copy2d(dh_out, wd, 0, dh2, wd, 0, r, wd, false);
        const dmb = try m.arena.f32s(r * d);
        const tm = now(m.io);
        try moe.backward(.{ .x = x, .ex = &m.experts, .gpa = m.gpa }, &t.mo, dbr2, dmb, false);
        m.prof.bwd_moe += try since(m, tm);
        try hc.backward(x, &t.sb, dmb, dinj2, dh2);
        const dbr = try m.arena.f32s(r * d);
        const dinj = try m.arena.f32s(r * s);
        try m.ops.injectBwd(dh2, t.sa.inj, t.br, dbr, dinj, r, s, d);
        const dh1 = if (t.p != null) try m.arena.f32s(r * wd) else dh_in;
        try m.ops.copy2d(dh2, wd, 0, dh1, wd, 0, r, wd, false);
        const dma = try m.arena.f32s(r * d);
        const tx = now(m.io);
        if (t.g) |*g| try gdn.backward(x, g, dbr, dma, false) else try attn.backward(x, &(t.at.?), cs, sn, dbr, dma, false);
        if (t.g != null) m.prof.bwd_gdn += try since(m, tx) else m.prof.bwd_attn += try since(m, tx);
        try hc.backward(x, &t.sa, dma, dinj, dh1);
        if (t.p) |*p| {
            try m.ops.copy2d(dh1, wd, 0, dh_in, wd, 0, r, wd, false);
            try ple.backward(x, p, dh1, dh_in, de);
        }
    }

    pub const Result = struct { loss: f64, rows: usize };

    /// Loss of `targets` at rows `at` (row i predicts targets[i]) of `tokens`, and d(e) [R][16 * 160] at the n-gram
    /// rows `e` (fp32, the rows layer 1 looks up); with `dump`, every layer's input [L + 1][R][S*D] is copied there.
    /// With `served`, every layer's input and its routes come from that served forward: no training forward runs,
    /// and each layer is recomputed for its backward from the served activations on the served routes.
    pub fn lesson(m: *Model, tokens: []const u32, at: []const u32, targets: []const u32, e: u64, de: u64, dump: ?u64, served: ?Served) !Result {
        const one = [_]hc.Seg{.{ .start = 0, .len = tokens.len }};
        const sv = [_]Served{if (served) |x| x else undefined};
        return m.lessons(tokens, &one, at, targets, e, de, dump, if (served != null) &sv else null);
    }

    /// `lesson` over several sequences side by side (`segs` cover `tokens` in order; `at` indexes all rows): row-wise
    /// work batched, attention, the delta rule and the convs within each sequence; `served` one capture a sequence.
    pub fn lessons(m: *Model, tokens: []const u32, segs: []const hc.Seg, at: []const u32, targets: []const u32, e: u64, de: u64, dump: ?u64, served: ?[]const Served) !Result {
        var p = try m.begin(tokens, segs, at, targets, e, de, dump, served);
        errdefer m.abandon(&p);
        while (!try m.step(&p)) {}
        return m.finish(&p);
    }

    /// The pass up to its layers: the captured (or recomputed) layer inputs, the head and the loss. `step` then takes
    /// the layers one at a time (the server serves requests between them), `finish` ends it. The rows `e` and the
    /// row gradients `de` must live until `finish`; the captures in `served` only until `begin` returns.
    pub fn begin(m: *Model, tokens: []const u32, segs: []const hc.Seg, at: []const u32, targets: []const u32, e: u64, de: u64, dump: ?u64, served: ?[]const Served) !Pass {
        const c = m.c;
        const r = tokens.len;
        const own_segs = try m.gpa.dupe(hc.Seg, segs);
        errdefer m.gpa.free(own_segs);
        m.segs = own_segs;
        try m.stage("the pass's start", 0);
        errdefer m.segs = &.{};
        const wd: usize = c.streams * c.hidden;
        const d: usize = c.hidden;
        const o = &m.ops;
        const nl: usize = c.layers;
        m.arena.reset(0);
        // RoPE tables (HF: fp32 angles, cos / sin rounded to the activations' bf16)
        const rot: usize = c.rope.rotary_dim;
        const tab = try m.gpa.alloc(f32, 2 * r * rot);
        defer m.gpa.free(tab);
        for (segs) |sg| for (sg.start..sg.start + sg.len) |p| for (0..rot / 2) |i| {
            const inv: f32 = @floatCast(1.0 / std.math.pow(f64, c.rope.theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rot))));
            const ang: f32 = @as(f32, @floatFromInt(p - sg.start)) * inv;
            for ([_]usize{ i, i + rot / 2 }) |j| {
                tab[p * rot + j] = bf16Round(@cos(ang));
                tab[r * rot + p * rot + j] = bf16Round(@sin(ang));
            }
        };
        try m.stage("the arena reset", 0);
        const cs = try m.arena.f32s(2 * r * rot);
        const sn = cs + r * rot * 4;
        try o.upload(cs, std.mem.sliceAsBytes(tab));
        // the kept layer inputs, then the embedding spread over the streams
        const keep = try m.arena.f32s((nl + 1) * r * wd);
        const kk: usize = c.top_k;
        const forced = try m.gpa.alloc(i32, if (served != null) nl * r * kk else 0);
        errdefer m.gpa.free(forced);
        if (served) |svs| {
            if (svs.len != segs.len) return error.CaptureSegments;
            for (svs, segs) |sv, sg| {
                const n = sg.len;
                if (sv.cap.rows != n) return error.CaptureRows;
                for (0..nl) |li| try o.toF32(sv.cap.h_in + li * n * wd * 2, keep + (li * r + sg.start) * wd * 4, n * wd);
                try o.toF32(sv.cap.final, keep + (nl * r + sg.start) * wd * 4, n * wd);
                for (0..nl) |li| for (0..n) |row| for (0..kk) |k| {
                    forced[(li * r + sg.start + row) * kk + k] = sv.picks[(li * n + row) * sv.cap.slots + k];
                };
            }
        } else {
            const x0 = try m.arena.f32s(r * d);
            for (tokens, 0..) |tok, i| try o.toF32(m.w.embed + @as(u64, tok) * d * 2, x0 + i * d * 4, d);
            try o.heads(x0, keep, r, 1, c.streams, d, false);
        }
        try m.stage("the captured inputs", 0);
        const base = m.arena.mark();
        for (0..nl) |li| {
            if (served != null) break;
            m.arena.reset(base);
            var t = try m.layerForward(li, keep + li * r * wd * 4, e, cs, sn, keep + (li + 1) * r * wd * 4, r, null);
            if (m.routes) |rt| {
                const n = r * c.top_k * 4;
                try (cuda.DeviceBuffer{ .d = m.d, .ptr = rt, .len = nl * n }).copyFrom(li * n, t.mo.ids, n, o.s.handle);
            }
            t.mo.deinit(m.gpa);
            m.experts.clear();
        }
        // the final mixer, the head on the answer rows, cross-entropy
        m.arena.reset(base);
        const th = now(m.io);
        const x = m.ctx(r);
        const hfin = keep + nl * r * wd * 4;
        const fin = try hc.forward(x, &m.w.mixer, hfin);
        const na = at.len;
        const idx = try m.arena.bytes(na * 4);
        const at_i = try m.gpa.alloc(i32, na);
        defer m.gpa.free(at_i);
        for (at, at_i) |a, *b| b.* = @intCast(a);
        try o.upload(idx, std.mem.sliceAsBytes(at_i));
        const rows = try m.arena.f32s(na * d);
        try o.gather(fin.mixed, idx, rows, na, d);
        const v: usize = c.vocab;
        const logits = try m.arena.f32s(na * v);
        const head: tw.Lin = .{ .dense = .{ .w = m.head.ptr, .n = @intCast(v), .k = @intCast(d) } };
        try m.weights.forward(head, rows, logits, na);
        const tg = try m.arena.bytes(na * 4);
        for (targets, at_i) |a, *b| b.* = @intCast(a);
        try o.upload(tg, std.mem.sliceAsBytes(at_i));
        const loss = try m.arena.f32s(na);
        const dlogits = try m.arena.f32s(na * v);
        try o.xent(logits, tg, loss, dlogits, na, v, 1);
        const drows = try m.arena.f32s(na * d);
        try m.weights.backward(head, dlogits, drows, na, false);
        const dmixed = try m.arena.zeros(r * d, o.s);
        try o.scatterAdd(drows, idx, 0, dmixed, na, d);
        // two gradient buffers outside the per-layer scratch
        const g0 = try m.arena.zeros(r * wd, o.s);
        const g1 = try m.arena.zeros(r * wd, o.s);
        try hc.backward(x, &fin, dmixed, 0, g0);
        try m.stage("head and mixer", nl);
        m.prof.head += try since(m, th);
        m.prof.passes += 1;
        const lh = try m.gpa.alloc(f32, na);
        defer m.gpa.free(lh);
        try o.s.synchronize();
        try (cuda.DeviceBuffer{ .d = m.d, .ptr = loss, .len = na * 4 }).download(0, std.mem.sliceAsBytes(lh));
        var total: f64 = 0;
        for (lh) |v_| total += v_;
        const zero_e = (cuda.DeviceBuffer{ .d = m.d, .ptr = de, .len = r * c.ple_dim * 4 });
        try zero_e.fill8(0, o.s.handle);
        return .{ .r = r, .e = e, .de = de, .cs = cs, .sn = sn, .keep = keep, .forced = forced, .kk = kk, .cur = g0,
            .nxt = g1, .base2 = m.arena.mark(), .li = nl, .total = total, .dump = dump, .segs = own_segs };
    }

    /// One layer of the pass (the last not yet done): recomputed from its kept input on its routes, back-propagated.
    /// True once every layer is done.
    pub fn step(m: *Model, p: *Pass) !bool {
        if (p.li == 0) return true;
        m.segs = p.segs;
        const c = m.c;
        const r = p.r;
        const wd: usize = c.streams * c.hidden;
        p.li -= 1;
        const li = p.li;
        m.arena.reset(p.base2);
        var t = try m.layerForward(li, p.keep + li * r * wd * 4, p.e, p.cs, p.sn, try m.arena.f32s(r * wd), r, forcedOf(p.forced, li, r, p.kk));
        try m.stage("recompute", li);
        defer t.mo.deinit(m.gpa);
        const tb = now(m.io);
        try m.layerBackward(&t, p.cs, p.sn, p.cur, p.nxt, p.de, r);
        try m.stage("backward", li);
        m.prof.bwd_layer += try since(m, tb);
        m.experts.clear();
        std.mem.swap(u64, &p.cur, &p.nxt);
        return p.li == 0;
    }

    /// The pass ended: `de` holds the row gradients.
    pub fn finish(m: *Model, p: *Pass) !Result {
        defer m.abandon(p);
        const c = m.c;
        const wd: usize = c.streams * c.hidden;
        const nl: usize = c.layers;
        if (p.dump) |dst| try (cuda.DeviceBuffer{ .d = m.d, .ptr = dst, .len = (nl + 1) * p.r * wd * 4 }).copyFrom(0, p.keep, (nl + 1) * p.r * wd * 4, m.ops.s.handle);
        try m.ops.s.synchronize();
        return .{ .loss = p.total, .rows = p.r };
    }

    /// A pass's own memory freed (after finish, or when it is dropped half way).
    pub fn abandon(m: *Model, p: *Pass) void {
        m.gpa.free(p.forced);
        m.gpa.free(p.segs);
        p.forced = &.{};
        p.segs = &.{};
        m.segs = &.{};
    }
};

/// A pass between `begin` and `finish` (train_model.Model): where it is and what it holds.
pub const Pass = struct {
    r: usize,
    e: u64,
    de: u64,
    cs: u64,
    sn: u64,
    keep: u64,
    forced: []i32,
    kk: usize,
    cur: u64,
    nxt: u64,
    base2: usize,
    li: usize,
    total: f64,
    dump: ?u64,
    segs: []hc.Seg,
};

fn forcedOf(f: []const i32, li: usize, rows: usize, k: usize) ?[]const i32 {
    return if (f.len == 0) null else f[li * rows * k ..][0 .. rows * k];
}

/// Seconds a pass spent in each part (summed over passes; `profile` prints and clears them).
pub const Prof = struct {
    fwd_gdn: f64 = 0,
    fwd_attn: f64 = 0,
    fwd_moe: f64 = 0,
    bwd_layer: f64 = 0,
    bwd_moe: f64 = 0,
    bwd_gdn: f64 = 0,
    bwd_attn: f64 = 0,
    head: f64 = 0,
    passes: usize = 0,
};

fn now(io: ?std.Io) ?std.Io.Timestamp {
    return if (io) |i| std.Io.Timestamp.now(i, .awake) else null;
}

fn since(m: *Model, t: ?std.Io.Timestamp) !f64 {
    const t0 = t orelse return 0;
    try m.ops.s.synchronize();
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(m.io.?, .awake)).nanoseconds)) / 1e9;
}

fn bf16Round(x: f32) f32 {
    const u: u32 = @bitCast(x);
    const rnd = u +% 0x7FFF +% ((u >> 16) & 1);
    return @bitCast(rnd & 0xFFFF0000);
}
