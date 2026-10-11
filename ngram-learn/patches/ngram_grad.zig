//! ngram-learn: the training pass on lessons with the n-gram rows as served (the probe worker's and the server's
//! grad op). A lesson's served prompt forward is captured (layer inputs, routes, final streams), then each layer is
//! recomputed and back-propagated on those; the reply is the summed answer NLL and each selected row's gradient.
const std = @import("std");
const cuda = @import("cuda");
const train_model = @import("train_model.zig");
const train_capture = @import("train_capture.zig");
const Overlay = @import("ngram_overlay.zig").Overlay;
const Engine = @import("cuda_engine.zig").Engine;
const Allocator = std.mem.Allocator;

/// The training model, built on the first grad op and kept (its scratch, dequantised head and plans).
pub const State = struct {
    gpa: Allocator,
    model: ?*train_model.Model = null,

    pub fn deinit(st: *State) void {
        if (st.model) |m| {
            m.deinit();
            st.gpa.destroy(m);
        }
        st.model = null;
    }
};

pub fn idsOf(gpa: Allocator, v: ?std.json.Value) ![]u32 {
    const items = if (v) |x| x.array.items else &.{};
    const out = try gpa.alloc(u32, items.len);
    for (items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

/// A case's overlay: "copy" pairs read the source rows from the table as it stands, then "overlay" rows (bf16 bits).
pub fn caseOverlay(gpa: Allocator, table: anytype, ob: std.json.ObjectMap, ov: *Overlay) !void {
    const row = try gpa.alloc(u16, table.width);
    defer gpa.free(row);
    if (ob.get("copy")) |pairs| for (pairs.array.items) |pr| {
        try table.gather(&.{pr.array.items[1].integer}, row);
        try ov.put(pr.array.items[0].integer, row);
    };
    if (ob.get("overlay")) |o| {
        const row_ids = o.object.get("ids").?.array.items;
        const rows = o.object.get("rows").?.array.items;
        if (row_ids.len != rows.len) return error.OverlayShape;
        for (row_ids, rows) |id, r| {
            if (r.array.items.len != row.len) return error.OverlayRowWidth;
            for (row, r.array.items) |*d, x| d.* = @intCast(x.integer);
            try ov.put(id.integer, row);
        }
    }
}

/// {"op": "grad", "tokens", "at", "targets", "sel", "overlay"?, "copy"?, "served"?}: the in-engine training pass on
/// the lesson with the rows as served (base and case overlays applied); the loss and each `sel` row's gradient.
/// {"op": "grad", "lessons": [{"tokens", "at", "targets"}, ...], "sel", ...}: several lessons in one pass (each its
/// own sequence: n-gram history, positions, attention and served forward), the losses summed.
pub fn grad(st: *State, e: *Engine, base: ?*const Overlay, ob: std.json.ObjectMap, out: *std.ArrayList(u8)) !void {
    const j = try begin(st, e, base, ob);
    while (true) {
        const done = step(st, j) catch |err| {
            drop(st, j);
            return err;
        };
        if (done) break;
    }
    return finish(st, j, out); // frees j
}

/// A grad op between `begin` and `finish`: its rows and row gradients on the device, the ids it reports, the pass.
pub const Job = struct {
    rid: []i64,
    ef: []f32,
    sel: []i64,
    eb: cuda.DeviceBuffer,
    de: cuda.DeviceBuffer,
    pass: train_model.Pass,
    rows: usize,
    width: usize,
};

/// The op up to the pass's layers: the lessons parsed, the rows gathered (as served), each lesson's served prefill
/// captured and the pass begun (head and loss). The engine's state and overlays are as before when it returns.
pub fn begin(st: *State, e: *Engine, base: ?*const Overlay, ob: std.json.ObjectMap) !*Job {
    const gpa = st.gpa;
    if (st.model == null) {
        const m = try gpa.create(train_model.Model);
        errdefer gpa.destroy(m);
        try m.init(gpa, e.ctx, &e.f, &e.w, &e.c, 256, 6.0);
        st.model = m;
        // TF_TRAIN_PROF=1: each pass timed by part (each part synchronised), printed after it
        if (std.c.getenv("TF_TRAIN_PROF")) |v| if (v[0] == '1') {
            m.io = e.io;
        };
    }
    const m = st.model.?;
    const Seg = train_model.Seg;
    // every lesson's tokens, at (offset to its rows) and targets, side by side
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    var at: std.ArrayList(u32) = .empty;
    defer at.deinit(gpa);
    var targets: std.ArrayList(u32) = .empty;
    defer targets.deinit(gpa);
    var segs: std.ArrayList(Seg) = .empty;
    defer segs.deinit(gpa);
    const one = [_]std.json.Value{.{ .object = ob }};
    const ls: []const std.json.Value = if (ob.get("lessons")) |v| v.array.items else &one;
    for (ls) |lv| {
        const lo = lv.object;
        const start = tokens.items.len;
        const t = try idsOf(gpa, lo.get("tokens"));
        defer gpa.free(t);
        const a = try idsOf(gpa, lo.get("at"));
        defer gpa.free(a);
        const g = try idsOf(gpa, lo.get("targets"));
        defer gpa.free(g);
        if (a.len != g.len) return error.AtTargets;
        try tokens.appendSlice(gpa, t);
        for (a) |x| {
            if (x >= t.len) return error.AtOutOfRange;
            try at.append(gpa, @intCast(start + x));
        }
        try targets.appendSlice(gpa, g);
        try segs.append(gpa, .{ .start = start, .len = t.len });
    }
    const sel_v = ob.get("sel").?.array.items;
    const table = &e.w.table.?;
    var ov = try Overlay.init(gpa, table.width);
    defer ov.deinit();
    try caseOverlay(gpa, table, ob, &ov);
    ov.under = base;
    const before = table.overlay;
    table.overlay = if (ov.count() > 0 or ov.under != null) &ov else null;
    defer table.overlay = before;
    const pl = e.w.layers[1].ple orelse return error.NoNgramLayer;
    const heads = pl.ngram.heads;
    const r = tokens.items.len;
    const j = try gpa.create(Job);
    errdefer gpa.destroy(j);
    j.rows = r;
    j.width = table.width;
    // each lesson's n-gram rows from its own fresh history
    j.rid = try gpa.alloc(i64, r * heads);
    errdefer gpa.free(j.rid);
    for (segs.items) |sg| {
        var hist: [4]i64 = undefined;
        pl.ngram.initialHistory(hist[0 .. pl.ngram.n - 1]);
        const toks = try gpa.alloc(i64, sg.len);
        defer gpa.free(toks);
        for (tokens.items[sg.start..][0..sg.len], toks) |a, *b| b.* = a;
        try pl.ngram.ids(hist[0 .. pl.ngram.n - 1], toks, j.rid[sg.start * heads ..][0 .. sg.len * heads]);
    }
    j.sel = try gpa.alloc(i64, sel_v.len);
    errdefer gpa.free(j.sel);
    for (sel_v, j.sel) |v, *x| x.* = v.integer;
    const w = table.width;
    const bits = try gpa.alloc(u16, r * heads * w);
    defer gpa.free(bits);
    try table.gather(j.rid, bits);
    j.ef = try gpa.alloc(f32, bits.len);
    errdefer gpa.free(j.ef);
    for (bits, j.ef) |b, *f| f.* = @bitCast(@as(u32, b) << 16);
    j.eb = try cuda.DeviceBuffer.fromHost(e.ctx.d, std.mem.sliceAsBytes(j.ef));
    errdefer j.eb.free();
    j.de = try cuda.DeviceBuffer.alloc(e.ctx.d, j.ef.len * 4);
    errdefer j.de.free();
    // each lesson's served prompt forward (rows as served), its layer inputs and routes captured
    const nl: usize = e.c.layers;
    const wd: usize = e.c.streams * e.c.hidden;
    const slots: usize = e.c.top_k + 1;
    const served_on = if (ob.get("served")) |sv| sv.bool else true;
    const ns = segs.items.len;
    const bufs = try gpa.alloc(cuda.DeviceBuffer, 3 * ns);
    defer gpa.free(bufs);
    var nb: usize = 0;
    defer for (bufs[0..nb]) |*bf| bf.free();
    const caps = try gpa.alloc(train_capture.Capture, ns);
    defer gpa.free(caps);
    const served = try gpa.alloc(train_model.Served, ns);
    defer gpa.free(served);
    const picks = try gpa.alloc(i32, nl * r * slots);
    defer gpa.free(picks);
    if (served_on) for (segs.items, caps, served) |sg, *cap, *sv| {
        const n = sg.len;
        for ([_]usize{ nl * n * wd * 2, nl * n * slots * 4, n * wd * 2 }) |len| {
            bufs[nb] = try cuda.DeviceBuffer.alloc(e.ctx.d, len);
            nb += 1;
        }
        cap.* = .{ .h_in = bufs[nb - 3].ptr, .picks = bufs[nb - 2].ptr, .final = bufs[nb - 1].ptr, .rows = n, .slots = slots };
        e.f.capture = cap;
        defer e.f.capture = null;
        try e.reset();
        _ = e.prefill(tokens.items[sg.start..][0..n], null) catch |err| {
            if (m.check) {
                std.debug.print("train check: the served prefill failed ({s})\n", .{@errorName(err)});
                addressMap(e, m, &.{ .{ "rows in", j.eb }, .{ "row grads", j.de } }, bufs[0..nb]);
            }
            return err;
        };
        e.ctx.synchronize() catch |err| {
            if (m.check) {
                std.debug.print("train check: the served prefill's sync failed ({s})\n", .{@errorName(err)});
                addressMap(e, m, &.{ .{ "rows in", j.eb }, .{ "row grads", j.de } }, bufs[0..nb]);
            }
            return err;
        };
        const pk = picks[nl * sg.start * slots ..][0 .. nl * n * slots];
        try bufs[nb - 2].download(0, std.mem.sliceAsBytes(pk));
        for (0..nl * n) |row| for (0..slots - 1) |k| {
            const x = pk[row * slots + k];
            if (x < 0 or x >= e.c.experts) return error.CapturedPicksOutOfRange;
        };
        sv.* = .{ .cap = cap, .picks = pk };
    };
    j.pass = m.begin(tokens.items, segs.items, at.items, targets.items, j.eb.ptr, j.de.ptr, null, if (served_on) served else null) catch |err| {
        if (m.check) addressMap(e, m, &.{ .{ "rows in", j.eb }, .{ "row grads", j.de } }, bufs[0..nb]);
        return err;
    };
    return j;
}

/// One layer of the job's pass; true once every layer is done.
pub fn step(st: *State, j: *Job) !bool {
    return st.model.?.step(&j.pass);
}

/// The pass ended and the reply written: the loss and each `sel` row's gradient (summed over its lookups). Frees `j`.
pub fn finish(st: *State, j: *Job, out: *std.ArrayList(u8)) !void {
    const gpa = st.gpa;
    defer drop(st, j);
    const m = st.model.?;
    const res = try m.finish(&j.pass);
    try j.de.download(0, std.mem.sliceAsBytes(j.ef));
    if (m.io != null) {
        const pf = m.prof;
        std.debug.print("pass profile ({d} rows): fwd gdn {d:.2} attn {d:.2} moe {d:.2} | bwd layers {d:.2} (moe {d:.2} gdn {d:.2} attn {d:.2}) | head {d:.2}\n", .{ j.rows, pf.fwd_gdn, pf.fwd_attn, pf.fwd_moe, pf.bwd_layer, pf.bwd_moe, pf.bwd_gdn, pf.bwd_attn, pf.head });
        m.prof = .{};
    }
    const w = j.width;
    try out.print(gpa, "{{\"loss\":{d:.6},\"grad\":[", .{res.loss});
    const g = try gpa.alloc(f32, w);
    defer gpa.free(g);
    for (j.sel, 0..) |sv, n| {
        @memset(g, 0);
        for (j.rid, 0..) |id, k| if (id == sv) {
            for (g, j.ef[k * w ..][0..w]) |*a, b| a.* += b;
        };
        try out.appendSlice(gpa, if (n == 0) "[" else ",[");
        for (g, 0..) |v, i| try out.print(gpa, "{s}{e:.6}", .{ if (i == 0) "" else ",", v });
        try out.appendSlice(gpa, "]");
    }
    try out.appendSlice(gpa, "]}");
}

/// A job's memory freed (after finish, or dropped half way).
pub fn drop(st: *State, j: *Job) void {
    const gpa = st.gpa;
    if (st.model) |m| m.abandon(&j.pass);
    j.eb.free();
    j.de.free();
    gpa.free(j.rid);
    gpa.free(j.ef);
    gpa.free(j.sel);
    gpa.destroy(j);
}


/// TF_TRAIN_CHECK: where every large buffer the pass touches lives, to place an Xid 31 fault address (dmesg).
fn addressMap(e: *Engine, m: *train_model.Model, named: []const struct { []const u8, cuda.DeviceBuffer }, caps: []const cuda.DeviceBuffer) void {
    const p = std.debug.print;
    p("address map (fault hunt):\n", .{});
    p("  model arena  0x{x} +{d}\n", .{ m.arena.buf.ptr, m.arena.buf.len });
    p("  expert arena 0x{x} +{d}\n", .{ m.experts.arena.buf.ptr, m.experts.arena.buf.len });
    p("  head         0x{x} +{d}\n", .{ m.head.ptr, m.head.len });
    p("  cublas work  0x{x} +{d}\n", .{ m.ops.work.ptr, m.ops.work.len });
    p("  bf16 scratch 0x{x} +{d}\n", .{ m.weights.xb.ptr, m.weights.xb.len });
    var it = m.weights.wt.valueIterator();
    var lo: u64 = std.math.maxInt(u64);
    var hi: u64 = 0;
    while (it.next()) |b| {
        lo = @min(lo, b.ptr);
        hi = @max(hi, b.ptr + b.len);
    }
    p("  W^T cache    0x{x} .. 0x{x} ({d} faces)\n", .{ lo, hi, m.weights.wt.count() });
    for (named) |nb| p("  {s:<12} 0x{x} +{d}\n", .{ nb[0], nb[1].ptr, nb[1].len });
    for (caps, 0..) |b, i| p("  capture {d}    0x{x} +{d}\n", .{ i, b.ptr, b.len });
    for (e.own.regions.items, 0..) |r, i| p("  own region {d:>3} 0x{x} reserved {d} mapped {d}{s}\n", .{ i, r.base, r.reserved, r.mapped, if (r.ring) " (ring)" else "" });
}
