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
        const done = step(st, e, j) catch |err| {
            drop(st, j);
            return err;
        };
        if (done) break;
    }
    return finish(st, j, out); // frees j
}

/// A grad op in steps: one lesson's served prefill a step (`capture`), then the pass's start (`head`: the head and
/// the loss), then one layer a step (`layers`). The server runs a step an idle moment, so a request waits for one
/// step, not the op.
pub const Job = struct {
    phase: enum { capture, head, layers },
    rid: []i64,
    ef: []f32,
    sel: []i64,
    eb: cuda.DeviceBuffer,
    de: cuda.DeviceBuffer,
    pass: train_model.Pass = undefined,
    rows: usize,
    width: usize,
    // until the pass begins
    tokens: std.ArrayList(u32) = .empty,
    at: std.ArrayList(u32) = .empty,
    targets: std.ArrayList(u32) = .empty,
    segs: std.ArrayList(train_model.Seg) = .empty,
    ov: Overlay,
    served_on: bool,
    next: usize = 0,
    bufs: []cuda.DeviceBuffer,
    nb: usize = 0,
    caps: []train_capture.Capture,
    served: []train_model.Served,
    picks: []i32,
};

/// Whether the job's next step runs the engine (a prefill on its own sequence): the caller borrows it around it.
pub fn needsEngine(j: *const Job) bool {
    return j.phase == .capture;
}

/// The op parsed and its rows gathered (as served: the op's own overlay over `base`); no GPU work yet.
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
    const table = &e.w.table.?;
    const j = try gpa.create(Job);
    errdefer gpa.destroy(j);
    j.* = .{ .phase = .capture, .rid = &.{}, .ef = &.{}, .sel = &.{}, .eb = .{ .d = e.ctx.d, .ptr = 0, .len = 0 },
        .de = .{ .d = e.ctx.d, .ptr = 0, .len = 0 }, .rows = 0, .width = table.width, .ov = try Overlay.init(gpa, table.width),
        .served_on = if (ob.get("served")) |sv| sv.bool else true, .bufs = &.{}, .caps = &.{}, .served = &.{}, .picks = &.{} };
    errdefer dropParts(st, j);
    // every lesson's tokens, at (offset to its rows) and targets, side by side
    const one = [_]std.json.Value{.{ .object = ob }};
    const ls: []const std.json.Value = if (ob.get("lessons")) |v| v.array.items else &one;
    for (ls) |lv| {
        const lo = lv.object;
        const start = j.tokens.items.len;
        const t = try idsOf(gpa, lo.get("tokens"));
        defer gpa.free(t);
        const a = try idsOf(gpa, lo.get("at"));
        defer gpa.free(a);
        const g = try idsOf(gpa, lo.get("targets"));
        defer gpa.free(g);
        if (a.len != g.len) return error.AtTargets;
        try j.tokens.appendSlice(gpa, t);
        for (a) |x| {
            if (x >= t.len) return error.AtOutOfRange;
            try j.at.append(gpa, @intCast(start + x));
        }
        try j.targets.appendSlice(gpa, g);
        try j.segs.append(gpa, .{ .start = start, .len = t.len });
    }
    const sel_v = ob.get("sel").?.array.items;
    try caseOverlay(gpa, table, ob, &j.ov);
    j.ov.under = base;
    const pl = e.w.layers[1].ple orelse return error.NoNgramLayer;
    const heads = pl.ngram.heads;
    const r = j.tokens.items.len;
    j.rows = r;
    // each lesson's n-gram rows from its own fresh history
    j.rid = try gpa.alloc(i64, r * heads);
    for (j.segs.items) |sg| {
        var hist: [4]i64 = undefined;
        pl.ngram.initialHistory(hist[0 .. pl.ngram.n - 1]);
        const toks = try gpa.alloc(i64, sg.len);
        defer gpa.free(toks);
        for (j.tokens.items[sg.start..][0..sg.len], toks) |a, *b| b.* = a;
        try pl.ngram.ids(hist[0 .. pl.ngram.n - 1], toks, j.rid[sg.start * heads ..][0 .. sg.len * heads]);
    }
    j.sel = try gpa.alloc(i64, sel_v.len);
    for (sel_v, j.sel) |v, *x| x.* = v.integer;
    const w = table.width;
    const bits = try gpa.alloc(u16, r * heads * w);
    defer gpa.free(bits);
    {
        const before = table.overlay;
        table.overlay = if (j.ov.count() > 0 or j.ov.under != null) &j.ov else null;
        defer table.overlay = before;
        try table.gather(j.rid, bits);
    }
    j.ef = try gpa.alloc(f32, bits.len);
    for (bits, j.ef) |b, *f| f.* = @bitCast(@as(u32, b) << 16);
    j.eb = try cuda.DeviceBuffer.fromHost(e.ctx.d, std.mem.sliceAsBytes(j.ef));
    j.de = try cuda.DeviceBuffer.alloc(e.ctx.d, j.ef.len * 4);
    const ns = j.segs.items.len;
    const nl: usize = e.c.layers;
    const slots: usize = e.c.top_k + 1;
    j.bufs = try gpa.alloc(cuda.DeviceBuffer, 3 * ns);
    j.caps = try gpa.alloc(train_capture.Capture, ns);
    j.served = try gpa.alloc(train_model.Served, ns);
    j.picks = try gpa.alloc(i32, nl * r * slots);
    if (!j.served_on) j.phase = .head;
    return j;
}

/// The job's next step; true once its pass is done (then `finish`).
pub fn step(st: *State, e: *Engine, j: *Job) !bool {
    const m = st.model.?;
    switch (j.phase) {
        .capture => {
            try capture(m, e, j);
            j.next += 1;
            if (j.next == j.segs.items.len) j.phase = .head;
            return false;
        },
        .head => {
            j.pass = m.begin(j.tokens.items, j.segs.items, j.at.items, j.targets.items, j.eb.ptr, j.de.ptr, null, if (j.served_on) j.served else null) catch |err| {
                if (m.check) addressMap(e, m, &.{ .{ "rows in", j.eb }, .{ "row grads", j.de } }, j.bufs[0..j.nb]);
                return err;
            };
            j.phase = .layers;
            dropCaptures(st, j); // the pass holds its own copies now
            return false;
        },
        .layers => return m.step(&j.pass),
    }
}

/// Lesson `j.next`'s served prompt forward (the rows as the op serves them), its layer inputs and routes captured.
fn capture(m: *train_model.Model, e: *Engine, j: *Job) !void {
    const sg = j.segs.items[j.next];
    const n = sg.len;
    const nl: usize = e.c.layers;
    const wd: usize = e.c.streams * e.c.hidden;
    const slots: usize = e.c.top_k + 1;
    for ([_]usize{ nl * n * wd * 2, nl * n * slots * 4, n * wd * 2 }) |len| {
        j.bufs[j.nb] = try cuda.DeviceBuffer.alloc(e.ctx.d, len);
        j.nb += 1;
    }
    const b = j.bufs[j.nb - 3 .. j.nb];
    const cap = &j.caps[j.next];
    cap.* = .{ .h_in = b[0].ptr, .picks = b[1].ptr, .final = b[2].ptr, .rows = n, .slots = slots };
    const table = &e.w.table.?;
    const before = table.overlay;
    table.overlay = if (j.ov.count() > 0 or j.ov.under != null) &j.ov else null;
    defer table.overlay = before;
    e.f.capture = cap;
    defer e.f.capture = null;
    try e.reset();
    _ = e.prefill(j.tokens.items[sg.start..][0..n], null) catch |err| {
        if (m.check) {
            std.debug.print("train check: the served prefill failed ({s})\n", .{@errorName(err)});
            addressMap(e, m, &.{ .{ "rows in", j.eb }, .{ "row grads", j.de } }, j.bufs[0..j.nb]);
        }
        return err;
    };
    e.ctx.synchronize() catch |err| {
        if (m.check) {
            std.debug.print("train check: the served prefill's sync failed ({s})\n", .{@errorName(err)});
            addressMap(e, m, &.{ .{ "rows in", j.eb }, .{ "row grads", j.de } }, j.bufs[0..j.nb]);
        }
        return err;
    };
    const pk = j.picks[nl * sg.start * slots ..][0 .. nl * n * slots];
    try b[1].download(0, std.mem.sliceAsBytes(pk));
    for (0..nl * n) |row| for (0..slots - 1) |k| {
        const x = pk[row * slots + k];
        if (x < 0 or x >= e.c.experts) return error.CapturedPicksOutOfRange;
    };
    j.served[j.next] = .{ .cap = cap, .picks = pk };
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

/// The captures and the lessons' host copies freed (once the pass has begun, or with the job).
fn dropCaptures(st: *State, j: *Job) void {
    const gpa = st.gpa;
    for (j.bufs[0..j.nb]) |*bf| bf.free();
    j.nb = 0;
    gpa.free(j.bufs);
    gpa.free(j.caps);
    gpa.free(j.served);
    gpa.free(j.picks);
    j.bufs = &.{};
    j.caps = &.{};
    j.served = &.{};
    j.picks = &.{};
    j.tokens.deinit(gpa);
    j.at.deinit(gpa);
    j.targets.deinit(gpa);
    j.segs.deinit(gpa);
    j.tokens = .empty;
    j.at = .empty;
    j.targets = .empty;
    j.segs = .empty;
}

fn dropParts(st: *State, j: *Job) void {
    const gpa = st.gpa;
    dropCaptures(st, j);
    if (j.eb.ptr != 0) j.eb.free();
    if (j.de.ptr != 0) j.de.free();
    gpa.free(j.rid);
    gpa.free(j.ef);
    gpa.free(j.sel);
    j.ov.deinit();
}

/// A job's memory freed (after finish, or dropped half way).
pub fn drop(st: *State, j: *Job) void {
    if (j.phase == .layers) if (st.model) |m| m.abandon(&j.pass);
    dropParts(st, j);
    st.gpa.destroy(j);
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
