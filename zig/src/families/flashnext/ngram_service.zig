//! ngram-learn in the server (TF_NGRAM_LEARN=1, one rank): learn ops, JSON objects as the probe worker takes them,
//! queued by the HTTP threads (POST /v1/ngram/op) and run on the engine thread while no stream lives (LaneHost's
//! idle hook), and the n-gram overlays by tag that requests pick with metadata.tag.
//!
//!   {"op": "grad", "tag"?, "lessons" | "tokens"..., "sel", "overlay"?}   ngram_grad.grad under the tag's overlay
//!   {"op": "set", "tag"?, "clear"?, "overlay"?: {"ids", "rows"}, "copy"?, "drop"?: [ids]}   change a tag's overlay
//!   {"op": "load", "tag"?, "path", "clear"?}   rows from a safetensors overlay file (ids I64, rows BF16)
//!   {"op": "hits", "tag"?, "prompt", "cont", "overlay"?}   teacher-forced: how many of cont's tokens are greedy
//!   {"op": "status"}   tags and their row counts
//!
//! The untagged overlay is tag "": it replaces TF_NGRAM_OVERLAY's rows once set. A tag's Overlay is created once and
//! never freed (a sequence keeps a pointer to it); "set" with "clear" empties it. Overlays change only on the engine
//! thread, between streams, so no forward reads one while it changes; the prompt cache is cleared after a change.
const std = @import("std");
const cuda = @import("cuda");
const cuda_engine = @import("cuda_engine.zig");
const ngram_grad = @import("ngram_grad.zig");
const Overlay = @import("ngram_overlay.zig").Overlay;
const Engine = cuda_engine.Engine;
const Allocator = std.mem.Allocator;

/// The overlays by tag (engine thread only).
pub const Tags = struct {
    gpa: Allocator,
    map: std.StringHashMapUnmanaged(*Overlay) = .empty,

    pub fn lookup(t: *const Tags, tag: []const u8) ?*const Overlay {
        const o = t.map.get(tag) orelse return null;
        return o;
    }

    fn get(t: *Tags, tag: []const u8, width: usize) !*Overlay {
        if (t.map.get(tag)) |o| return o;
        const key = try t.gpa.dupe(u8, tag);
        errdefer t.gpa.free(key);
        const o = try t.gpa.create(Overlay);
        errdefer t.gpa.destroy(o);
        o.* = try Overlay.init(t.gpa, width);
        try t.map.put(t.gpa, key, o);
        return o;
    }

    fn deinit(t: *Tags) void {
        var it = t.map.iterator();
        while (it.next()) |kv| {
            kv.value_ptr.*.deinit();
            t.gpa.destroy(kv.value_ptr.*);
            t.gpa.free(kv.key_ptr.*);
        }
        t.map.deinit(t.gpa);
    }
};

/// What an idle call did: nothing waited, an op ran, or an op ran and changed an overlay (kept prompts go stale).
pub const Did = enum { none, ran, changed };

pub const Service = struct {
    gpa: Allocator,
    io: std.Io,
    e: *Engine,
    mutex: std.Io.Mutex = .init,
    finished: std.Io.Condition = .init,
    queue: std.ArrayList(*Op) = .empty,
    tags: Tags,
    train: ngram_grad.State,
    ops: u64 = 0,
    /// the grad op in progress: its pass goes one layer an idle call, so a request waits for a layer, not a pass
    active: ?struct { job: *ngram_grad.Job, op: *Op } = null,

    const Op = struct { body: []const u8, reply: ?[]u8 = null, done: bool = false };

    pub fn create(gpa: Allocator, io: std.Io, e: *Engine) !*Service {
        if (e.world != 1) return error.NgramLearnNeedsOneRank;
        if (e.w.table == null) return error.NoNgramTable;
        const s = try gpa.create(Service);
        s.* = .{ .gpa = gpa, .io = io, .e = e, .tags = .{ .gpa = gpa }, .train = .{ .gpa = gpa } };
        e.tags = &s.tags;
        return s;
    }

    pub fn destroy(s: *Service) void {
        if (s.active) |act| ngram_grad.drop(&s.train, act.job);
        s.e.tags = null;
        s.train.deinit();
        s.tags.deinit();
        s.queue.deinit(s.gpa);
        s.gpa.destroy(s);
    }

    /// An HTTP thread's op: queued, then its reply (allocated in `a`) once the engine thread has run it.
    pub fn submit(s: *Service, a: Allocator, body: []const u8) ![]u8 {
        var op: Op = .{ .body = body };
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        try s.queue.append(s.gpa, &op);
        while (!op.done) s.finished.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromMilliseconds(200), .clock = .awake } }) catch {};
        const r = op.reply orelse return error.NgramOpFailed;
        defer s.gpa.free(r);
        return a.dupe(u8, r);
    }

    /// The engine thread, no stream live: one layer of the grad op in progress, else the oldest queued op run (a grad
    /// op only begun: its prefills, head and loss).
    pub fn idle(s: *Service) Did {
        if (s.active) |act| {
            var out: std.ArrayList(u8) = .empty;
            var failed = false;
            const done = ngram_grad.step(&s.train, act.job) catch |err| blk: {
                ngram_grad.drop(&s.train, act.job);
                out.print(s.gpa, "{{\"error\":\"{s}\"}}", .{@errorName(err)}) catch {};
                failed = true;
                break :blk true;
            };
            if (!done) return .ran;
            if (!failed) ngram_grad.finish(&s.train, act.job, &out) catch |err| { // frees the job either way
                out.clearRetainingCapacity();
                out.print(s.gpa, "{{\"error\":\"{s}\"}}", .{@errorName(err)}) catch {};
            };
            s.active = null;
            s.reply(act.op, &out);
            return .ran;
        }
        s.mutex.lockUncancelable(s.io);
        const op = if (s.queue.items.len > 0) s.queue.orderedRemove(0) else null;
        s.mutex.unlock(s.io);
        const o = op orelse return .none;
        var out: std.ArrayList(u8) = .empty;
        var changed = false;
        const begun = s.run(o, &out, &changed) catch |err| blk: {
            out.clearRetainingCapacity();
            out.print(s.gpa, "{{\"error\":\"{s}\"}}", .{@errorName(err)}) catch {};
            break :blk false;
        };
        if (begun) {
            out.deinit(s.gpa); // a grad op begun: it replies when its pass ends
            return .ran;
        }
        s.reply(o, &out);
        return if (changed) .changed else .ran;
    }

    fn reply(s: *Service, o: *Op, out: *std.ArrayList(u8)) void {
        s.ops += 1;
        s.mutex.lockUncancelable(s.io);
        o.reply = out.toOwnedSlice(s.gpa) catch null;
        o.done = true;
        s.finished.broadcast(s.io);
        s.mutex.unlock(s.io);
    }

    /// One op run, or (grad) begun: true when it was begun and replies later.
    fn run(s: *Service, op_: *Op, out: *std.ArrayList(u8), changed: *bool) !bool {
        const gpa = s.gpa;
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, op_.body, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.OpNotAnObject;
        const ob = parsed.value.object;
        const op = if (ob.get("op")) |v| v.string else return error.OpMissing;
        const tag = if (ob.get("tag")) |v| v.string else "";
        const table = &s.e.w.table.?;
        if (std.mem.eql(u8, op, "grad")) {
            const keep = s.borrow();
            defer s.giveBack(keep);
            const base = s.tags.lookup(tag) orelse s.e.w.table.?.overlay;
            s.active = .{ .job = try ngram_grad.begin(&s.train, s.e, base, ob), .op = op_ };
            return true;
        }
        if (std.mem.eql(u8, op, "hits")) {
            try s.hits(tag, ob, out);
            return false;
        }
        if (std.mem.eql(u8, op, "set")) {
            const o = try s.tags.get(tag, table.width);
            if (ob.get("clear")) |c| if (c.bool) o.clear();
            if (ob.get("drop")) |d| try dropRows(gpa, o, d.array.items);
            try ngram_grad.caseOverlay(gpa, table, ob, o);
            changed.* = true;
            try out.print(gpa, "{{\"tag\":{f},\"rows\":{d}}}", .{ std.json.fmt(tag, .{}), o.count() });
            return false;
        }
        if (std.mem.eql(u8, op, "load")) {
            const o = try s.tags.get(tag, table.width);
            if (ob.get("clear")) |c| if (c.bool) o.clear();
            const n = try o.load(s.io, (ob.get("path") orelse return error.LoadNeedsPath).string);
            changed.* = true;
            try out.print(gpa, "{{\"tag\":{f},\"loaded\":{d},\"rows\":{d}}}", .{ std.json.fmt(tag, .{}), n, o.count() });
            return false;
        }
        if (std.mem.eql(u8, op, "status")) {
            try out.print(gpa, "{{\"ops\":{d},\"default_rows\":{d},\"tags\":{{", .{ s.ops, if (table.overlay) |o| o.count() else 0 });
            var it = s.tags.map.iterator();
            var n: usize = 0;
            while (it.next()) |kv| : (n += 1) try out.print(gpa, "{s}{f}:{d}", .{ if (n == 0) "" else ",", std.json.fmt(kv.key_ptr.*, .{}), kv.value_ptr.*.count() });
            try out.appendSlice(gpa, "}}");
            return false;
        }
        return error.UnknownOp;
    }

    /// The engine as an op runs it: its own sequence (no lane holds it), graphs, MTP absorb and sampling off.
    const Saved = struct { bound: *cuda_engine.Seq, graphs: bool, absorbing: bool, sampling: @FieldType(Engine, "sampling") };

    fn borrow(s: *Service) Saved {
        const e = s.e;
        const keep: Saved = .{ .bound = e.bound, .graphs = e.graphs_on, .absorbing = e.absorbing, .sampling = e.sampling };
        e.bound = e.own;
        e.graphs_on = false;
        e.absorbing = false;
        e.setSampling(null);
        return keep;
    }

    fn giveBack(s: *Service, keep: Saved) void {
        const e = s.e;
        e.f.reset(e.own) catch {};
        e.bound = keep.bound;
        e.graphs_on = keep.graphs;
        e.absorbing = keep.absorbing;
        e.sampling = keep.sampling;
    }

    /// Teacher-forced greedy check: the prompt prefilled, then each of cont's tokens scored against the argmax and
    /// fed (greedy_hits counts the matches; first_miss is the first wrong position, cont's length when none).
    fn hits(s: *Service, tag: []const u8, ob: std.json.ObjectMap, out: *std.ArrayList(u8)) !void {
        const gpa = s.gpa;
        const e = s.e;
        const keep = s.borrow();
        defer s.giveBack(keep);
        const prompt = try ngram_grad.idsOf(gpa, ob.get("prompt"));
        defer gpa.free(prompt);
        const cont = try ngram_grad.idsOf(gpa, ob.get("cont"));
        defer gpa.free(cont);
        if (prompt.len == 0 or cont.len == 0) return error.HitsNeedPromptAndCont;
        const table = &e.w.table.?;
        var ov = try Overlay.init(gpa, table.width);
        defer ov.deinit();
        try ngram_grad.caseOverlay(gpa, table, ob, &ov);
        ov.under = s.tags.lookup(tag) orelse table.overlay;
        const before = table.overlay;
        table.overlay = &ov;
        defer table.overlay = before;
        const host = try gpa.alloc(u16, e.g.head_n);
        defer gpa.free(host);
        try e.reset();
        _ = try e.prefill(prompt, null);
        var n: usize = 0;
        var first: ?usize = null;
        for (cont, 0..) |tok, i| {
            const best = try argmax(e, host, if (i == 0) e.last_logits else e.buf.b.logits);
            if (best == tok) n += 1 else if (first == null) first = i;
            if (i + 1 < cont.len) {
                var one: [1]u32 = undefined;
                try e.forward(&.{tok});
                try e.sample(1, e.pos() + 1, &one);
                try e.commit(1, 1);
            }
        }
        return out.print(gpa, "{{\"greedy_hits\":{d},\"first_miss\":{d},\"of\":{d}}}", .{ n, first orelse cont.len, cont.len });
    }
};

/// The argmax of one bf16 logits row on the device.
fn argmax(e: *Engine, host: []u16, ptr: u64) !u32 {
    try e.stream.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = ptr, .len = host.len * 2 };
    try buf.download(0, std.mem.sliceAsBytes(host));
    var best: f32 = -std.math.inf(f32);
    var at: u32 = 0;
    for (host, 0..) |x, i| {
        const v: f32 = @bitCast(@as(u32, x) << 16);
        if (v > best) {
            best = v;
            at = @intCast(i);
        }
    }
    return at;
}

fn dropRows(gpa: Allocator, o: *Overlay, ids: []const std.json.Value) !void {
    // an overlay keeps rows densely: rebuild it without the dropped ids
    var keep = try Overlay.init(gpa, o.width);
    defer keep.deinit();
    var it = o.map.iterator();
    outer: while (it.next()) |kv| {
        for (ids) |d| if (d.integer == kv.key_ptr.*) continue :outer;
        try keep.put(kv.key_ptr.*, o.rows.items[@as(usize, kv.value_ptr.*) * o.width ..][0..o.width]);
    }
    o.clear();
    var jt = keep.map.iterator();
    while (jt.next()) |kv| try o.put(kv.key_ptr.*, keep.rows.items[@as(usize, kv.value_ptr.*) * keep.width ..][0..keep.width]);
}
