"""ngram-learn Phase 4 (engine side): learning in the server and overlays by tag (run in the fork root).

- core: Request.tag; lanes.Stream.tag; api.Learn (an op queue a backend runs on the engine thread while idle);
  Engine.learn; LaneHost runs it when no stream lives, clears kept prompts after an overlay change, and keeps
  no prompt state for a tagged request; prompt_cache.Store.clear.
- server: metadata.tag -> Request.tag; POST /v1/ngram/op -> Engine.learn.
- native/cuda.zig: a family's Loaded.learn wired to the lane host, its calls with the context current.
- flashnext: ngram_grad.zig and ngram_service.zig (TF_NGRAM_LEARN=1); Seq.overlay picked from the stream's tag at
  admission; the host n-gram gather per sequence with its overlay (else the table's).
"""

import pathlib

FILES: dict[str, str] = {}


def edit(path: str, old: str, new: str, count: int = 1) -> None:
    p = pathlib.Path(path)
    s = FILES.get(path) or p.read_text()
    assert s.count(old) == count, (path, old[:120], s.count(old))
    FILES[path] = s.replace(old, new)


# -- core/engine_api.zig ---------------------------------------------------------------------------------------
api = "zig/src/core/engine_api.zig"
edit(api, """    /// The prompt's images and videos (``Info.media`` engines only); such a prompt is never kept or resumed.
    media: ?*const Media = null,
};
""", """    /// The prompt's images and videos (``Info.media`` engines only); such a prompt is never kept or resumed.
    media: ?*const Media = null,
    /// ngram-learn: the n-gram overlay the request reads (metadata.tag; empty: the default). A tagged prompt is never
    /// kept or resumed (kept states are the default overlay's).
    tag: []const u8 = "",
};
""")
edit(api, """        memory: *const fn (ctx: *anyopaque, reset_peak: bool) ?Memory,
    };
""", """        memory: *const fn (ctx: *anyopaque, reset_peak: bool) ?Memory,
        /// ngram-learn: one learn op (JSON) run by the engine between streams; its JSON reply in `a`.
        learn: ?*const fn (ctx: *anyopaque, a: std.mem.Allocator, body: []const u8) anyerror![]u8 = null,
    };

    /// A learn op's reply, or error.NoLearner when the backend learns nothing.
    pub fn learn(e: Engine, a: std.mem.Allocator, body: []const u8) anyerror![]u8 {
        const f = e.vtable.learn orelse return error.NoLearner;
        return f(e.ctx, a, body);
    }
""")
edit(api, """/// The lane core served to the HTTP threads (lane_host.zig).""", """/// ngram-learn: a backend's learn ops. `submit` (any thread) queues one and waits for its reply; `idle` (the engine
/// thread, no stream live) runs the oldest.
pub const Learn = struct {
    ctx: *anyopaque,
    submit: *const fn (ctx: *anyopaque, a: std.mem.Allocator, body: []const u8) anyerror![]u8,
    idle: *const fn (ctx: *anyopaque) Did,

    pub const Did = enum { none, ran, changed };
};

/// The lane core served to the HTTP threads (lane_host.zig).""")

# -- core/lanes/stream.zig --------------------------------------------------------------------------------------
st = "zig/src/core/lanes/stream.zig"
edit(st, """    media: ?*const Media = null,
};

pub const Stream = struct {""", """    media: ?*const Media = null,
    /// ngram-learn: the n-gram overlay's tag (empty: the default)
    tag: []const u8 = "",
};

pub const Stream = struct {""")
edit(st, """    /// the prompt's images and video frames (null: text); valid until the stream is released
    media: ?*const Media = null,
    cached: u32 = 0,""", """    /// the prompt's images and video frames (null: text); valid until the stream is released
    media: ?*const Media = null,
    /// ngram-learn: the n-gram overlay's tag (empty: the default); valid until the stream is released
    tag: []const u8 = "",
    cached: u32 = 0,""")
edit(st, """            .reuse = spec.reuse,
            .media = spec.media,
        };""", """            .reuse = spec.reuse,
            .media = spec.media,
            .tag = spec.tag,
        };""")

# -- core/prompt_cache.zig --------------------------------------------------------------------------------------
pc = "zig/src/core/prompt_cache.zig"
edit(pc, """    fn free(s: *Store, e: *Entry) void {""", """    /// ngram-learn: every kept state dropped (an overlay changed, so their rows are stale); between streams only.
    pub fn clear(s: *Store) void {
        for (s.entries.items) |e| s.free(e);
        s.entries.clearRetainingCapacity();
        s.held = 0;
    }

    fn free(s: *Store, e: *Entry) void {""")

# -- core/lane_host.zig -----------------------------------------------------------------------------------------
lh = "zig/src/core/lane_host.zig"
edit(lh, """    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
""", """    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    learn: ?api.Learn = null, // ngram-learn: learn ops the engine thread runs while no stream lives
""")
edit(lh, """        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn } };""",
     """        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .learn = learnFn } };""")
edit(lh, """    fn infoFn(ctx: *anyopaque) Info {""", """    fn learnFn(ctx: *anyopaque, a: Allocator, body: []const u8) anyerror![]u8 {
        const l = self(ctx).learn orelse return error.NoLearner;
        return l.submit(l.ctx, a, body);
    }

    fn infoFn(ctx: *anyopaque) Info {""")
edit(lh, """        if (r.media == null) if (h.cache) |store|""", """        if (r.media == null and r.tag.len == 0) if (h.cache) |store|""")
edit(lh, """            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .loop_guard = r.loop_guard,""", """            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .loop_guard = r.loop_guard,
            .tag = r.tag,""")
edit(lh, """            if (h.core.activeCount() == 0 and h.core.fillingCount() == 0) {
                if (h.cancels.items.len == 0 and (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes))""",
     """            if (h.core.activeCount() == 0 and h.core.fillingCount() == 0) {
                // ngram-learn: nothing waits, nothing lives: one learn op (it may hold the thread for a few seconds)
                if (h.learn) |l| if (h.cancels.items.len == 0 and h.queued.items.len == 0 and h.admitted.items.len == 0) {
                    h.unlock();
                    switch (l.idle(l.ctx)) {
                        .none => {},
                        .ran => continue,
                        .changed => {
                            if (h.cache) |store| store.clear();
                            continue;
                        },
                    }
                    h.lock();
                    if (h.closing or h.queued.items.len > 0 or h.cancels.items.len > 0) {
                        h.unlock();
                        continue;
                    }
                };
                if (h.cancels.items.len == 0 and (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes))""")

# -- server ------------------------------------------------------------------------------------------------------
ch = "zig/src/server/chat.zig"
edit(ch, """        .media = rendered.media,
    };
    try srv.checkFeatures(""", """        .media = rendered.media,
        .tag = ngramTag(f),
    };
    try srv.checkFeatures(""")
edit(ch, """/// Render ``input`` and run every check that can refuse it, before anything reaches the client or the engine.""",
     """/// ngram-learn: the n-gram overlay a request reads, from ``metadata.tag`` (OpenAI's metadata map); empty: the default.
fn ngramTag(f: json.Value) []const u8 {
    const md = f.field("metadata") orelse return "";
    const t = md.field("tag") orelse return "";
    return if (t == .string) t.string else "";
}

/// Render ``input`` and run every check that can refuse it, before anything reaches the client or the engine.""")
edit("zig/src/server/openai.zig", """"priority", "draft", "thinking_budget", "ignore_eos", "stop" }""",
     """"priority", "draft", "thinking_budget", "ignore_eos", "stop", "metadata" }""")
rt = "zig/src/server/routes.zig"
edit(rt, """    if (anthropic.route(conn.path)) return anthropic.post(srv, conn, a);""", """    if (std.mem.eql(u8, route, "/v1/ngram/op")) return ngramOp(srv, conn, a);
    if (anthropic.route(conn.path)) return anthropic.post(srv, conn, a);""")
edit(rt, """fn post(srv: *Server, conn: *Conn, a: Allocator) void {""", """/// ngram-learn: one learn op (docs in families/flashnext/ngram_service.zig), run by the engine between streams.
fn ngramOp(srv: *Server, conn: *Conn, a: Allocator) void {
    const body = switch (http_body.read(conn, a, http_body.limit) catch return) {
        .ok => |b| b,
        .refused => |m| return conn.sendError(a, 400, m, null) catch {},
    };
    const reply = srv.engine.learn(a, body) catch |e| {
        const msg = std.fmt.allocPrint(a, "{{\\"error\\":\\"{s}\\"}}", .{@errorName(e)}) catch return;
        return conn.sendJson(if (e == error.NoLearner) 404 else 500, msg);
    };
    conn.sendJson(if (std.mem.startsWith(u8, reply, "{\\"error\\"")) 400 else 200, reply);
}

fn post(srv: *Server, conn: *Conn, a: Allocator) void {""")

# -- native/cuda.zig ---------------------------------------------------------------------------------------------
nc = "zig/src/native/cuda.zig"
edit(nc, """    store_vt: api.prompt_cache.Snapshots.VTable = undefined,
""", """    store_vt: api.prompt_cache.Snapshots.VTable = undefined,
    /// ngram-learn: the family's learn ops (Loaded.learn), run on the engine thread with the context current
    learn_ctx: *anyopaque = undefined,
    learn_submit: *const fn (*anyopaque, Allocator, []const u8) anyerror![]u8 = undefined,
    learn_idle: *const fn (*anyopaque) u8 = undefined,

    fn learnSubmit(p: *anyopaque, a: Allocator, body: []const u8) anyerror![]u8 {
        const h: *Host = @ptrCast(@alignCast(p));
        return h.learn_submit(h.learn_ctx, a, body);
    }

    fn learnIdle(p: *anyopaque) api.Learn.Did {
        const x = bind(p);
        return switch (x.learn_idle(x.learn_ctx)) {
            0 => .none,
            1 => .ran,
            else => .changed,
        };
    }
""")
edit(nc, """    try h.host.start();
    return .{ .engine = h.host.engine(),""", """    if (@hasField(@TypeOf(loaded), "learn")) if (loaded.learn) |l| {
        h.learn_ctx = l.ctx;
        h.learn_submit = l.submit;
        h.learn_idle = l.idle;
        h.host.learn = .{ .ctx = h, .submit = Host.learnSubmit, .idle = Host.learnIdle };
    };
    try h.host.start();
    return .{ .engine = h.host.engine(),""")

# -- flashnext ---------------------------------------------------------------------------------------------------
fx = "zig/src/families/flashnext/"
edit(fx + "cuda.zig", """pub const train_capture = @import("train_capture.zig");
""", """pub const train_capture = @import("train_capture.zig");
pub const ngram_grad = @import("ngram_grad.zig");
pub const ngram_service = @import("ngram_service.zig");
""")
cn = fx + "cuda_native.zig"
edit(cn, """    /// the prompt cache's family calls and budget (rank 0 or one rank; native/cuda.zig makes the Store)
    cache: ?lane_backend.CacheHooks = null,
};""", """    /// the prompt cache's family calls and budget (rank 0 or one rank; native/cuda.zig makes the Store)
    cache: ?lane_backend.CacheHooks = null,
    /// ngram-learn (TF_NGRAM_LEARN=1, one rank): learn ops and overlays by tag (idle: 0 none, 1 ran, 2 changed)
    learn: ?Learn = null,
};

pub const Learn = struct {
    ctx: *anyopaque,
    submit: *const fn (*anyopaque, std.mem.Allocator, []const u8) anyerror![]u8,
    idle: *const fn (*anyopaque) u8,
};""")
edit(cn, """        .cache = if (o.rank == 0) own.lanes.cacheHooks(cacheBudget(o, &own.lanes)) else null,
    };
}""", """        .cache = if (o.rank == 0) own.lanes.cacheHooks(cacheBudget(o, &own.lanes)) else null,
        .learn = if (own.service) |sv| .{ .ctx = sv, .submit = Own.learnSubmit, .idle = Own.learnIdle } else null,
    };
}""")
edit(cn, """    var backend = own.lanes.backend();
    own.lanes.follower = o.tp == 2 and o.rank == 1;""", """    if (M == @import("cuda_engine.zig")) {
        if (o.tp == 1) if (getenv("TF_NGRAM_LEARN")) |v| if (std.mem.eql(u8, v, "1")) {
            own.service = try ngram_service.Service.create(gpa, io, own.e);
            std.log.info("ngram-learn: learn ops on POST /v1/ngram/op, overlays by metadata.tag", .{});
        };
    }
    var backend = own.lanes.backend();
    own.lanes.follower = o.tp == 2 and o.rank == 1;""")
edit(cn, """const lane_backend = @import("cuda_lanes.zig");
""", """const lane_backend = @import("cuda_lanes.zig");
const ngram_service = @import("ngram_service.zig");
""")
edit(cn, """        watch: ?mirror.Watch = null,

        fn freeWire(own: *Self) void {""", """        watch: ?mirror.Watch = null,
        service: ?*ngram_service.Service = null,

        fn learnSubmit(p: *anyopaque, a: std.mem.Allocator, body: []const u8) anyerror![]u8 {
            const sv: *ngram_service.Service = @ptrCast(@alignCast(p));
            return sv.submit(a, body);
        }

        fn learnIdle(p: *anyopaque) u8 {
            const sv: *ngram_service.Service = @ptrCast(@alignCast(p));
            return @intFromEnum(sv.idle());
        }

        fn freeWire(own: *Self) void {""")
edit(cn, """            if (own.watch) |*w| w.deinit();
            own.lanes.deinit();
            own.e.deinit();""", """            if (own.watch) |*w| w.deinit();
            own.lanes.deinit();
            if (own.service) |sv| sv.destroy();
            own.e.deinit();""")

en = fx + "cuda_engine.zig"
edit(en, """    graphs_on: bool = true,""", """    graphs_on: bool = true,
    /// ngram-learn: the overlays by tag (ngram_service.zig; null: TF_NGRAM_OVERLAY's alone)
    tags: ?*const @import("ngram_service.zig").Tags = null,""")
edit(en, """    pub fn bind(e: *Engine, s: *Seq) void {
        e.bound = s;
    }""", """    pub fn bind(e: *Engine, s: *Seq) void {
        e.bound = s;
    }

    /// ngram-learn: sequence `s` reads tag `tag`'s overlay (none set: the table's own).
    pub fn tagSeq(e: *Engine, s: *Seq, tag: []const u8) void {
        s.overlay = if (e.tags) |t| t.lookup(tag) else null;
    }""")

fw = fx + "cuda_forward.zig"
edit(fw, """    /// the prompt's images and video frames (null: text)
    media: ?SeqMedia = null,
""", """    /// the prompt's images and video frames (null: text)
    media: ?SeqMedia = null,
    /// ngram-learn: the n-gram overlay this sequence reads (its request's tag); null: the table's own
    overlay: ?*const @import("ngram_overlay.zig").Overlay = null,
""")
edit(fw, """            const out = std.mem.bytesAsSlice(u16, f.pin.bytes[rows_at..][0 .. R * heads * table.width * 2]);
            try table.gather(nid, @alignCast(out));""", """            const out: []u16 = @alignCast(std.mem.bytesAsSlice(u16, f.pin.bytes[rows_at..][0 .. R * heads * table.width * 2]));
            // each sequence's rows through its own overlay (its request's tag), else the table's
            for (segs[0..windows.len]) |sg| try table.gatherOver(nid[sg.a0 * heads .. sg.a1 * heads], out[sg.a0 * heads * table.width .. sg.a1 * heads * table.width], sg.seq.overlay orelse table.overlay);""")

wt = fx + "cuda_weights.zig"
edit(wt, """    pub fn gather(t: *const NgramTable, ids: []const i64, out: []u16) !void {
        if (out.len != ids.len * t.width) return error.NgramOutLength;
        for (ids, 0..) |id, i| {
            if (id < 0 or id >= t.rows) return error.NgramIdOutOfRange;
            const dst = out[i * t.width ..][0..t.width];
            if (t.overlay) |o| if (o.get(id)) |r| {""", """    pub fn gather(t: *const NgramTable, ids: []const i64, out: []u16) !void {
        return t.gatherOver(ids, out, t.overlay);
    }

    /// `gather` through overlay `ov` instead of the table's own (ngram-learn: a sequence's tag).
    pub fn gatherOver(t: *const NgramTable, ids: []const i64, out: []u16, ov: ?*const ngram_overlay.Overlay) !void {
        if (out.len != ids.len * t.width) return error.NgramOutLength;
        for (ids, 0..) |id, i| {
            if (id < 0 or id >= t.rows) return error.NgramIdOutOfRange;
            const dst = out[i * t.width ..][0..t.width];
            if (ov) |o| if (o.get(id)) |r| {""")

ln = fx + "cuda_lanes.zig"
edit(ln, """            l.* = .{ .seq = e.newSeq(limit) catch |err| {
                _ = self.lanes.remove(s);
                self.gpa.destroy(l);
                return err;
            } };""", """            l.* = .{ .seq = e.newSeq(limit) catch |err| {
                _ = self.lanes.remove(s);
                self.gpa.destroy(l);
                return err;
            } };
            if (@hasDecl(E, "tagSeq")) e.tagSeq(l.seq, s.tag);""")

for path, text in FILES.items():
    pathlib.Path(path).write_text(text)
print("patched serve learn:", len(FILES), "files")
