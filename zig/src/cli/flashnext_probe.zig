//! ngram-learn probes for Flash Next on CUDA. A case: a prompt, a continuation scored teacher-forced (each token's
//! log-probability), optional greedy generation, the n-gram rows muted at chosen positions (Forward.ple_mute) and an
//! overlay of replacement rows: explicit bf16 rows ("overlay": {"ids", "rows"}) or table rows ("copy": [[dst, src]]).
//! `score` runs a file of cases; `worker` takes one JSON line at a time on stdin and answers one line on stdout,
//! with a persistent base overlay ({"op": "base", ...}) under every case's own.
const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("flashnext");
const cli = @import("flashnext_cuda.zig");

const Engine = flashnext.Engine;
const Overlay = flashnext.ngram_overlay.Overlay;
const Allocator = std.mem.Allocator;

/// One row's log-softmax at `target` and its argmax, from bf16 logits on the device at `ptr`.
const RowScore = struct { logprob: f32, argmax: u32 };

fn scoreRow(e: *Engine, host: []u16, ptr: u64, target: u32) !RowScore {
    try e.stream.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = ptr, .len = host.len * 2 };
    try buf.download(0, std.mem.sliceAsBytes(host));
    var best: f32 = -std.math.inf(f32);
    var at: u32 = 0;
    for (host, 0..) |x, i| {
        const v = bf16f(x);
        if (v > best) {
            best = v;
            at = @intCast(i);
        }
    }
    var sum: f64 = 0;
    for (host) |x| sum += @exp(@as(f64, bf16f(x) - best));
    const lse = @as(f64, best) + @log(sum);
    return .{ .logprob = @floatCast(@as(f64, bf16f(host[target])) - lse), .argmax = at };
}

fn bf16f(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

fn idsOf(gpa: Allocator, v: ?std.json.Value) ![]u32 {
    const items = if (v) |x| x.array.items else &.{};
    const out = try gpa.alloc(u32, items.len);
    for (items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

/// The probe's state across cases: the engine, a logits row on the host and the persistent base overlay.
const Probe = struct {
    gpa: Allocator,
    io: std.Io,
    e: *Engine,
    host: []u16,
    base: Overlay,

    fn init(gpa: Allocator, io: std.Io, e: *Engine) !Probe {
        if (e.world != 1) return error.ProbeNeedsOneRank;
        e.graphs_on = false;
        e.absorbing = false;
        e.setSampling(null);
        const host = try gpa.alloc(u16, e.g.head_n);
        errdefer gpa.free(host);
        return .{ .gpa = gpa, .io = io, .e = e, .host = host, .base = try Overlay.init(gpa, e.w.table.?.width) };
    }

    fn deinit(p: *Probe) void {
        p.base.deinit();
        p.gpa.free(p.host);
    }

    /// {"op": "base", "clear"?, "copy"?, "overlay"?}: the base overlay cleared and/or added to.
    fn setBase(p: *Probe, ob: std.json.ObjectMap, out: *std.ArrayList(u8)) !void {
        if (ob.get("clear")) |c| if (c.bool) p.base.clear();
        try caseOverlay(p.gpa, &p.e.w.table.?, ob, &p.base);
        try out.print(p.gpa, "{{\"base_rows\":{d}}}", .{p.base.count()});
    }

    /// One case, its result object appended to `out`.
    fn run(p: *Probe, ob: std.json.ObjectMap, out: *std.ArrayList(u8)) !void {
        const gpa = p.gpa;
        const e = p.e;
        const prompt = try idsOf(gpa, ob.get("prompt"));
        defer gpa.free(prompt);
        const cont = try idsOf(gpa, ob.get("cont"));
        defer gpa.free(cont);
        const mute = try gpa.alloc(bool, prompt.len + cont.len);
        defer gpa.free(mute);
        @memset(mute, false);
        var muted: usize = 0;
        if (ob.get("mute")) |m| for (m.array.items) |x| {
            const at: usize = @intCast(x.integer);
            if (at < mute.len and !mute[at]) {
                mute[at] = true;
                muted += 1;
            }
        };
        e.f.ple_mute = if (muted > 0) mute else null;
        defer e.f.ple_mute = null;
        const table = &e.w.table.?;
        var ov = try Overlay.init(gpa, table.width);
        defer ov.deinit();
        try caseOverlay(gpa, table, ob, &ov);
        if (p.base.count() > 0) ov.under = &p.base;
        const before = table.overlay;
        table.overlay = if (ov.count() > 0 or ov.under != null) &ov else null;
        defer table.overlay = before;
        const name = if (ob.get("name")) |n| n.string else "";
        try out.print(gpa, "{{\"name\":{f},\"muted\":{d},\"overlay_rows\":{d},\"base_rows\":{d}", .{ std.json.fmt(name, .{}), muted, ov.count(), p.base.count() });
        if (cont.len > 0) {
            try e.reset();
            _ = try e.prefill(prompt, null);
            var total: f64 = 0;
            var hits: usize = 0;
            var lps: std.ArrayList(u8) = .empty;
            defer lps.deinit(gpa);
            var maxes: std.ArrayList(u8) = .empty;
            defer maxes.deinit(gpa);
            for (cont, 0..) |tok, i| {
                const r = try scoreRow(e, p.host, if (i == 0) e.last_logits else e.buf.b.logits, tok);
                total += r.logprob;
                hits += @intFromBool(r.argmax == tok);
                try lps.print(gpa, "{s}{d:.5}", .{ if (i == 0) "" else ",", r.logprob });
                try maxes.print(gpa, "{s}{d}", .{ if (i == 0) "" else ",", r.argmax });
                if (i + 1 < cont.len) {
                    var one: [1]u32 = undefined;
                    try e.forward(&.{tok});
                    try e.sample(1, e.pos() + 1, &one);
                    try e.commit(1, 1);
                }
            }
            try out.print(gpa, ",\"logprobs\":[{s}],\"sum\":{d:.5},\"greedy_hits\":{d},\"argmax\":[{s}]", .{ lps.items, total, hits, maxes.items });
        }
        if (ob.get("gen")) |n| if (n.integer > 0) {
            const g = try cli.generate(gpa, p.io, e, prompt, @intCast(n.integer), null, false, true);
            defer gpa.free(g.tokens);
            try out.appendSlice(gpa, ",\"gen\":[");
            for (g.tokens, 0..) |t, i| try out.print(gpa, "{s}{d}", .{ if (i == 0) "" else ",", t });
            try out.appendSlice(gpa, "]");
        };
        try out.appendSlice(gpa, "}");
    }
};

/// A file of cases ({"cases": [...]}) into one report ({"cases": [...]}).
pub fn score(gpa: Allocator, io: std.Io, e: *Engine, path: []const u8, report_path: ?[]const u8) !u8 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const cases = parsed.value.object.get("cases").?.array.items;
    var p = try Probe.init(gpa, io, e);
    defer p.deinit();
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.appendSlice(gpa, "{\"cases\":[");
    for (cases, 0..) |c, n| {
        if (n > 0) try report.append(gpa, ',');
        try p.run(c.object, &report);
        std.debug.print("score {d}/{d}\n", .{ n + 1, cases.len });
    }
    try report.appendSlice(gpa, "]}");
    if (report_path) |rp| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = rp, .data = report.items });
    return 0;
}

/// One JSON object a line on stdin, one answer a line on stdout ("ready" first); {"op": "quit"} or EOF ends it.
pub fn worker(gpa: Allocator, io: std.Io, e: *Engine) !u8 {
    var p = try Probe.init(gpa, io, e);
    defer p.deinit();
    const in_buf = try gpa.alloc(u8, 256 << 20);
    defer gpa.free(in_buf);
    var in = std.Io.File.stdin().reader(io, in_buf);
    var out_buf: [64 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    try out.interface.writeAll("{\"ready\":true}\n");
    try out.interface.flush();
    var line_out: std.ArrayList(u8) = .empty;
    defer line_out.deinit(gpa);
    while (try in.interface.takeDelimiter('\n')) |line| {
        if (std.mem.trim(u8, line, " \r\t").len == 0) continue;
        line_out.clearRetainingCapacity();
        const parsed = std.json.parseFromSlice(std.json.Value, gpa, line, .{}) catch {
            try out.interface.writeAll("{\"error\":\"bad json\"}\n");
            try out.interface.flush();
            continue;
        };
        defer parsed.deinit();
        const ob = parsed.value.object;
        const op = if (ob.get("op")) |o| o.string else "case";
        if (std.mem.eql(u8, op, "quit")) break;
        const result = if (std.mem.eql(u8, op, "base")) p.setBase(ob, &line_out) else p.run(ob, &line_out);
        result catch |err| {
            line_out.clearRetainingCapacity();
            try line_out.print(gpa, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
        };
        try line_out.append(gpa, '\n');
        try out.interface.writeAll(line_out.items);
        try out.interface.flush();
    }
    return 0;
}

/// A case's overlay: "copy" pairs read the source rows from the table as it stands, then "overlay" rows (bf16 bits).
fn caseOverlay(gpa: Allocator, table: anytype, ob: std.json.ObjectMap, ov: *Overlay) !void {
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

/// ngram-learn B1 check: Wᵀ of a few block-FP8 faces through train_dequant (identity rows on the serving matmul),
/// dumped as bf16 [k][n] files beside a JSON of their shapes, for tools/dequant_check.py to compare.
pub fn dequantCheck(gpa: Allocator, io: std.Io, e: *Engine, dir: []const u8) !u8 {
    const td = flashnext.train_dequant;
    var id = try td.Ident.init(gpa, e.ctx.d, 6144);
    defer id.deinit(gpa);
    const Face = struct { name: []const u8, l: flashnext.fp8.Linear };
    var faces: std.ArrayList(Face) = .empty;
    defer faces.deinit(gpa);
    for (e.w.layers, 0..) |l, i| {
        if (i > 3) break;
        var nb: [64]u8 = undefined;
        if (l.gdn) |g| {
            if (g.proj8) |x| try faces.append(gpa, .{ .name = try gpa.dupe(u8, try std.fmt.bufPrint(&nb, "L{d}_gdn_proj8", .{i})), .l = x });
            if (g.out8) |x| try faces.append(gpa, .{ .name = try gpa.dupe(u8, try std.fmt.bufPrint(&nb, "L{d}_gdn_out8", .{i})), .l = x });
        }
        if (l.attn) |a| {
            if (a.proj8) |x| try faces.append(gpa, .{ .name = try gpa.dupe(u8, try std.fmt.bufPrint(&nb, "L{d}_attn_proj8", .{i})), .l = x });
            if (a.o8) |x| try faces.append(gpa, .{ .name = try gpa.dupe(u8, try std.fmt.bufPrint(&nb, "L{d}_attn_o8", .{i})), .l = x });
        }
        if (l.moe.shared8) |s| {
            try faces.append(gpa, .{ .name = try gpa.dupe(u8, try std.fmt.bufPrint(&nb, "L{d}_shared_gu8", .{i})), .l = s.gu });
            try faces.append(gpa, .{ .name = try gpa.dupe(u8, try std.fmt.bufPrint(&nb, "L{d}_shared_down8", .{i})), .l = s.down });
        }
    }
    var meta: std.ArrayList(u8) = .empty;
    defer meta.deinit(gpa);
    try meta.appendSlice(gpa, "{");
    for (faces.items, 0..) |fc, n| {
        defer gpa.free(fc.name);
        const bytes = @as(usize, fc.l.k) * fc.l.n * 2;
        var out = try cuda.DeviceBuffer.alloc(e.ctx.d, bytes);
        defer out.free();
        const t0 = std.Io.Timestamp.now(io, .awake);
        try td.fp8Transposed(&e.f, &id, fc.l, out.ptr);
        const dt = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
        const host = try gpa.alloc(u8, bytes);
        defer gpa.free(host);
        try out.download(0, host);
        var pb: [512]u8 = undefined;
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/{s}.bin", .{ dir, fc.name }), .data = host });
        try meta.print(gpa, "{s}\"{s}\":{{\"k\":{d},\"n\":{d},\"seconds\":{d:.3}}}", .{ if (n == 0) "" else ",", fc.name, fc.l.k, fc.l.n, dt });
        std.debug.print("{s}: k {d} n {d} in {d:.3}s\n", .{ fc.name, fc.l.k, fc.l.n, dt });
    }
    // the int4 side: experts 0, 1 and 511 of layer 0 (gate, up, down) and the head's first 8192 vocabulary rows
    var tk = try flashnext.train_kernels.Kernels.load(e.ctx);
    defer tk.deinit();
    const x = e.w.layers[0].moe.int4 orelse return error.NoInt4Experts;
    for ([_]usize{ 0, 1, 511 }) |ex| for (0..3) |m| {
        const n: usize = if (m < 2) x.width else x.dims;
        const k: usize = if (m < 2) x.dims else x.width;
        var out = try cuda.DeviceBuffer.alloc(e.ctx.d, n * k * 2);
        defer out.free();
        try tk.expert(e.f.s, x, ex, m, out.ptr);
        try e.f.s.synchronize();
        const host = try gpa.alloc(u8, n * k * 2);
        defer gpa.free(host);
        try out.download(0, host);
        var nb: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, "L0_expert{d}_{s}", .{ ex, ([_][]const u8{ "gate", "up", "down" })[m] });
        var pb: [512]u8 = undefined;
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/{s}.bin", .{ dir, name }), .data = host });
        try meta.print(gpa, ",\"{s}\":{{\"n\":{d},\"k\":{d},\"seconds\":0}}", .{ name, n, k });
    };
    const h = e.w.head4 orelse return error.NoInt4Head;
    {
        const n: usize = h.n;
        const k: usize = h.k;
        var out = try cuda.DeviceBuffer.alloc(e.ctx.d, n * k * 2);
        defer out.free();
        const t0 = std.Io.Timestamp.now(io, .awake);
        try tk.dequantInt4(e.f.s, h.w, h.s, out.ptr, n, k, h.gs);
        try e.f.s.synchronize();
        const dt = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
        const rows: usize = 8192;
        const host = try gpa.alloc(u8, rows * k * 2);
        defer gpa.free(host);
        try out.download(0, host);
        var pb: [512]u8 = undefined;
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/head8192.bin", .{dir}), .data = host });
        try meta.print(gpa, ",\"head8192\":{{\"n\":{d},\"k\":{d},\"seconds\":{d:.3}}}", .{ rows, k, dt });
        std.debug.print("head: n {d} k {d} gs {d} dequantised in {d:.3}s\n", .{ n, k, h.gs, dt });
    }
    try meta.appendSlice(gpa, "}");
    var pb: [512]u8 = undefined;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/faces.json", .{dir}), .data = meta.items });
    return 0;
}
