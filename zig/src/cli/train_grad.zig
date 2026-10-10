//! ngram-learn: the in-engine training pass checked against the PyTorch reference. DIR/case.json holds a lesson
//! (tokens, answer rows `at`, `targets`); this writes its loss, every layer's input and d(rows) for
//! tools/train_grad_check.py to compare.
const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("flashnext");

const Engine = flashnext.Engine;
const Allocator = std.mem.Allocator;

fn ids(gpa: Allocator, v: std.json.Value) ![]u32 {
    const out = try gpa.alloc(u32, v.array.items.len);
    for (v.array.items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

pub fn gradCheck(gpa: Allocator, io: std.Io, e: *Engine, dir: []const u8) !u8 {
    var pb: [512]u8 = undefined;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&pb, "{s}/case.json", .{dir}), gpa, .limited(1 << 24));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const ob = parsed.value.object;
    const tokens = try ids(gpa, ob.get("tokens").?);
    defer gpa.free(tokens);
    const at = try ids(gpa, ob.get("at").?);
    defer gpa.free(at);
    const targets = try ids(gpa, ob.get("targets").?);
    defer gpa.free(targets);
    const c = &e.c;
    const r = tokens.len;
    const wd: usize = c.streams * c.hidden;
    // the n-gram rows layer 1 looks up, as the engine decodes them
    const p = e.w.layers[1].ple orelse return error.NoNgramLayer;
    const heads = p.ngram.heads;
    var hist: [4]i64 = undefined;
    p.ngram.initialHistory(hist[0 .. p.ngram.n - 1]);
    const toks = try gpa.alloc(i64, r);
    defer gpa.free(toks);
    for (tokens, toks) |a, *b| b.* = a;
    const rid = try gpa.alloc(i64, r * heads);
    defer gpa.free(rid);
    try p.ngram.ids(hist[0 .. p.ngram.n - 1], toks, rid);
    const table = &e.w.table.?;
    const bits = try gpa.alloc(u16, r * heads * table.width);
    defer gpa.free(bits);
    try table.gather(rid, bits);
    const ef = try gpa.alloc(f32, bits.len);
    defer gpa.free(ef);
    for (bits, ef) |b, *f| f.* = @bitCast(@as(u32, b) << 16);
    var eb = try cuda.DeviceBuffer.fromHost(e.ctx.d, std.mem.sliceAsBytes(ef));
    defer eb.free();
    var de = try cuda.DeviceBuffer.alloc(e.ctx.d, ef.len * 4);
    defer de.free();
    var dump = try cuda.DeviceBuffer.alloc(e.ctx.d, (c.layers + 1) * r * wd * 4);
    defer dump.free();
    var m: flashnext.train_model.Model = undefined;
    try m.init(gpa, e.ctx, &e.f, &e.w, c, 256, 6.0);
    defer m.deinit();
    var routes = try cuda.DeviceBuffer.alloc(e.ctx.d, c.layers * r * c.top_k * 4);
    defer routes.free();
    m.routes = routes.ptr;
    e.graphs_on = false;
    const t0 = std.Io.Timestamp.now(io, .awake);
    const res = try m.lesson(tokens, at, targets, eb.ptr, de.ptr, dump.ptr);
    const dt = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
    m.routes = null;
    const t1 = std.Io.Timestamp.now(io, .awake);
    _ = try m.lesson(tokens, at, targets, eb.ptr, de.ptr, null);
    const dt2 = @as(f64, @floatFromInt(t1.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
    std.debug.print("loss {d:.5} over {d} answer rows; first pass {d:.2}s (builds the Wᵀ cache), second {d:.2}s; Wᵀ cache {d:.2} GiB; arena peak {d:.2} GiB\n", .{ res.loss, at.len, dt, dt2, @as(f64, @floatFromInt(m.weights.cached())) / (1 << 30), @as(f64, @floatFromInt(m.arena.peak)) / (1 << 30) });
    const save = struct {
        fn f(g: Allocator, i: std.Io, d: []const u8, name: []const u8, buf: cuda.DeviceBuffer) !void {
            const host = try g.alloc(u8, buf.len);
            defer g.free(host);
            try buf.download(0, host);
            var b: [512]u8 = undefined;
            try std.Io.Dir.cwd().writeFile(i, .{ .sub_path = try std.fmt.bufPrint(&b, "{s}/{s}", .{ d, name }), .data = host });
        }
    }.f;
    try save(gpa, io, dir, "engine_de.bin", de);
    try save(gpa, io, dir, "engine_routes.bin", routes);
    try save(gpa, io, dir, "engine_hidden.bin", dump);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/engine_ids.bin", .{dir}), .data = std.mem.sliceAsBytes(rid) });
    var lb: [128]u8 = undefined;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/engine_loss.json", .{dir}), .data = try std.fmt.bufPrint(&lb, "{{\"loss\":{d},\"seconds\":{d},\"seconds_warm\":{d}}}", .{ res.loss, dt, dt2 }) });
    return 0;
}
