//! ngram-learn: the backward's ops checked without weights. DIR/ops.json lists ops over .bin files in DIR (written by
//! tools/train_ops_check.py); each op's outputs are written back as .bin for the script to compare against PyTorch.
const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("flashnext");

const Allocator = std.mem.Allocator;
const Obj = std.json.ObjectMap;

const Bufs = struct {
    gpa: Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    dir: []const u8,
    live: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn path(b: *Bufs, name: []const u8, pb: []u8) ![]const u8 {
        return std.fmt.bufPrint(pb, "{s}/{s}", .{ b.dir, name });
    }

    /// The file `name` on the device.
    fn in(b: *Bufs, name: []const u8) !u64 {
        var pb: [512]u8 = undefined;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(b.io, try b.path(name, &pb), b.gpa, .limited(1 << 32));
        defer b.gpa.free(bytes);
        const buf = try cuda.DeviceBuffer.fromHost(b.d, bytes);
        try b.live.append(b.gpa, buf);
        return buf.ptr;
    }

    /// A zeroed device buffer of `len` bytes.
    fn out(b: *Bufs, len: usize) !u64 {
        const buf = try cuda.DeviceBuffer.alloc(b.d, len);
        try buf.fill8(0, null);
        try b.live.append(b.gpa, buf);
        return buf.ptr;
    }

    /// `len` bytes at `ptr` into the file `name`.
    fn save(b: *Bufs, name: []const u8, ptr: u64, len: usize) !void {
        const host = try b.gpa.alloc(u8, len);
        defer b.gpa.free(host);
        const buf: cuda.DeviceBuffer = .{ .d = b.d, .ptr = ptr, .len = len };
        try buf.download(0, host);
        var pb: [512]u8 = undefined;
        try std.Io.Dir.cwd().writeFile(b.io, .{ .sub_path = try b.path(name, &pb), .data = host });
    }

    fn deinit(b: *Bufs) void {
        for (b.live.items) |*x| x.free();
        b.live.deinit(b.gpa);
    }
};

fn int(o: Obj, key: []const u8) usize {
    return @intCast(o.get(key).?.integer);
}

fn str(o: Obj, key: []const u8) []const u8 {
    return o.get(key).?.string;
}

pub fn opCheck(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !u8 {
    var pb: [512]u8 = undefined;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&pb, "{s}/ops.json", .{dir}), gpa, .limited(1 << 24));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var k = try flashnext.train_kernels.Kernels.load(ctx);
    defer k.deinit();
    var ops = try flashnext.train_ops.Ops.init(gpa, ctx.d, stream, &k);
    defer ops.deinit();
    for (parsed.value.array.items) |item| {
        const o = item.object;
        const op = str(o, "op");
        var b: Bufs = .{ .gpa = gpa, .io = io, .d = ctx.d, .dir = dir };
        defer b.deinit();
        if (std.mem.eql(u8, op, "gemm")) {
            const m = int(o, "m");
            const n = int(o, "n");
            const kk = int(o, "k");
            const f32_out = std.mem.eql(u8, str(o, "out_dtype"), "f32");
            const add = o.get("add").?.bool;
            const x = try b.in(str(o, "x"));
            const w = try b.in(str(o, "w"));
            const d = if (add) try b.in(str(o, "d")) else try b.out(m * n * 4);
            try ops.gemm(x, w, d, m, n, kk, if (f32_out) .f32 else .bf16, add);
            try stream.synchronize();
            try b.save(str(o, "out"), d, m * n * @as(usize, if (f32_out) 4 else 2));
        } else if (std.mem.eql(u8, op, "transpose")) {
            const rows = int(o, "rows");
            const cols = int(o, "cols");
            const x = try b.in(str(o, "in"));
            const y = try b.out(rows * cols * 2);
            try ops.transpose(x, y, rows, cols);
            try stream.synchronize();
            try b.save(str(o, "out"), y, rows * cols * 2);
        } else {
            std.debug.print("unknown op {s}\n", .{op});
            return 2;
        }
        std.debug.print("op {s}: done\n", .{op});
    }
    return 0;
}
