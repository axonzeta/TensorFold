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
        } else if (std.mem.eql(u8, op, "gemm_nn")) {
            const m = int(o, "m");
            const n = int(o, "n");
            const kk = int(o, "k");
            const x = try b.in(str(o, "a"));
            const w = try b.in(str(o, "b"));
            const d = try b.out(m * kk * 4);
            try ops.gemmNN(x, w, d, m, n, kk, .f32, false);
            try stream.synchronize();
            try b.save(str(o, "out"), d, m * kk * 4);
        } else if (std.mem.eql(u8, op, "transpose")) {
            const rows = int(o, "rows");
            const cols = int(o, "cols");
            const x = try b.in(str(o, "in"));
            const y = try b.out(rows * cols * 2);
            try ops.transpose(x, y, rows, cols);
            try stream.synchronize();
            try b.save(str(o, "out"), y, rows * cols * 2);
        } else if (std.mem.eql(u8, op, "rms")) {
            const r = int(o, "r");
            const g = int(o, "g");
            const d = int(o, "d");
            const mode: flashnext.train_ops.Ops.Gain = @enumFromInt(@as(i32, @intCast(int(o, "mode"))));
            const x = try b.in(str(o, "x"));
            const w = if (mode == .none) 0 else try b.in(str(o, "w"));
            const dy = try b.in(str(o, "dy"));
            const y = try b.out(r * g * d * 4);
            const rinv = try b.out(r * g * 4);
            const dx = try b.out(r * g * d * 4);
            try ops.rmsFwd(x, w, mode, y, rinv, r, g, d, 1e-6);
            try ops.rmsBwd(dy, x, w, mode, rinv, dx, r, g, d, false);
            try stream.synchronize();
            try b.save(str(o, "y"), y, r * g * d * 4);
            try b.save(str(o, "dx"), dx, r * g * d * 4);
        } else if (std.mem.eql(u8, op, "ew")) {
            const n = int(o, "n");
            const f: flashnext.train_ops.Ops.Fn = @enumFromInt(@as(i32, @intCast(int(o, "f"))));
            const alpha: f32 = @floatCast(o.get("alpha").?.float);
            const beta: f32 = @floatCast(o.get("beta").?.float);
            const x = try b.in(str(o, "x"));
            const dy = try b.in(str(o, "dy"));
            const y = try b.out(n * 4);
            const dx = try b.out(n * 4);
            try ops.ew(f, x, y, n, alpha, beta);
            try ops.ewBwd(f, dy, x, dx, n, alpha, beta, false);
            try stream.synchronize();
            try b.save(str(o, "y"), y, n * 4);
            try b.save(str(o, "dx"), dx, n * 4);
        } else if (std.mem.eql(u8, op, "mix")) {
            const r = int(o, "r");
            const sn = int(o, "s");
            const d = int(o, "d");
            const u = try b.in(str(o, "u"));
            const hn = try b.in(str(o, "hn"));
            const dm = try b.in(str(o, "dm"));
            const m = try b.out(r * d * 4);
            const du = try b.out(r * sn * d * 4);
            const dhn = try b.out(r * sn * d * 4);
            try ops.mixFwd(u, hn, m, r, sn, d);
            try ops.mixBwd(dm, u, hn, du, dhn, r, sn, d, false);
            try stream.synchronize();
            try b.save(str(o, "m"), m, r * d * 4);
            try b.save(str(o, "du"), du, r * sn * d * 4);
            try b.save(str(o, "dhn"), dhn, r * sn * d * 4);
        } else if (std.mem.eql(u8, op, "inject")) {
            const r = int(o, "r");
            const sn = int(o, "s");
            const d = int(o, "d");
            const res = try b.in(str(o, "res"));
            const inj = try b.in(str(o, "inj"));
            const br = try b.in(str(o, "br"));
            const dh = try b.in(str(o, "dh"));
            const h = try b.out(r * sn * d * 4);
            const dbr = try b.out(r * d * 4);
            const dinj = try b.out(r * sn * 4);
            try ops.injectFwd(res, inj, br, h, r, sn, d);
            try ops.injectBwd(dh, inj, br, dbr, dinj, r, sn, d);
            try stream.synchronize();
            try b.save(str(o, "h"), h, r * sn * d * 4);
            try b.save(str(o, "dbr"), dbr, r * d * 4);
            try b.save(str(o, "dinj"), dinj, r * sn * 4);
        } else if (std.mem.eql(u8, op, "rope")) {
            const r = int(o, "r");
            const h = int(o, "h");
            const hd = int(o, "hd");
            const rot = int(o, "rot");
            const x = try b.in(str(o, "x"));
            const cs = try b.in(str(o, "cos"));
            const sn = try b.in(str(o, "sin"));
            const dy = try b.in(str(o, "dy"));
            const y = try b.out(r * h * hd * 4);
            const dx = try b.out(r * h * hd * 4);
            try ops.rope(x, cs, sn, y, r, h, hd, rot, false);
            try ops.rope(dy, cs, sn, dx, r, h, hd, rot, true);
            try stream.synchronize();
            try b.save(str(o, "y"), y, r * h * hd * 4);
            try b.save(str(o, "dx"), dx, r * h * hd * 4);
        } else if (std.mem.eql(u8, op, "attn")) {
            const r = int(o, "r");
            const hq = int(o, "hq");
            const hk = int(o, "hk");
            const d = int(o, "d");
            const scale: f32 = @floatCast(o.get("scale").?.float);
            const q = try b.in(str(o, "q"));
            const kt = try b.in(str(o, "k"));
            const v = try b.in(str(o, "v"));
            const dout = try b.in(str(o, "do"));
            const p = try b.out(hq * r * r * 4);
            const ds = try b.out(hq * r * r * 4);
            const out = try b.out(r * hq * d * 4);
            const dq = try b.out(r * hq * d * 4);
            const dk = try b.out(r * hk * d * 4);
            const dv = try b.out(r * hk * d * 4);
            try ops.attnFwd(q, kt, v, p, out, r, hq, hk, d, scale);
            try ops.attnBwd(q, kt, v, p, dout, ds, dq, dk, dv, r, hq, hk, d, scale);
            try stream.synchronize();
            try b.save(str(o, "o"), out, r * hq * d * 4);
            try b.save(str(o, "dq"), dq, r * hq * d * 4);
            try b.save(str(o, "dk"), dk, r * hk * d * 4);
            try b.save(str(o, "dv"), dv, r * hk * d * 4);
        } else if (std.mem.eql(u8, op, "conv")) {
            const r = int(o, "r");
            const c = int(o, "c");
            const kk = int(o, "k");
            const dil = int(o, "dil");
            const x = try b.in(str(o, "x"));
            const w = try b.in(str(o, "w"));
            const dy = try b.in(str(o, "dy"));
            const y = try b.out(r * c * 4);
            const dx = try b.out(r * c * 4);
            try ops.convFwd(x, w, y, r, c, kk, dil);
            try ops.convBwd(dy, w, dx, r, c, kk, dil, false);
            try stream.synchronize();
            try b.save(str(o, "y"), y, r * c * 4);
            try b.save(str(o, "dx"), dx, r * c * 4);
        } else if (std.mem.eql(u8, op, "gates")) {
            const r = int(o, "r");
            const h = int(o, "h");
            const ain = try b.in(str(o, "a"));
            const bin = try b.in(str(o, "b"));
            const alog = try b.in(str(o, "alog"));
            const dt = try b.in(str(o, "dt"));
            const dg = try b.in(str(o, "dg"));
            const dbeta = try b.in(str(o, "dbeta"));
            const g = try b.out(r * h * 4);
            const beta = try b.out(r * h * 4);
            const da = try b.out(r * h * 4);
            const db = try b.out(r * h * 4);
            try ops.gdnGates(ain, bin, alog, dt, g, beta, r, h);
            try ops.gdnGatesBwd(ain, beta, alog, dt, dg, dbeta, da, db, r, h);
            try stream.synchronize();
            inline for (.{ .{ "g", g }, .{ "beta", beta }, .{ "da", da }, .{ "db", db } }) |e| try b.save(str(o, e[0]), e[1], r * h * 4);
        } else if (std.mem.eql(u8, op, "delta")) {
            const r = int(o, "r");
            const h = int(o, "h");
            const dk = int(o, "dk");
            const dv = int(o, "dv");
            const q = try b.in(str(o, "q"));
            const kt = try b.in(str(o, "k"));
            const v = try b.in(str(o, "v"));
            const g = try b.in(str(o, "g"));
            const beta = try b.in(str(o, "beta"));
            const dout = try b.in(str(o, "do"));
            const out = try b.out(r * h * dv * 4);
            const st = try b.out(h * (r + 1) * dk * dv * 4);
            const dq = try b.out(r * h * dk * 4);
            const dkk = try b.out(r * h * dk * 4);
            const dvv = try b.out(r * h * dv * 4);
            const dg = try b.out(r * h * 4);
            const dbeta = try b.out(r * h * 4);
            try ops.deltaFwd(q, kt, v, g, beta, out, st, r, h, dk, dv);
            const scratch = try b.out(flashnext.train_ops.Ops.deltaScratch(r, h, dk, dv) * 4);
            try ops.deltaBwd(q, kt, v, g, beta, dout, st, dq, dkk, dvv, dg, dbeta, r, h, dk, dv, scratch);
            try stream.synchronize();
            try b.save(str(o, "o"), out, r * h * dv * 4);
            try b.save(str(o, "dq_out"), dq, r * h * dk * 4);
            try b.save(str(o, "dk_out"), dkk, r * h * dk * 4);
            try b.save(str(o, "dv_out"), dvv, r * h * dv * 4);
            try b.save(str(o, "dg_out"), dg, r * h * 4);
            try b.save(str(o, "dbeta_out"), dbeta, r * h * 4);
        } else if (std.mem.eql(u8, op, "sdot")) {
            const rs = int(o, "rs");
            const d = int(o, "d");
            const scale: f32 = @floatCast(o.get("scale").?.float);
            const x = try b.in(str(o, "a"));
            const y = try b.in(str(o, "b"));
            const dout = try b.in(str(o, "dout"));
            const out = try b.out(rs * 4);
            const dx = try b.out(rs * d * 4);
            const dy = try b.out(rs * d * 4);
            try ops.sdot(x, y, out, rs, d, scale);
            try ops.sdotBwd(dout, x, y, dx, dy, rs, d, scale, false);
            try stream.synchronize();
            try b.save(str(o, "out"), out, rs * 4);
            try b.save(str(o, "da"), dx, rs * d * 4);
            try b.save(str(o, "db"), dy, rs * d * 4);
        } else if (std.mem.eql(u8, op, "topk")) {
            const r = int(o, "r");
            const e = int(o, "e");
            const kk = int(o, "k");
            const lg = try b.in(str(o, "logits"));
            const dw = try b.in(str(o, "dw"));
            const ids = try b.out(r * kk * 4);
            const w = try b.out(r * kk * 4);
            const dl = try b.out(r * e * 4);
            try ops.topk(lg, ids, w, r, e, kk);
            try ops.topkBwd(ids, w, dw, dl, r, e, kk);
            try stream.synchronize();
            try b.save(str(o, "ids"), ids, r * kk * 4);
            try b.save(str(o, "w"), w, r * kk * 4);
            try b.save(str(o, "dlogits"), dl, r * e * 4);
        } else if (std.mem.eql(u8, op, "gather_scatter")) {
            const n = int(o, "n");
            const d = int(o, "d");
            const rows_n = int(o, "rows");
            const src = try b.in(str(o, "src"));
            const ids = try b.in(str(o, "ids"));
            const sc = try b.in(str(o, "scale"));
            const g = try b.out(n * d * 4);
            const acc = try b.out(rows_n * d * 4);
            try ops.gather(src, ids, g, n, d);
            try ops.scatterAdd(g, ids, sc, acc, n, d);
            try stream.synchronize();
            try b.save(str(o, "gathered"), g, n * d * 4);
            try b.save(str(o, "scattered"), acc, rows_n * d * 4);
        } else if (std.mem.eql(u8, op, "xent")) {
            const r = int(o, "r");
            const v = int(o, "v");
            const lg = try b.in(str(o, "logits"));
            const tg = try b.in(str(o, "target"));
            const loss = try b.out(r * 4);
            const dl = try b.out(r * v * 4);
            try ops.xent(lg, tg, loss, dl, r, v, @floatCast(o.get("gscale").?.float));
            try stream.synchronize();
            try b.save(str(o, "loss"), loss, r * 4);
            try b.save(str(o, "dlogits"), dl, r * v * 4);
        } else {
            std.debug.print("unknown op {s}\n", .{op});
            return 2;
        }
        std.debug.print("op {s}: done\n", .{op});
    }
    return 0;
}
