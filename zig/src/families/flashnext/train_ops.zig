//! ngram-learn training: GEMMs through cuBLASLt (bf16 in, fp32 accumulate, fp32 or bf16 out, optionally added to the
//! output) and fn_train's small kernels. The backward's dx = dy·W is `gemm(dy, Wᵀ)` with Wᵀ [k][n] as the weight.
const std = @import("std");
const cuda = @import("cuda");
const lt = cuda.cublaslt;
const tk = @import("train_kernels.zig");

pub const Ops = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const tk.Kernels,
    lib: lt.Library,
    plans: std.AutoHashMapUnmanaged(Key, lt.Linear) = .empty,
    work: cuda.DeviceBuffer,

    const Key = struct { m: u64, n: u64, k: u64, out: lt.DataType };
    const workspace: usize = 32 << 20;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, k: *const tk.Kernels) !Ops {
        var lib = try lt.Library.open();
        errdefer lib.close();
        return .{ .gpa = gpa, .d = d, .s = s, .k = k, .lib = lib, .work = try cuda.DeviceBuffer.alloc(d, workspace) };
    }

    pub fn deinit(o: *Ops) void {
        var it = o.plans.valueIterator();
        while (it.next()) |p| p.deinit();
        o.plans.deinit(o.gpa);
        o.work.free();
        o.lib.close();
    }

    /// D[m][n] (= or +=) X[m][k] · W[n][k]ᵀ; X and W bf16, D fp32 (`out` .f32) or bf16.
    pub fn gemm(o: *Ops, x: u64, w: u64, d: u64, m: usize, n: usize, k: usize, out: lt.DataType, add: bool) !void {
        const key: Key = .{ .m = m, .n = n, .k = k, .out = out };
        const gop = try o.plans.getOrPut(o.gpa, key);
        if (!gop.found_existing) {
            gop.value_ptr.* = lt.Linear.init(&o.lib, m, n, k, out, workspace) catch |e| {
                _ = o.plans.remove(key);
                return e;
            };
        }
        const p = gop.value_ptr;
        const one: f32 = 1;
        const beta: f32 = if (add) 1 else 0;
        try o.lib.check(o.lib.api.cublasLtMatmul(p.handle, p.desc, &one, w, p.a, x, p.b, &beta, d, p.c, d, p.c, &p.algo, o.work.ptr, workspace, o.s.handle), "cublasLtMatmul");
    }

    /// bf16 [rows][cols] at `in` -> [cols][rows] at `out`.
    pub fn transpose(o: *Ops, in: u64, out: u64, rows: usize, cols: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(cols)));
        try cuda.launch.launch(o.k.transpose, .{ .grid = .{ .x = @intCast((cols + 31) / 32), .y = @intCast((rows + 31) / 32) }, .block = .{ .x = 32, .y = 8 } }, o.s, &a);
    }

    /// fp32 -> bf16, n values.
    pub fn toBf16(o: *Ops, in: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(o.k.to_bf16, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
    }
};
