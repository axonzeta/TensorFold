//! ngram-learn training: the engine's weights as the training pass reads them, fp32 activations either side.
//! Forward products run on the serving block-FP8 matmul (fp32 out) or cuBLASLt (bf16 rows, dequantised int4);
//! backward products dx = dy·W run on cuBLASLt against Wᵀ, built once a lesson and cached by the weight's address:
//! identity rows through an FP8 face, a transpose of bf16 rows.
const std = @import("std");
const cuda = @import("cuda");
const fp8 = @import("cuda_fp8.zig");
const W = @import("cuda_weights.zig");
const fwd = @import("cuda_forward.zig");
const tops = @import("train_ops.zig");
const td = @import("train_dequant.zig");

/// A bf16 [n][k] matrix the training pass made itself (a dequantised expert, the head).
pub const Dense = struct { w: u64, n: u32, k: u32 };

/// One linear of the model: y = x · Wᵀ, W [n][k].
pub const Lin = union(enum) {
    fp8: fp8.Linear,
    rows: W.Rows,
    dense: Dense,

    pub fn n(l: Lin) usize {
        return switch (l) {
            .fp8 => |x| x.n,
            .rows => |x| x.n,
            .dense => |x| x.n,
        };
    }

    pub fn k(l: Lin) usize {
        return switch (l) {
            .fp8 => |x| x.k,
            .rows => |x| x.k,
            .dense => |x| x.k,
        };
    }

    fn key(l: Lin) u64 {
        return switch (l) {
            .fp8 => |x| x.w8,
            .rows => |x| x.weight,
            .dense => |x| x.w,
        };
    }
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    f: *fwd.Forward,
    ops: *tops.Ops,
    id: td.Ident,
    wt: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty,
    xb: cuda.DeviceBuffer,
    max: usize,

    /// Scratch for bf16 copies of up to `max` values (rows × widest input or output).
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, f: *fwd.Forward, ops: *tops.Ops, max: usize) !Weights {
        var id = try td.Ident.init(gpa, d, 16384);
        errdefer id.deinit(gpa);
        return .{ .gpa = gpa, .d = d, .f = f, .ops = ops, .id = id, .xb = try cuda.DeviceBuffer.alloc(d, max * 2), .max = max };
    }

    pub fn deinit(t: *Weights) void {
        t.drop();
        t.wt.deinit(t.gpa);
        t.xb.free();
        t.id.deinit(t.gpa);
    }

    /// Every cached Wᵀ freed (the end of a lesson).
    pub fn drop(t: *Weights) void {
        var it = t.wt.valueIterator();
        while (it.next()) |b| b.free();
        t.wt.clearRetainingCapacity();
    }

    /// Bytes the cached Wᵀ hold.
    pub fn cached(t: *const Weights) usize {
        var n: usize = 0;
        var it = t.wt.valueIterator();
        while (it.next()) |b| n += b.len;
        return n;
    }

    /// y[m][n] = x[m][k] · Wᵀ (fp32 x, fp32 y).
    pub fn forward(t: *Weights, l: Lin, x: u64, y: u64, m: usize) !void {
        const kk = l.k();
        if (m * kk > t.max) return error.TrainScratch;
        try t.ops.toBf16(x, t.xb.ptr, m * kk);
        switch (l) {
            .fp8 => |x8| try fp8.matmul(t.f.k8 orelse return error.NoFp8Kernels, t.f.s, t.xb.ptr, kk, x8, y, true, m),
            .rows => |r| try t.ops.gemm(t.xb.ptr, r.weight, y, m, r.n, kk, .f32, false),
            .dense => |dn| try t.ops.gemm(t.xb.ptr, dn.w, y, m, dn.n, kk, .f32, false),
        }
    }

    /// dx[m][k] (+)= dy[m][n] · W (fp32 both).
    pub fn backward(t: *Weights, l: Lin, dy: u64, dx: u64, m: usize, add: bool) !void {
        const n = l.n();
        if (m * n > t.max) return error.TrainScratch;
        try t.ops.toBf16(dy, t.xb.ptr, m * n);
        switch (l) {
            .fp8 => try t.ops.gemm(t.xb.ptr, try t.transposed(l), dx, m, l.k(), n, .f32, add),
            .rows => |r| try t.ops.gemmNN(t.xb.ptr, r.weight, dx, m, n, l.k(), .f32, add),
            .dense => |dn| try t.ops.gemmNN(t.xb.ptr, dn.w, dx, m, n, l.k(), .f32, add),
        }
    }

    /// Wᵀ [k][n] bf16, built on first use.
    pub fn transposed(t: *Weights, l: Lin) !u64 {
        const gop = try t.wt.getOrPut(t.gpa, l.key());
        if (gop.found_existing) return gop.value_ptr.ptr;
        errdefer _ = t.wt.remove(l.key());
        var buf = try cuda.DeviceBuffer.alloc(t.d, l.n() * l.k() * 2);
        errdefer buf.free();
        switch (l) {
            .fp8 => |x8| try td.fp8Transposed(t.f, &t.id, x8, buf.ptr),
            .rows => |r| try t.ops.transpose(r.weight, buf.ptr, r.n, r.k),
            .dense => |dn| try t.ops.transpose(dn.w, buf.ptr, dn.n, dn.k),
        }
        gop.value_ptr.* = buf;
        return buf.ptr;
    }
};
