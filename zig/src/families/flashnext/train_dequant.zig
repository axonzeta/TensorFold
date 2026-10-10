//! ngram-learn training: a block-FP8 face's weight as bf16 Wᵀ [k][n], read through the face's own serving matmul.
//! Identity rows in give the weight's columns out (row j = e_j -> W[:, j]), each value the kernel's own bf16 rounding of
//! code × scale, so the backward's dx = dy·W runs on exactly the weights the forward used, with no layout decoding.
const std = @import("std");
const cuda = @import("cuda");
const fp8 = @import("cuda_fp8.zig");
const fwd = @import("cuda_forward.zig");

/// Identity rows a matmul takes at once.
pub const chunk: usize = 256;

/// Scratch for `fp8Transposed`: `chunk` identity rows of width up to `max_k`, and their host image.
pub const Ident = struct {
    d: *const cuda.Driver,
    buf: cuda.DeviceBuffer,
    host: []u16,
    max_k: usize,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, max_k: usize) !Ident {
        const host = try gpa.alloc(u16, chunk * max_k);
        errdefer gpa.free(host);
        return .{ .d = d, .buf = try cuda.DeviceBuffer.alloc(d, chunk * max_k * 2), .host = host, .max_k = max_k };
    }

    pub fn deinit(i: *Ident, gpa: std.mem.Allocator) void {
        i.buf.free();
        gpa.free(i.host);
    }

    /// Rows r < m of width k: ones at column j0 + r (bf16 1.0 = 0x3F80), zeros elsewhere, on the device.
    fn set(i: *Ident, k: usize, j0: usize, m: usize) !void {
        if (k > i.max_k) return error.IdentityTooNarrow;
        const h = i.host[0 .. chunk * k];
        @memset(h, 0);
        for (0..m) |r| h[r * k + j0 + r] = 0x3F80;
        try i.buf.upload(0, std.mem.sliceAsBytes(h));
    }
};

/// Wᵀ of face `l` as bf16 [l.k][l.n] at `out` (l.k * l.n * 2 bytes), through fp8.matmul on identity rows.
pub fn fp8Transposed(f: *fwd.Forward, id: *Ident, l: fp8.Linear, out: u64) !void {
    const k: usize = l.k;
    const n: usize = l.n;
    var j: usize = 0;
    while (j < k) : (j += chunk) {
        const m = @min(chunk, k - j);
        try f.s.synchronize();
        try id.set(k, j, m);
        try fp8.matmul(f.k8 orelse return error.NoFp8Kernels, f.s, id.buf.ptr, k, l, out + j * n * 2, false, m);
    }
    try f.s.synchronize();
}
