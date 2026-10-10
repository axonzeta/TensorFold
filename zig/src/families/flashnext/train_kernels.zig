//! ngram-learn training kernels (zig/kernels/cuda/fn_train.cu): the backward pass's helpers, loaded once.
const std = @import("std");
const cuda = @import("cuda");
const int4 = @import("cuda_int4.zig");

pub const Kernels = struct {
    module: cuda.Module,
    int4_dequant: cuda.Function,
    transpose: cuda.Function,
    to_bf16: cuda.Function,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        var m = try cuda.Module.load(ctx.d, cuda.kernels.fn_train);
        errdefer m.unload();
        return .{ .module = m, .int4_dequant = try m.function("fn_train_int4_dequant"), .transpose = try m.function("fn_train_transpose_bf16"), .to_bf16 = try m.function("fn_train_f32_to_bf16") };
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }

    /// A packed int4 [n, k] matrix (words, fp16 scales, groups of gs) as bf16 W [n][k] at `out`.
    pub fn dequantInt4(k: *const Kernels, s: cuda.Stream, words: u64, scales: u64, out: u64, n: usize, kk: usize, gs: usize) !void {
        if (n % 8 != 0 or kk % gs != 0 or (gs != 64 and gs != 128)) return error.BadShape;
        const total = n * kk / 8;
        var a: cuda.Args = .{};
        a.add(words);
        a.add(scales);
        a.add(out);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(kk)));
        a.add(@as(i32, @intCast(gs)));
        try cuda.launch.launch(k.int4_dequant, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Expert `e`'s gate (m = 0), up (1) or down (2) of `x` as bf16 [n][k] at `out`.
    pub fn expert(k: *const Kernels, s: cuda.Stream, x: int4.Experts, e: usize, m: usize, out: u64) !void {
        const ni: usize = x.width;
        const d: usize = x.dims;
        if (m < 2) {
            const uw = int4.wordsOf(ni, d);
            const us = int4.scalesOf(ni, d, 128);
            return k.dequantInt4(s, x.up + (e * 2 + m) * uw * 4, x.up_s + (e * 2 + m) * us * 2, out, ni, d, 128);
        }
        const dw = int4.wordsOf(d, ni);
        const ds = int4.scalesOf(d, ni, x.gs_down);
        return k.dequantInt4(s, x.down + e * dw * 4, x.down_s + e * ds * 2, out, d, ni, x.gs_down);
    }
};
