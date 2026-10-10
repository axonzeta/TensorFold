//! ngram-learn training kernels (zig/kernels/cuda/fn_train.cu): the backward pass's helpers, loaded once.
const std = @import("std");
const cuda = @import("cuda");
const int4 = @import("cuda_int4.zig");

pub const Kernels = struct {
    module: cuda.Module,
    int4_dequant: cuda.Function,
    transpose: cuda.Function,
    to_bf16: cuda.Function,
    to_f32: cuda.Function,
    rms_fwd: cuda.Function,
    rms_bwd: cuda.Function,
    ew: cuda.Function,
    ew_bwd: cuda.Function,
    ew2: cuda.Function,
    mix_fwd: cuda.Function,
    mix_bwd: cuda.Function,
    inject_fwd: cuda.Function,
    inject_bwd: cuda.Function,
    copy2d: cuda.Function,
    rope: cuda.Function,
    attn_probs: cuda.Function,
    attn_out: cuda.Function,
    attn_ds: cuda.Function,
    attn_dq: cuda.Function,
    attn_dkv: cuda.Function,
    conv_fwd: cuda.Function,
    conv_bwd: cuda.Function,
    gdn_gates: cuda.Function,
    gdn_gates_bwd: cuda.Function,
    heads: cuda.Function,
    delta_fwd: cuda.Function,
    delta_bwd: cuda.Function,
    sdot: cuda.Function,
    sdot_bwd: cuda.Function,
    topk: cuda.Function,
    topk_bwd: cuda.Function,
    gather: cuda.Function,
    scatter_add: cuda.Function,
    xent: cuda.Function,
    pick_softmax: cuda.Function,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        var m = try cuda.Module.load(ctx.d, cuda.kernels.fn_train);
        errdefer m.unload();
        var k: Kernels = undefined;
        k.module = m;
        inline for (.{ .{ "int4_dequant", "fn_train_int4_dequant" }, .{ "transpose", "fn_train_transpose_bf16" }, .{ "to_bf16", "fn_train_f32_to_bf16" }, .{ "to_f32", "fn_train_bf16_to_f32" }, .{ "rms_fwd", "fn_train_rms_fwd" }, .{ "rms_bwd", "fn_train_rms_bwd" }, .{ "ew", "fn_train_ew" }, .{ "ew_bwd", "fn_train_ew_bwd" }, .{ "ew2", "fn_train_ew2" }, .{ "mix_fwd", "fn_train_mix_fwd" }, .{ "mix_bwd", "fn_train_mix_bwd" }, .{ "inject_fwd", "fn_train_inject_fwd" }, .{ "inject_bwd", "fn_train_inject_bwd" }, .{ "copy2d", "fn_train_copy2d" }, .{ "rope", "fn_train_rope" }, .{ "attn_probs", "fn_train_attn_probs" }, .{ "attn_out", "fn_train_attn_out" }, .{ "attn_ds", "fn_train_attn_ds" }, .{ "attn_dq", "fn_train_attn_dq" }, .{ "attn_dkv", "fn_train_attn_dkv" }, .{ "conv_fwd", "fn_train_conv_fwd" }, .{ "conv_bwd", "fn_train_conv_bwd" }, .{ "gdn_gates", "fn_train_gdn_gates" }, .{ "gdn_gates_bwd", "fn_train_gdn_gates_bwd" }, .{ "heads", "fn_train_heads" }, .{ "delta_fwd", "fn_train_delta_fwd" }, .{ "delta_bwd", "fn_train_delta_bwd" }, .{ "sdot", "fn_train_sdot" }, .{ "sdot_bwd", "fn_train_sdot_bwd" }, .{ "topk", "fn_train_topk_softmax" }, .{ "topk_bwd", "fn_train_topk_softmax_bwd" }, .{ "gather", "fn_train_gather" }, .{ "scatter_add", "fn_train_scatter_add" }, .{ "xent", "fn_train_xent" }, .{ "pick_softmax", "fn_train_pick_softmax" } }) |e| {
            @field(k, e[0]) = try m.function(e[1]);
        }
        return k;
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
