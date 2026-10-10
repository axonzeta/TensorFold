"""B1: the fn_train kernel module (build list, embedded fatbin, family export; run in the fork root)."""

import pathlib


def once(path, old, new):
    p = pathlib.Path(path)
    s = p.read_text()
    assert s.count(old) == 1, (path, old)
    p.write_text(s.replace(old, new))


once("zig/build/cuda.zig", '    .{ .name = "fn_qsa_scores", .flags = &.{"-O3"} }, // ours: attention._scores\' bits from row tiles (the prompt indexer)\n',
     '    .{ .name = "fn_qsa_scores", .flags = &.{"-O3"} }, // ours: attention._scores\' bits from row tiles (the prompt indexer)\n'
     '    .{ .name = "fn_train", .flags = &.{"-O3"} }, // ngram-learn: the backward pass\'s kernels (flashnext/train_*.zig)\n')
once("zig/src/cuda/kernels.zig", 'pub const fn_pack: []const u8 = if (available) &Blob("fatbin_fn_pack").bytes else &.{};\n',
     'pub const fn_pack: []const u8 = if (available) &Blob("fatbin_fn_pack").bytes else &.{};\n'
     'pub const fn_train: []const u8 = if (available) &Blob("fatbin_fn_train").bytes else &.{};\n')
once("zig/src/families/flashnext/cuda.zig", 'pub const train_dequant = @import("train_dequant.zig");\n',
     'pub const train_dequant = @import("train_dequant.zig");\npub const train_kernels = @import("train_kernels.zig");\n')
print("patched fn_train")
