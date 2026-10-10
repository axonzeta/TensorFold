"""B2: train_ops export and the train-op-check command (no weights; run in the fork root)."""

import pathlib


def once(path, old, new):
    p = pathlib.Path(path)
    s = p.read_text()
    assert s.count(old) == 1, (path, old)
    p.write_text(s.replace(old, new))


once("zig/src/families/flashnext/cuda.zig", 'pub const train_kernels = @import("train_kernels.zig");\n',
     'pub const train_kernels = @import("train_kernels.zig");\npub const train_ops = @import("train_ops.zig");\n')
once("zig/src/cli/flashnext_cuda.zig", '    if (std.mem.eql(u8, cmd, "glue-check")) return glueCheck(gpa, init.io, &ctx, kernels);\n',
     '    if (std.mem.eql(u8, cmd, "glue-check")) return glueCheck(gpa, init.io, &ctx, kernels);\n'
     '    if (std.mem.eql(u8, cmd, "train-op-check") and rest.len == 1) return @import("train_check.zig").opCheck(gpa, init.io, &ctx, rest[0]);\n')
print("patched train-op-check")
