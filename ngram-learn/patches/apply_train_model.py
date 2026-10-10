"""B3–B10: the training runtime modules' exports and the train-grad-check command (run in the fork root)."""

import pathlib


def once(path, old, new):
    p = pathlib.Path(path)
    s = p.read_text()
    assert s.count(old) == 1, (path, old)
    p.write_text(s.replace(old, new))


once("zig/src/families/flashnext/cuda.zig", 'pub const train_ops = @import("train_ops.zig");\n',
     'pub const train_ops = @import("train_ops.zig");\npub const train_model = @import("train_model.zig");\n')
once("zig/src/cli/flashnext_cuda.zig",
     '    if (std.mem.eql(u8, cmd, "worker")) return @import("flashnext_probe.zig").worker(gpa, init.io, e);\n',
     '    if (std.mem.eql(u8, cmd, "worker")) return @import("flashnext_probe.zig").worker(gpa, init.io, e);\n'
     '    if (std.mem.eql(u8, cmd, "train-grad-check") and rest.len == 1) return @import("train_grad.zig").gradCheck(gpa, init.io, e, rest[0]);\n')
print("patched train runtime")
