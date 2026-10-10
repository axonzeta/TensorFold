"""B1: train_dequant export and the dequant-check command (run in the fork root)."""

import pathlib

r = pathlib.Path("zig/src/families/flashnext/cuda.zig")
s = r.read_text()
a = 'pub const ngram_overlay = @import("ngram_overlay.zig");\n'
assert s.count(a) == 1
r.write_text(s.replace(a, a + 'pub const train_dequant = @import("train_dequant.zig");\n'))
q = pathlib.Path("zig/src/cli/flashnext_cuda.zig")
t = q.read_text()
c = '    if (std.mem.eql(u8, cmd, "worker")) return @import("flashnext_probe.zig").worker(gpa, init.io, e);\n'
assert t.count(c) == 1
q.write_text(t.replace(c, c + '    if (std.mem.eql(u8, cmd, "dequant-check") and rest.len == 1) return @import("flashnext_probe.zig").dequantCheck(gpa, init.io, e, rest[0]);\n'))
print("patched dequant")
