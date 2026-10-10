"""Phase 2b: the `worker` command dispatch (on top of apply_probe.py's edits; run in the fork root)."""

import pathlib

q = pathlib.Path("zig/src/cli/flashnext_cuda.zig")
t = q.read_text()
c = ('    if (std.mem.eql(u8, cmd, "score") and rest.len == 1) return @import("flashnext_probe.zig").score(gpa, '
     'init.io, e, rest[0], o.report);\n')
assert t.count(c) == 1
t = t.replace(c, c + '    if (std.mem.eql(u8, cmd, "worker")) return @import("flashnext_probe.zig").worker(gpa, init.io, e);\n')
u = "    \\\\  tensorfold gate MODEL CAPTURE_DIR"
t = t.replace(u, "    \\\\  tensorfold worker MODEL   (ngram-learn: score cases one JSON line at a time on stdin, answers on stdout)\n" + u)
q.write_text(t)
print("patched worker")
