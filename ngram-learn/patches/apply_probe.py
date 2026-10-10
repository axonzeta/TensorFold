"""Phase 1d/1e engine edits: the ple_mute hook in pleBlock and the `score` command dispatch (run in the fork root)."""

import pathlib

p = pathlib.Path("zig/src/families/flashnext/cuda_forward.zig")
s = p.read_text()
a = "    dump: ?*Dump = null,\n"
assert s.count(a) == 1
s = s.replace(a, a + "    /// ngram-learn probe: positions whose n-gram rows contribute nothing (gated rows zeroed); null in serving\n"
                     "    ple_mute: ?[]const bool = null,\n")
b = ("        try f.t.pleGate(b.ple_keys, b.ple_vals, b.h, p.norm_key, p.norm_query, b.ple_gated, b.ple_pss, R, g.hidden, "
     "g.streams, f.eps);\n        const Wd = g.wide();\n")
assert s.count(b) == 1
s = s.replace(b, b + "        if (f.ple_mute) |mute| for (segs) |sg| for (sg.a0..sg.a1) |r| {\n"
                     "            const at: usize = @as(usize, @intCast(sg.seq.st.pos)) + (r - sg.a0);\n"
                     "            if (at < mute.len and mute[at]) try f.th.zero(b.ple_gated + r * Wd * 2, Wd * 2);\n"
                     "        };\n")
p.write_text(s)

q = pathlib.Path("zig/src/cli/flashnext_cuda.zig")
t = q.read_text()
c = '    if (std.mem.eql(u8, cmd, "agree") and rest.len == 1) return agree(gpa, init.io, e, rest[0], o);\n'
assert t.count(c) == 1
t = t.replace(c, c + '    if (std.mem.eql(u8, cmd, "score") and rest.len == 1) return @import("flashnext_probe.zig").score(gpa, '
                     'init.io, e, rest[0], o.report);\n'
                     '    if (std.mem.eql(u8, cmd, "worker")) return @import("flashnext_probe.zig").worker(gpa, init.io, e);\n')
u = "    \\\\  tensorfold gate MODEL CAPTURE_DIR"
assert t.count(u) == 1
t = t.replace(u, "    \\\\  tensorfold score MODEL CASES.json --report OUT.json   (ngram-learn: teacher-forced log-probs of each\n"
                 "    \\\\           case's continuation after its prompt, n-gram rows muted at its listed positions)\n"
                 "    \\\\  tensorfold worker MODEL   (ngram-learn: the same cases one JSON line at a time on stdin, answers on stdout)\n" + u)
q.write_text(t)
print("patched")
