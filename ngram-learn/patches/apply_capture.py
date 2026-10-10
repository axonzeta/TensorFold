"""Training from a served forward: Forward.capture hooks (null in serving; run in the fork root)."""

import pathlib

p = pathlib.Path("zig/src/families/flashnext/cuda_forward.zig")
s = p.read_text()


def once(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


once("    ple_mute: ?[]const bool = null,\n",
     "    ple_mute: ?[]const bool = null,\n"
     "    /// ngram-learn: a prompt forward's layer inputs, picks and final streams for training; null in serving\n"
     "    capture: ?*const @import(\"train_capture.zig\").Capture = null,\n")
once("""            if (sp) |q| try f.gatherRows(b.h, q, g.wide() * 2, .bf16, 2);
            try f.mark(.hc_writeback);
            try f.pleBlock(p, segs, x, R);""", """            if (sp) |q| try f.gatherRows(b.h, q, g.wide() * 2, .bf16, 2);
            try f.mark(.hc_writeback);
            try f.captureIn(layer, b.h, R);
            try f.pleBlock(p, segs, x, R);""")
once("""        try f.hcBlock(&layer.attn_hc, x, R, b.h, pend, b.inj_a, sp);
        try f.putLayer(layer, "hc_a", b.mixed, R * g.hidden * 2);""", """        try f.hcBlock(&layer.attn_hc, x, R, b.h, pend, b.inj_a, sp);
        if (layer.ple == null) try f.captureIn(layer, b.h, R);
        try f.putLayer(layer, "hc_a", b.mixed, R * g.hidden * 2);""")
once("""        const moe = try f.moeBlock(&layer.moe, x, R, mtp);""", """        const moe = try f.moeBlock(&layer.moe, x, R, mtp);
        if (f.capture) |c| if (layer.index >= 0 and !mtp) try f.th.copy(c.picks + @as(usize, @intCast(layer.index)) * c.rows * c.slots * 4, b.moe_pick, R * c.slots * 4);""")
once("""            try f.th.copy(b.streams, b.h, R * Wd * 2);
            try f.writeback(b.streams, x, R, pending);
        }""", """            try f.th.copy(b.streams, b.h, R * Wd * 2);
            try f.writeback(b.streams, x, R, pending);
        }
        if (f.capture) |c| try f.th.copy(c.final, b.streams, R * Wd * 2);""")
once("""    fn putLayer(f: *Forward, layer: *const W.Layer, what: []const u8, ptr: u64, bytes: usize) !void {""",
     """    /// ngram-learn: layer `layer`'s input (its rows in h) into the capture.
    fn captureIn(f: *Forward, layer: *const W.Layer, h: u64, R: usize) !void {
        const c = f.capture orelse return;
        if (layer.index < 0 or R != c.rows) return;
        const wd = f.g.wide() * 2;
        try f.th.copy(c.h_in + @as(usize, @intCast(layer.index)) * c.rows * wd, h, R * wd);
    }

    fn putLayer(f: *Forward, layer: *const W.Layer, what: []const u8, ptr: u64, bytes: usize) !void {""")
p.write_text(s)
r = pathlib.Path("zig/src/families/flashnext/cuda.zig")
rs = r.read_text()
a = 'pub const train_model = @import("train_model.zig");\n'
assert rs.count(a) == 1
r.write_text(rs.replace(a, a + 'pub const train_capture = @import("train_capture.zig");\n'))
print("patched capture")
