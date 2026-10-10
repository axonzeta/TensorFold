"""Phase 2 engine edits: the n-gram overlay in NgramTable.gather, TF_NGRAM_OVERLAY at load (run in the fork root)."""

import pathlib

p = pathlib.Path("zig/src/families/flashnext/cuda_weights.zig")
s = p.read_text()


def once(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


once('const int4 = @import("cuda_int4.zig");\n',
     'const int4 = @import("cuda_int4.zig");\nconst ngram_overlay = @import("ngram_overlay.zig");\n')
once("    gpu: ?struct { rows: u64, lut: u64, base: u64, count: u64, head0: u32, heads: u32 } = null,\n",
     "    gpu: ?struct { rows: u64, lut: u64, base: u64, count: u64, head0: u32, heads: u32 } = null,\n"
     "    /// ngram-learn: rows that replace the table's in `gather` (host gather only), and the one loaded at start\n"
     "    overlay: ?*const ngram_overlay.Overlay = null,\n"
     "    owned_overlay: ?*ngram_overlay.Overlay = null,\n")
once("    pub fn deinit(t: *NgramTable) void {\n",
     "    pub fn deinit(t: *NgramTable) void {\n"
     "        if (t.owned_overlay) |o| {\n            o.deinit();\n            t.gpa.destroy(o);\n        }\n")
once("            const src = t.row(@intCast(id));\n            const dst = out[i * t.width ..][0..t.width];\n",
     "            const dst = out[i * t.width ..][0..t.width];\n"
     "            if (t.overlay) |o| if (o.get(id)) |r| {\n                @memcpy(dst, r);\n                continue;\n            };\n"
     "            const src = t.row(@intCast(id));\n")
once("""            std.log.err("n-gram tables hold {d} rows, expected {d}", .{ t.rows, g.rows });
            return error.NgramRows;
        }
        return t;
""", """            std.log.err("n-gram tables hold {d} rows, expected {d}", .{ t.rows, g.rows });
            return error.NgramRows;
        }
        if (std.c.getenv("TF_NGRAM_OVERLAY")) |z| {
            const path = std.mem.span(z);
            const o = try L.gpa.create(ngram_overlay.Overlay);
            errdefer L.gpa.destroy(o);
            o.* = try ngram_overlay.Overlay.init(L.gpa, t.width);
            errdefer o.deinit();
            const n = try o.load(L.io, path);
            if (L.o.world != 1) return error.OverlayNeedsOneRank;
            t.owned_overlay = o;
            t.overlay = o;
            std.log.info("n-gram overlay: {d} rows from {s}", .{ n, path });
        }
        return t;
""")
p.write_text(s)
r = pathlib.Path("zig/src/families/flashnext/cuda.zig")
rs = r.read_text()
anchor = 'pub const int4_check = @import("cuda_int4_check.zig");\n'
assert rs.count(anchor) == 1
r.write_text(rs.replace(anchor, anchor + 'pub const ngram_overlay = @import("ngram_overlay.zig");\n'))
print("patched cuda_weights.zig, cuda.zig")
