//! ngram-learn: replacement n-gram table rows (global row id -> bf16 bits), applied in NgramTable.gather.
const std = @import("std");
const core = @import("core");
const st = core.safetensors;

/// The widest row an overlay holds (Flash Next's rows are 160 wide).
pub const max_width = 256;

pub const Overlay = struct {
    gpa: std.mem.Allocator,
    width: usize,
    map: std.AutoHashMapUnmanaged(i64, u32) = .empty,
    rows: std.ArrayList(u16) = .empty,
    /// rows an id falls through to when this overlay has none (a case's overlay over a persistent one)
    under: ?*const Overlay = null,

    pub fn init(gpa: std.mem.Allocator, width: usize) !Overlay {
        if (width == 0 or width > max_width) return error.OverlayRowWidth;
        return .{ .gpa = gpa, .width = width };
    }

    pub fn deinit(o: *Overlay) void {
        o.map.deinit(o.gpa);
        o.rows.deinit(o.gpa);
    }

    pub fn count(o: *const Overlay) usize {
        return o.map.count();
    }

    pub fn clear(o: *Overlay) void {
        o.map.clearRetainingCapacity();
        o.rows.clearRetainingCapacity();
    }

    /// Row `id` set to `row` (bf16 bits), replacing an earlier one.
    pub fn put(o: *Overlay, id: i64, row: []const u16) !void {
        if (row.len != o.width) return error.OverlayRowWidth;
        if (o.map.get(id)) |i| {
            @memcpy(o.rows.items[i * o.width ..][0..o.width], row);
            return;
        }
        const at: u32 = @intCast(o.rows.items.len / o.width);
        try o.rows.appendSlice(o.gpa, row);
        errdefer o.rows.shrinkRetainingCapacity(@as(usize, at) * o.width);
        try o.map.put(o.gpa, id, at);
    }

    pub fn get(o: *const Overlay, id: i64) ?[]const u16 {
        const i = o.map.get(id) orelse return if (o.under) |u| u.get(id) else null;
        return o.rows.items[@as(usize, i) * o.width ..][0..o.width];
    }

    /// `ids` (I64 [N]) and `rows` (BF16 [N, width]) from a safetensors file, added; returns N.
    pub fn load(o: *Overlay, io: std.Io, path: []const u8) !usize {
        var f = try st.File.open(o.gpa, io, path);
        defer f.close(io);
        const ids = f.get("ids") orelse return error.OverlayNoIds;
        const rows = f.get("rows") orelse return error.OverlayNoRows;
        const n = ids.numel();
        if (ids.dtype != .i64 or rows.dtype != .bf16 or rows.numel() != n * o.width) return error.OverlayShape;
        var row: [max_width]u16 = undefined;
        for (0..n) |i| {
            const id = std.mem.readInt(i64, ids.bytes[i * 8 ..][0..8], .little);
            for (row[0..o.width], 0..) |*r, j| r.* = std.mem.readInt(u16, rows.bytes[(i * o.width + j) * 2 ..][0..2], .little);
            try o.put(id, row[0..o.width]);
        }
        return n;
    }
};

test "an overlay row replaces, and an absent id falls through" {
    var o = try Overlay.init(std.testing.allocator, 4);
    defer o.deinit();
    try o.put(7, &.{ 1, 2, 3, 4 });
    try o.put(9, &.{ 5, 6, 7, 8 });
    try o.put(7, &.{ 0, 0, 0, 1 });
    try std.testing.expectEqualSlices(u16, &.{ 0, 0, 0, 1 }, o.get(7).?);
    try std.testing.expectEqualSlices(u16, &.{ 5, 6, 7, 8 }, o.get(9).?);
    try std.testing.expect(o.get(8) == null);
    try std.testing.expectEqual(@as(usize, 2), o.count());
    try std.testing.expectError(error.OverlayRowWidth, o.put(1, &.{ 1, 2 }));
    var top = try Overlay.init(std.testing.allocator, 4);
    defer top.deinit();
    top.under = &o;
    try top.put(9, &.{ 4, 4, 4, 4 });
    try std.testing.expectEqualSlices(u16, &.{ 4, 4, 4, 4 }, top.get(9).?);
    try std.testing.expectEqualSlices(u16, &.{ 0, 0, 0, 1 }, top.get(7).?);
}
