//! ngram-learn training: one device buffer handed out by bumping (256-byte aligned), reset between layers.
const std = @import("std");
const cuda = @import("cuda");

pub const Arena = struct {
    buf: cuda.DeviceBuffer,
    at: usize = 0,
    peak: usize = 0,

    pub fn init(d: *const cuda.Driver, size: usize) !Arena {
        return .{ .buf = try cuda.DeviceBuffer.alloc(d, size) };
    }

    pub fn deinit(a: *Arena) void {
        a.buf.free();
    }

    /// `n` fp32 values (or any `n * 4` bytes).
    pub fn f32s(a: *Arena, n: usize) !u64 {
        return a.bytes(n * 4);
    }

    pub fn bytes(a: *Arena, n: usize) !u64 {
        const at = std.mem.alignForward(usize, a.at, 256);
        if (at + n > a.buf.len) return error.TrainArenaFull;
        a.at = at + n;
        a.peak = @max(a.peak, a.at);
        return a.buf.ptr + at;
    }

    /// `n` fp32 zeros.
    pub fn zeros(a: *Arena, n: usize, s: cuda.Stream) !u64 {
        const p = try a.f32s(n);
        try (cuda.DeviceBuffer{ .d = a.buf.d, .ptr = p, .len = n * 4 }).fill8(0, s.handle);
        return p;
    }

    pub fn mark(a: *const Arena) usize {
        return a.at;
    }

    pub fn reset(a: *Arena, to: usize) void {
        a.at = to;
    }
};
