//! ngram-learn training: GEMMs through cuBLASLt (bf16 in, fp32 accumulate, fp32 or bf16 out, optionally added to the
//! output) and fn_train's small kernels. The backward's dx = dy·W is `gemm(dy, Wᵀ)` with Wᵀ [k][n] as the weight.
const std = @import("std");
const cuda = @import("cuda");
const lt = cuda.cublaslt;
const tk = @import("train_kernels.zig");

pub const Ops = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const tk.Kernels,
    lib: lt.Library,
    plans: std.AutoHashMapUnmanaged(Key, lt.Linear) = .empty,
    nn: std.AutoHashMapUnmanaged(Key, Plan) = .empty,
    work: cuda.DeviceBuffer,

    const Key = struct { m: u64, n: u64, k: u64, out: lt.DataType };
    const workspace: usize = 32 << 20;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, k: *const tk.Kernels) !Ops {
        var lib = try lt.Library.open();
        errdefer lib.close();
        return .{ .gpa = gpa, .d = d, .s = s, .k = k, .lib = lib, .work = try cuda.DeviceBuffer.alloc(d, workspace) };
    }

    pub fn deinit(o: *Ops) void {
        var it = o.plans.valueIterator();
        while (it.next()) |p| p.deinit();
        o.plans.deinit(o.gpa);
        var jt = o.nn.valueIterator();
        while (jt.next()) |p| p.deinit(&o.lib);
        o.nn.deinit(o.gpa);
        o.work.free();
        o.lib.close();
    }

    /// D[m][n] (= or +=) X[m][k] · W[n][k]ᵀ; X and W bf16, D fp32 (`out` .f32) or bf16.
    pub fn gemm(o: *Ops, x: u64, w: u64, d: u64, m: usize, n: usize, k: usize, out: lt.DataType, add: bool) !void {
        const key: Key = .{ .m = m, .n = n, .k = k, .out = out };
        const gop = try o.plans.getOrPut(o.gpa, key);
        if (!gop.found_existing) {
            gop.value_ptr.* = lt.Linear.init(&o.lib, m, n, k, out, workspace) catch |e| {
                _ = o.plans.remove(key);
                return e;
            };
        }
        const p = gop.value_ptr;
        const one: f32 = 1;
        const beta: f32 = if (add) 1 else 0;
        try o.lib.check(o.lib.api.cublasLtMatmul(p.handle, p.desc, &one, w, p.a, x, p.b, &beta, d, p.c, d, p.c, &p.algo, o.work.ptr, workspace, o.s.handle), "cublasLtMatmul");
    }

    /// One cuBLASLt product D[m][k] = A[m][n] · B[n][k] (row-major, bf16 in, fp32 accumulate): the backward's
    /// dx = dy·W straight from W, no transposed copy.
    const Plan = struct {
        handle: lt.Handle,
        desc: lt.MatmulDesc,
        a: lt.Layout,
        b: lt.Layout,
        c: lt.Layout,
        algo: lt.Algo,

        fn init(lib: *const lt.Library, m: u64, n: u64, k: u64, out: lt.DataType) !Plan {
            var p: Plan = undefined;
            try lib.check(lib.api.cublasLtCreate(&p.handle), "cublasLtCreate");
            errdefer _ = lib.api.cublasLtDestroy(p.handle);
            try lib.check(lib.api.cublasLtMatmulDescCreate(&p.desc, lt.compute_32f, .f32), "desc");
            errdefer _ = lib.api.cublasLtMatmulDescDestroy(p.desc);
            const nop: c_int = @intFromEnum(lt.Op.n);
            try lib.check(lib.api.cublasLtMatmulDescSetAttribute(p.desc, lt.desc_transa, &nop, @sizeOf(c_int)), "transa");
            try lib.check(lib.api.cublasLtMatmulDescSetAttribute(p.desc, lt.desc_transb, &nop, @sizeOf(c_int)), "transb");
            // column-major: Dᵀ (k x m) = Bᵀ (k x n, B stored row-major [n][k]) · Aᵀ (n x m, A row-major [m][n])
            try lib.check(lib.api.cublasLtMatrixLayoutCreate(&p.a, .bf16, k, n, @intCast(k)), "layout B");
            try lib.check(lib.api.cublasLtMatrixLayoutCreate(&p.b, .bf16, n, m, @intCast(n)), "layout A");
            try lib.check(lib.api.cublasLtMatrixLayoutCreate(&p.c, out, k, m, @intCast(k)), "layout D");
            var pref: lt.Preference = undefined;
            try lib.check(lib.api.cublasLtMatmulPreferenceCreate(&pref), "pref");
            defer _ = lib.api.cublasLtMatmulPreferenceDestroy(pref);
            const limit: u64 = workspace;
            try lib.check(lib.api.cublasLtMatmulPreferenceSetAttribute(pref, lt.pref_max_workspace_bytes, &limit, @sizeOf(u64)), "workspace");
            var found: [1]lt.Heuristic = undefined;
            var count: c_int = 0;
            try lib.check(lib.api.cublasLtMatmulAlgoGetHeuristic(p.handle, p.desc, p.a, p.b, p.c, p.c, pref, 1, &found, &count), "heuristic");
            if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
            p.algo = found[0].algo;
            return p;
        }

        fn deinit(p: *Plan, lib: *const lt.Library) void {
            _ = lib.api.cublasLtMatrixLayoutDestroy(p.c);
            _ = lib.api.cublasLtMatrixLayoutDestroy(p.b);
            _ = lib.api.cublasLtMatrixLayoutDestroy(p.a);
            _ = lib.api.cublasLtMatmulDescDestroy(p.desc);
            _ = lib.api.cublasLtDestroy(p.handle);
        }
    };

    /// D[m][k] (= or +=) A[m][n] · B[n][k]; A, B bf16 row-major, D fp32 or bf16.
    pub fn gemmNN(o: *Ops, a_in: u64, b_in: u64, d: u64, m: usize, n: usize, k: usize, out: lt.DataType, add: bool) !void {
        const key: Key = .{ .m = m, .n = n, .k = k, .out = out };
        const gop = try o.nn.getOrPut(o.gpa, key);
        if (!gop.found_existing) {
            gop.value_ptr.* = Plan.init(&o.lib, m, n, k, out) catch |e| {
                _ = o.nn.remove(key);
                return e;
            };
        }
        const p = gop.value_ptr;
        const one: f32 = 1;
        const beta: f32 = if (add) 1 else 0;
        try o.lib.check(o.lib.api.cublasLtMatmul(p.handle, p.desc, &one, b_in, p.a, a_in, p.b, &beta, d, p.c, d, p.c, &p.algo, o.work.ptr, workspace, o.s.handle), "cublasLtMatmul NN");
    }

    /// bf16 [rows][cols] at `in` -> [cols][rows] at `out`.
    pub fn transpose(o: *Ops, in: u64, out: u64, rows: usize, cols: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(cols)));
        try cuda.launch.launch(o.k.transpose, .{ .grid = .{ .x = @intCast((cols + 31) / 32), .y = @intCast((rows + 31) / 32) }, .block = .{ .x = 32, .y = 8 } }, o.s, &a);
    }

    /// fp32 -> bf16, n values.
    pub fn toBf16(o: *Ops, in: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(o.k.to_bf16, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    fn flat(n: usize) cuda.Config {
        return .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } };
    }

    /// bf16 -> fp32, n values.
    pub fn toF32(o: *Ops, in: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(o.k.to_f32, flat(n), o.s, &a);
    }

    /// Norm gain: none, (1 + w) or w.
    pub const Gain = enum(i32) { none = 0, one_plus = 1, plain = 2, f32 = 3 };

    /// RMSNorm of R rows of G groups of D (fp32): y and each group's rsqrt in `rinv` [R][G].
    pub fn rmsFwd(o: *Ops, x: u64, w: u64, mode: Gain, y: u64, rinv: u64, r: usize, g: usize, d: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(@intFromEnum(mode));
        a.add(y);
        a.add(rinv);
        a.add(@as(i32, @intCast(g)));
        a.add(@as(i32, @intCast(d)));
        a.add(eps);
        try cuda.launch.launch(o.k.rms_fwd, .{ .grid = .{ .x = @intCast(g), .y = @intCast(r) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// Its backward: dx (+)= the gradient at x of `rmsFwd`'s y, given dy.
    pub fn rmsBwd(o: *Ops, dy: u64, x: u64, w: u64, mode: Gain, rinv: u64, dx: u64, r: usize, g: usize, d: usize, add: bool) !void {
        var a: cuda.Args = .{};
        a.add(dy);
        a.add(x);
        a.add(w);
        a.add(@intFromEnum(mode));
        a.add(rinv);
        a.add(dx);
        a.add(@as(i32, @intCast(g)));
        a.add(@as(i32, @intCast(d)));
        a.add(@as(i32, @intFromBool(add)));
        try cuda.launch.launch(o.k.rms_bwd, .{ .grid = .{ .x = @intCast(g), .y = @intCast(r) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    pub const Fn = enum(i32) { identity = 0, silu = 1, sigmoid = 2, softplus = 3, exp = 4, sqrt_sigmoid = 5 };

    /// y = beta * f(alpha * x), n values.
    pub fn ew(o: *Ops, f: Fn, x: u64, y: u64, n: usize, alpha: f32, beta: f32) !void {
        var a: cuda.Args = .{};
        a.add(@intFromEnum(f));
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        a.add(alpha);
        a.add(beta);
        try cuda.launch.launch(o.k.ew, flat(n), o.s, &a);
    }

    /// dx (+)= dy * beta * alpha * f'(alpha * x).
    pub fn ewBwd(o: *Ops, f: Fn, dy: u64, x: u64, dx: u64, n: usize, alpha: f32, beta: f32, add: bool) !void {
        var a: cuda.Args = .{};
        a.add(@intFromEnum(f));
        a.add(dy);
        a.add(x);
        a.add(dx);
        a.add(@as(i64, @intCast(n)));
        a.add(alpha);
        a.add(beta);
        a.add(@as(i32, @intFromBool(add)));
        try cuda.launch.launch(o.k.ew_bwd, flat(n), o.s, &a);
    }

    pub const Bin = enum(i32) { mul = 0, add = 1, add_mul = 2 };

    /// y = a * b, a + b, or y += a * b.
    pub fn ew2(o: *Ops, op: Bin, x: u64, y: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(@intFromEnum(op));
        a.add(x);
        a.add(y);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(o.k.ew2, flat(n), o.s, &a);
    }

    /// mixed = mean over S streams of u * hn ([R][S][D] -> [R][D]).
    pub fn mixFwd(o: *Ops, u: u64, hn: u64, mixed: u64, r: usize, s: usize, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(u);
        a.add(hn);
        a.add(mixed);
        inline for (.{ r, s, d }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.mix_fwd, flat(r * d), o.s, &a);
    }

    /// du = dmixed * hn / S; dhn (+)= dmixed * u / S.
    pub fn mixBwd(o: *Ops, dmixed: u64, u: u64, hn: u64, du: u64, dhn: u64, r: usize, s: usize, d: usize, add: bool) !void {
        var a: cuda.Args = .{};
        inline for (.{ dmixed, u, hn, du, dhn }) |v| a.add(v);
        inline for (.{ r, s, d }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(i32, @intFromBool(add)));
        try cuda.launch.launch(o.k.mix_bwd, flat(r * s * d), o.s, &a);
    }

    /// h = res + inj[r][s] * branch[r] per stream.
    pub fn injectFwd(o: *Ops, res: u64, inj: u64, branch: u64, h: u64, r: usize, s: usize, d: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ res, inj, branch, h }) |v| a.add(v);
        inline for (.{ r, s, d }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.inject_fwd, flat(r * s * d), o.s, &a);
    }

    /// dbranch = sum_s inj * dh; dinj = sum_d dh * branch (dres is dh itself).
    pub fn injectBwd(o: *Ops, dh: u64, inj: u64, branch: u64, dbranch: u64, dinj: u64, r: usize, s: usize, d: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ dh, inj, branch, dbranch, dinj }) |v| a.add(v);
        inline for (.{ s, d }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.inject_bwd, .{ .grid = .{ .x = @intCast(r) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// dst[r][dst_off..] (+)= src[r][src_off..] for `width` columns of `rows` rows (fp32, leading dims in elements).
    pub fn copy2d(o: *Ops, src: u64, src_ld: usize, src_off: usize, dst: u64, dst_ld: usize, dst_off: usize, rows: usize, width: usize, add: bool) !void {
        var a: cuda.Args = .{};
        a.add(src);
        inline for (.{ src_ld, src_off }) |v| a.add(@as(i32, @intCast(v)));
        a.add(dst);
        inline for (.{ dst_ld, dst_off, rows, width }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(i32, @intFromBool(add)));
        try cuda.launch.launch(o.k.copy2d, flat(rows * width), o.s, &a);
    }

    /// Rotate-half RoPE on the first `rot` of each head's `hd` dims (backward: the transposed rotation of dy).
    pub fn rope(o: *Ops, x: u64, cs: u64, sn: u64, y: u64, r: usize, h: usize, hd: usize, rot: usize, backward: bool) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, cs, sn, y }) |v| a.add(v);
        inline for (.{ r, h, hd, rot }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(i32, @intFromBool(backward)));
        try cuda.launch.launch(o.k.rope, flat(r * h * hd), o.s, &a);
    }

    /// Causal GQA attention forward: probabilities P [hq][r][r] and output o [r][hq][d].
    pub fn attnFwd(o: *Ops, q: u64, k: u64, v: u64, p: u64, out: u64, r: usize, hq: usize, hk: usize, d: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ q, k, p }) |x| a.add(x);
        inline for (.{ r, hq, hk, d }) |x| a.add(@as(i32, @intCast(x)));
        a.add(scale);
        const grid: cuda.Dim3 = .{ .x = @intCast(hq), .y = @intCast(r) };
        try cuda.launch.launch(o.k.attn_probs, .{ .grid = grid, .block = .{ .x = 256 }, .shared = @intCast(d * 4) }, o.s, &a);
        var b: cuda.Args = .{};
        inline for (.{ p, v, out }) |x| b.add(x);
        inline for (.{ r, hq, hk, d }) |x| b.add(@as(i32, @intCast(x)));
        try cuda.launch.launch(o.k.attn_out, .{ .grid = grid, .block = .{ .x = 256 } }, o.s, &b);
    }

    /// Its backward: dq, dk, dv from dO (ds is [hq][r][r] scratch).
    pub fn attnBwd(o: *Ops, q: u64, k: u64, v: u64, p: u64, d_out: u64, ds: u64, dq: u64, dk: u64, dv: u64, r: usize, hq: usize, hk: usize, d: usize, scale: f32) !void {
        const grid: cuda.Dim3 = .{ .x = @intCast(hq), .y = @intCast(r) };
        var a: cuda.Args = .{};
        inline for (.{ p, d_out, v, ds }) |x| a.add(x);
        inline for (.{ r, hq, hk, d }) |x| a.add(@as(i32, @intCast(x)));
        try cuda.launch.launch(o.k.attn_ds, .{ .grid = grid, .block = .{ .x = 256 }, .shared = @intCast(d * 4) }, o.s, &a);
        var b: cuda.Args = .{};
        inline for (.{ ds, k, dq }) |x| b.add(x);
        inline for (.{ r, hq, hk, d }) |x| b.add(@as(i32, @intCast(x)));
        b.add(scale);
        try cuda.launch.launch(o.k.attn_dq, .{ .grid = grid, .block = .{ .x = 256 } }, o.s, &b);
        var c: cuda.Args = .{};
        inline for (.{ p, ds, q, d_out, dk, dv }) |x| c.add(x);
        inline for (.{ r, hq, hk, d }) |x| c.add(@as(i32, @intCast(x)));
        c.add(scale);
        try cuda.launch.launch(o.k.attn_dkv, .{ .grid = .{ .x = @intCast(hk), .y = @intCast(r) }, .block = .{ .x = 256 } }, o.s, &c);
    }

    /// Depthwise causal conv over rows (pre-activation): y[t][c] = sum_j w[c][j] x[t - (K-1-j) dil][c]; w bf16 [C][K].
    pub fn convFwd(o: *Ops, x: u64, w: u64, y: u64, r: usize, c: usize, k: usize, dil: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, w, y }) |v| a.add(v);
        inline for (.{ r, c, k, dil }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.conv_fwd, flat(r * c), o.s, &a);
    }

    /// Its backward: dx (+)= the gradient at x.
    pub fn convBwd(o: *Ops, dy: u64, w: u64, dx: u64, r: usize, c: usize, k: usize, dil: usize, add: bool) !void {
        var a: cuda.Args = .{};
        inline for (.{ dy, w, dx }) |v| a.add(v);
        inline for (.{ r, c, k, dil }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(i32, @intFromBool(add)));
        try cuda.launch.launch(o.k.conv_bwd, flat(r * c), o.s, &a);
    }

    /// GatedDeltaNet gates: g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b) ([R][H], A_log / dt_bias bf16).
    pub fn gdnGates(o: *Ops, a_in: u64, b_in: u64, a_log: u64, dt_bias: u64, g: u64, beta: u64, r: usize, h: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ a_in, b_in, a_log, dt_bias, g, beta }) |v| a.add(v);
        inline for (.{ r, h }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.gdn_gates, flat(r * h), o.s, &a);
    }

    /// Its backward: da, db from dg, dbeta.
    pub fn gdnGatesBwd(o: *Ops, a_in: u64, beta: u64, a_log: u64, dt_bias: u64, dg: u64, dbeta: u64, da: u64, db: u64, r: usize, h: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ a_in, beta, a_log, dt_bias, dg, dbeta, da, db }) |v| a.add(v);
        inline for (.{ r, h }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.gdn_gates_bwd, flat(r * h), o.s, &a);
    }

    /// Key heads repeated to value heads ([R][hk][d] -> [R][hk rep][d]), or summed back (backward).
    pub fn heads(o: *Ops, in: u64, out: u64, r: usize, hk: usize, rep: usize, d: usize, backward: bool) !void {
        var a: cuda.Args = .{};
        inline for (.{ in, out }) |v| a.add(v);
        inline for (.{ r, hk, rep, d }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(i32, @intFromBool(backward)));
        try cuda.launch.launch(o.k.heads, flat(r * hk * (if (backward) 1 else rep) * d), o.s, &a);
    }

    /// The gated delta rule over R rows of H value heads (q, k [R][H][dk] already normed, v [R][H][dv]); every
    /// step's state into `states` ([H][R + 1][dk][dv] fp32).
    pub fn deltaFwd(o: *Ops, q: u64, k: u64, v: u64, g: u64, beta: u64, out: u64, states: u64, r: usize, h: usize, dk: usize, dv: usize) !void {
        if (dk > 128 or dv > 1024) return error.BadShape;
        var a: cuda.Args = .{};
        inline for (.{ q, k, v, g, beta, out, states }) |x| a.add(x);
        inline for (.{ r, h, dk, dv }) |x| a.add(@as(i32, @intCast(x)));
        try cuda.launch.launch(o.k.delta_fwd, .{ .grid = .{ .x = @intCast(h) }, .block = .{ .x = @intCast(dv) } }, o.s, &a);
    }

    /// Its backward: dq, dk, dv, dg, dbeta from dO and the stored states.
    pub fn deltaBwd(o: *Ops, q: u64, k: u64, v: u64, g: u64, beta: u64, d_out: u64, states: u64, dq: u64, dk_out: u64, dv_out: u64, dg: u64, dbeta: u64, r: usize, h: usize, dk: usize, dv: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ q, k, v, g, beta, d_out, states, dq, dk_out, dv_out, dg, dbeta }) |x| a.add(x);
        inline for (.{ r, h, dk, dv }) |x| a.add(@as(i32, @intCast(x)));
        try cuda.launch.launch(o.k.delta_bwd, .{ .grid = .{ .x = @intCast(h) }, .block = .{ .x = @intCast(dv) } }, o.s, &a);
    }

    /// out[rs] = scale * <a[rs], b[rs]> over D, for `rs` (row, stream) pairs.
    pub fn sdot(o: *Ops, a_in: u64, b_in: u64, out: u64, rs: usize, d: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ a_in, b_in, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(d)));
        a.add(scale);
        try cuda.launch.launch(o.k.sdot, .{ .grid = .{ .x = @intCast(rs) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// Its backward: da (+)= scale dout b, db (+)= scale dout a (0 to skip either).
    pub fn sdotBwd(o: *Ops, dout: u64, a_in: u64, b_in: u64, da: u64, db: u64, rs: usize, d: usize, scale: f32, add: bool) !void {
        var a: cuda.Args = .{};
        inline for (.{ dout, a_in, b_in, da, db }) |v| a.add(v);
        a.add(@as(i64, @intCast(rs)));
        a.add(@as(i32, @intCast(d)));
        a.add(scale);
        a.add(@as(i32, @intFromBool(add)));
        try cuda.launch.launch(o.k.sdot_bwd, flat(rs * d), o.s, &a);
    }

    /// Top-k of each row's E logits (ties to the lower id): ids [R][K] i32 and softmax weights over them.
    pub fn topk(o: *Ops, logits: u64, ids: u64, w: u64, r: usize, e: usize, k: usize) !void {
        if (k > 16) return error.BadShape;
        var a: cuda.Args = .{};
        inline for (.{ logits, ids, w }) |v| a.add(v);
        inline for (.{ r, e, k }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.topk, .{ .grid = .{ .x = @intCast((r + 7) / 8) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// Its backward into dlogits [R][E] (pre-zeroed; only the chosen entries are written).
    pub fn topkBwd(o: *Ops, ids: u64, w: u64, dw: u64, dlogits: u64, r: usize, e: usize, k: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ ids, w, dw, dlogits }) |v| a.add(v);
        inline for (.{ r, e, k }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.topk_bwd, flat(r), o.s, &a);
    }

    /// dst[n] = src[rows[n]] (D floats a row).
    pub fn gather(o: *Ops, src: u64, rows: u64, dst: u64, n: usize, d: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ src, rows, dst }) |v| a.add(v);
        inline for (.{ n, d }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.gather, flat(n * d), o.s, &a);
    }

    /// dst[rows[n]] += scale[n] src[n] (scale 0: 1), atomically.
    pub fn scatterAdd(o: *Ops, src: u64, rows: u64, scale: u64, dst: u64, n: usize, d: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ src, rows, scale, dst }) |v| a.add(v);
        inline for (.{ n, d }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.scatter_add, flat(n * d), o.s, &a);
    }

    /// Cross-entropy of rows of V logits against targets (i32): loss [R]; dlogits (0: skip) = gscale (softmax - onehot).
    pub fn xent(o: *Ops, logits: u64, target: u64, loss: u64, dlogits: u64, r: usize, v: usize, gscale: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ logits, target, loss, dlogits }) |x| a.add(x);
        a.add(@as(i32, @intCast(v)));
        a.add(gscale);
        try cuda.launch.launch(o.k.xent, .{ .grid = .{ .x = @intCast(r) }, .block = .{ .x = 1024 } }, o.s, &a);
    }

    /// Softmax over each row's given experts (ids [R][K] i32): weights [R][K].
    pub fn pickSoftmax(o: *Ops, logits: u64, ids: u64, w: u64, r: usize, e: usize, k: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ logits, ids, w }) |v| a.add(v);
        inline for (.{ r, e, k }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(o.k.pick_softmax, flat(r), o.s, &a);
    }

    /// Host bytes into training scratch at `ptr`, once the stream's queued work (which may still read that scratch,
    /// arenas being reused) is done: plain uploads are not ordered with this non-blocking stream.
    pub fn upload(o: *Ops, ptr: u64, bytes: []const u8) !void {
        try o.s.synchronize();
        try (cuda.DeviceBuffer{ .d = o.d, .ptr = ptr, .len = bytes.len }).upload(0, bytes);
    }
};

