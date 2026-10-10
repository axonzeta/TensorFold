// ngram-learn training kernels (ours): the backward pass's helpers for Flash Next on CUDA.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// One packed GPTQ int4 matrix (cuda_int4.zig packWords / packScales of an [n, k] slice, groups of gs) -> bf16 W [n][k]:
// word at = ((nt * kg + g) * 32 + lane) * per + b holds inputs (g*gs + b*32)/8 + lane%4 (eight, nibble-shuffled) of
// column nt*8 + lane/4; its scale is scales[(nt * kg + g) * 8 + lane/4]; value = (code - 8) * scale.
extern "C" __global__ void fn_train_int4_dequant(const uint32_t* __restrict__ words, const __half* __restrict__ scales,
                                                 __nv_bfloat16* __restrict__ out, int n, int k, int gs) {
  const long long at = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)n * k / 8;
  if (at >= total) return;
  const int per = gs / 32;
  const int kg = k / gs;
  const int b = (int)(at % per);
  const long long r = at / per;
  const int lane = (int)(r % 32);
  const long long r2 = r / 32;
  const int g = (int)(r2 % kg);
  const int nt = (int)(r2 / kg);
  const int col = nt * 8 + lane / 4;
  const int k0 = ((g * gs + b * 32) / 8 + lane % 4) * 8;
  const float s = __half2float(scales[((long long)nt * kg + g) * 8 + lane / 4]);
  const uint32_t w = words[at];
  __nv_bfloat16* row = out + (long long)col * k + k0;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const int j = p < 4 ? 2 * p : 2 * (p - 4) + 1;
    const float code = (float)((w >> (4 * p)) & 0xF);
    row[j] = __float2bfloat16_rn((code - 8.0f) * s);
  }
}

// bf16 [rows][cols] -> [cols][rows] (32x32 tiles through shared memory).
extern "C" __global__ void fn_train_transpose_bf16(const uint16_t* __restrict__ in, uint16_t* __restrict__ out,
                                                   int rows, int cols) {
  __shared__ uint16_t tile[32][33];
  const int c = blockIdx.x * 32 + threadIdx.x;
  for (int i = threadIdx.y; i < 32; i += blockDim.y) {
    const int r = blockIdx.y * 32 + i;
    if (r < rows && c < cols) tile[i][threadIdx.x] = in[(long long)r * cols + c];
  }
  __syncthreads();
  const int r2 = blockIdx.y * 32 + threadIdx.x;
  for (int i = threadIdx.y; i < 32; i += blockDim.y) {
    const int c2 = blockIdx.x * 32 + i;
    if (c2 < cols && r2 < rows) out[(long long)c2 * rows + r2] = tile[threadIdx.x][i];
  }
}

// fp32 -> bf16 (round to nearest even), n values.
extern "C" __global__ void fn_train_f32_to_bf16(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16_rn(in[i]);
}

// ---- training runtime (fp32 activations and gradients) ----------------------------------------------------------

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(((uint32_t)v) << 16); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

// Block-wide sum of `v` (blockDim.x a multiple of 32, at most 1024).
__device__ float block_sum(float v) {
  __shared__ float part[32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  if (l == 0) part[w] = v;
  __syncthreads();
  v = (threadIdx.x < blockDim.x / 32) ? part[threadIdx.x] : 0.f;
  if (w == 0)
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (threadIdx.x == 0) part[0] = v;
  __syncthreads();
  const float r = part[0];
  __syncthreads();
  return r;
}

// Gain of a norm: mode 0 none, 1 (1 + w), 2 w (w bf16 [G*D]).
__device__ __forceinline__ float gain(const uint16_t* w, int mode, long long j) {
  return mode == 0 ? 1.f : (mode == 1 ? 1.f + bf2f(w[j]) : bf2f(w[j]));
}

// RMSNorm over groups of D: grid (G, R); y = x * rsqrt(mean(x^2) + eps) * gain, rinv [R][G].
extern "C" __global__ void fn_train_rms_fwd(const float* __restrict__ x, const uint16_t* __restrict__ w, int mode,
                                            float* __restrict__ y, float* __restrict__ rinv, int G, int D, float eps) {
  const long long base = ((long long)blockIdx.y * G + blockIdx.x) * D;
  float s = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) { const float v = x[base + i]; s += v * v; }
  const float r = rsqrtf(block_sum(s) / D + eps);
  if (threadIdx.x == 0) rinv[(long long)blockIdx.y * G + blockIdx.x] = r;
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    y[base + i] = x[base + i] * r * gain(w, mode, (long long)blockIdx.x * D + i);
}

// Its backward: dx (+)= r * (dyh - xh * mean(dyh * xh)), xh = x * r, dyh = dy * gain.
extern "C" __global__ void fn_train_rms_bwd(const float* __restrict__ dy, const float* __restrict__ x,
                                            const uint16_t* __restrict__ w, int mode, const float* __restrict__ rinv,
                                            float* __restrict__ dx, int G, int D, int add) {
  const long long base = ((long long)blockIdx.y * G + blockIdx.x) * D;
  const float r = rinv[(long long)blockIdx.y * G + blockIdx.x];
  float s = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    s += dy[base + i] * gain(w, mode, (long long)blockIdx.x * D + i) * x[base + i] * r;
  const float m = block_sum(s) / D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float dyh = dy[base + i] * gain(w, mode, (long long)blockIdx.x * D + i);
    const float v = r * (dyh - x[base + i] * r * m);
    dx[base + i] = add ? dx[base + i] + v : v;
  }
}

// Elementwise y = beta * f(alpha * x): op 0 identity, 1 silu, 2 sigmoid, 3 softplus, 4 exp.
__device__ __forceinline__ float ssq(float u) {  // sign(u) * sqrt(max(|u|, 1e-6)), sign(0) = 0 as torch's
  return u == 0.f ? 0.f : copysignf(sqrtf(fmaxf(fabsf(u), 1e-6f)), u);
}
__device__ __forceinline__ float ew_f(int op, float u) {
  switch (op) {
    case 5: return sigm(ssq(u));
    case 1: return u * sigm(u);
    case 2: return sigm(u);
    case 3: return u > 20.f ? u : log1pf(__expf(u));
    case 4: return __expf(u);
    default: return u;
  }
}
__device__ __forceinline__ float ew_df(int op, float u) {
  switch (op) {
    case 5: { const float s = sigm(ssq(u)); return fabsf(u) > 1e-6f ? s * (1.f - s) * 0.5f / sqrtf(fabsf(u)) : 0.f; }
    case 1: { const float s = sigm(u); return s * (1.f + u * (1.f - s)); }
    case 2: { const float s = sigm(u); return s * (1.f - s); }
    case 3: return sigm(u);
    case 4: return __expf(u);
    default: return 1.f;
  }
}
extern "C" __global__ void fn_train_ew(int op, const float* __restrict__ x, float* __restrict__ y, long long n,
                                       float alpha, float beta) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = beta * ew_f(op, alpha * x[i]);
}
// dx (+)= dy * beta * alpha * f'(alpha * x).
extern "C" __global__ void fn_train_ew_bwd(int op, const float* __restrict__ dy, const float* __restrict__ x,
                                           float* __restrict__ dx, long long n, float alpha, float beta, int add) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = dy[i] * beta * alpha * ew_df(op, alpha * x[i]);
  dx[i] = add ? dx[i] + v : v;
}

// y = a * b (op 0), a + b (op 1), or y += a * b (op 2); n values.
extern "C" __global__ void fn_train_ew2(int op, const float* __restrict__ a, const float* __restrict__ b,
                                        float* __restrict__ y, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (op == 0) y[i] = a[i] * b[i];
  else if (op == 1) y[i] = a[i] + b[i];
  else y[i] += a[i] * b[i];
}

// Hyper-connection mix: mixed[r][d] = mean_s u[r][s][d] * hn[r][s][d]  (S streams of D).
extern "C" __global__ void fn_train_mix_fwd(const float* __restrict__ u, const float* __restrict__ hn,
                                            float* __restrict__ mixed, int R, int S, int D) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const long long r = i / D, d = i % D;
  float s = 0.f;
  for (int k = 0; k < S; ++k) s += u[(r * S + k) * D + d] * hn[(r * S + k) * D + d];
  mixed[i] = s / S;
}
// Its backward: du = dmixed * hn / S, dhn (+)= dmixed * u / S.
extern "C" __global__ void fn_train_mix_bwd(const float* __restrict__ dmixed, const float* __restrict__ u,
                                            const float* __restrict__ hn, float* __restrict__ du,
                                            float* __restrict__ dhn, int R, int S, int D, int add) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * S * D) return;
  const long long r = i / ((long long)S * D), d = i % D;
  const float g = dmixed[r * D + d] / S;
  du[i] = g * hn[i];
  dhn[i] = add ? dhn[i] + g * u[i] : g * u[i];
}

// Injection: h[r][s][d] = res[r][s][d] + inj[r][s] * branch[r][d].
extern "C" __global__ void fn_train_inject_fwd(const float* __restrict__ res, const float* __restrict__ inj,
                                               const float* __restrict__ branch, float* __restrict__ h, int R, int S,
                                               int D) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * S * D) return;
  const long long r = i / ((long long)S * D), s = (i / D) % S, d = i % D;
  h[i] = res[i] + inj[r * S + s] * branch[r * D + d];
}
// Its backward (grid R, block 256): dbranch[r][d] = sum_s inj * dh; dinj[r][s] = sum_d dh * branch. dres = dh (aliased).
extern "C" __global__ void fn_train_inject_bwd(const float* __restrict__ dh, const float* __restrict__ inj,
                                               const float* __restrict__ branch, float* __restrict__ dbranch,
                                               float* __restrict__ dinj, int S, int D) {
  const long long r = blockIdx.x;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float s = 0.f;
    for (int k = 0; k < S; ++k) s += inj[r * S + k] * dh[(r * S + k) * D + d];
    dbranch[r * D + d] = s;
  }
  for (int k = 0; k < S; ++k) {
    float s = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) s += dh[(r * S + k) * D + d] * branch[r * D + d];
    s = block_sum(s);
    if (threadIdx.x == 0) dinj[r * S + k] = s;
  }
}

// bf16 bits -> fp32, n values.
extern "C" __global__ void fn_train_bf16_to_f32(const uint16_t* __restrict__ in, float* __restrict__ out, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = bf2f(in[i]);
}

// Strided copy: dst[r][dst_off + c] (+)= src[r][src_off + c] for c < width (fp32 rows of src_ld / dst_ld).
extern "C" __global__ void fn_train_copy2d(const float* __restrict__ src, int src_ld, int src_off, float* __restrict__ dst,
                                           int dst_ld, int dst_off, int rows, int width, int add) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)rows * width) return;
  const long long r = i / width, c = i % width;
  const float v = src[r * src_ld + src_off + c];
  float* d = dst + r * dst_ld + dst_off + c;
  *d = add ? *d + v : v;
}

// Rotate-half RoPE on the first `rot` dims of each head (x [R][H][hd]); cos/sin [R][rot] (halves repeated).
// fwd: y = x * cos + rotate_half(x) * sin; bwd (inverse = sign flip of sin): dx = dy * cos + rotate_half^T(dy * sin).
extern "C" __global__ void fn_train_rope(const float* __restrict__ x, const float* __restrict__ cs,
                                         const float* __restrict__ sn, float* __restrict__ y, int R, int H, int hd,
                                         int rot, int backward) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * H * hd) return;
  const int c = (int)(i % hd);
  const long long r = i / ((long long)H * hd);
  if (c >= rot) { y[i] = x[i]; return; }
  const int half = rot / 2;
  const float co = cs[r * rot + c], s = sn[r * rot + c];
  const long long base = i - c;
  float other;
  if (!backward) other = c < half ? -x[base + c + half] : x[base + c - half];  // rotate_half(x)[c]
  else other = c < half ? x[base + c + half] * sn[r * rot + c + half] : -x[base + c - half] * sn[r * rot + c - half];
  y[i] = backward ? x[i] * co + other : x[i] * co + other * s;
}

// Causal GQA attention, fp32. q [R][Hq][D], k/v [R][Hk][D] (head h reads kv h / (Hq/Hk)); P [Hq][R][R].
// Scores and softmax: grid (Hq, R), block 256 over keys j <= i.
extern "C" __global__ void fn_train_attn_probs(const float* __restrict__ q, const float* __restrict__ k,
                                               float* __restrict__ P, int R, int Hq, int Hk, int D, float scale) {
  const int h = blockIdx.x, i = blockIdx.y, kv = h / (Hq / Hk);
  extern __shared__ float qs[];
  for (int d = threadIdx.x; d < D; d += blockDim.x) qs[d] = q[((long long)i * Hq + h) * D + d];
  __syncthreads();
  float* row = P + ((long long)h * R + i) * R;
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < R; j += blockDim.x) {
    float s = -INFINITY;
    if (j <= i) {
      s = 0.f;
      const float* kr = k + ((long long)j * Hk + kv) * D;
      for (int d = 0; d < D; ++d) s += qs[d] * kr[d];
      s *= scale;
    }
    row[j] = s;
    mx = fmaxf(mx, s);
  }
  // block max
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  mx = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int j = threadIdx.x; j < R; j += blockDim.x) {
    const float e = j <= i ? __expf(row[j] - mx) : 0.f;
    row[j] = e;
    sum += e;
  }
  sum = block_sum(sum);
  for (int j = threadIdx.x; j < R; j += blockDim.x) row[j] /= sum;
}

// o[i][h][d] = sum_j P[h][i][j] v[j][kv][d]; grid (Hq, R), block over d.
extern "C" __global__ void fn_train_attn_out(const float* __restrict__ P, const float* __restrict__ v,
                                             float* __restrict__ o, int R, int Hq, int Hk, int D) {
  const int h = blockIdx.x, i = blockIdx.y, kv = h / (Hq / Hk);
  const float* row = P + ((long long)h * R + i) * R;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float s = 0.f;
    for (int j = 0; j <= i; ++j) s += row[j] * v[((long long)j * Hk + kv) * D + d];
    o[((long long)i * Hq + h) * D + d] = s;
  }
}

// Backward step 1, grid (Hq, R): dS[h][i][j] = P * (dP - rowsum(P * dP)), dP[j] = do_i . v_j; written over dP buffer.
extern "C" __global__ void fn_train_attn_ds(const float* __restrict__ P, const float* __restrict__ dO,
                                            const float* __restrict__ v, float* __restrict__ dS, int R, int Hq, int Hk,
                                            int D) {
  const int h = blockIdx.x, i = blockIdx.y, kv = h / (Hq / Hk);
  extern __shared__ float ds_do[];
  for (int d = threadIdx.x; d < D; d += blockDim.x) ds_do[d] = dO[((long long)i * Hq + h) * D + d];
  __syncthreads();
  const float* prow = P + ((long long)h * R + i) * R;
  float* drow = dS + ((long long)h * R + i) * R;
  float acc = 0.f;
  for (int j = threadIdx.x; j < R; j += blockDim.x) {
    float dp = 0.f;
    if (j <= i) {
      const float* vr = v + ((long long)j * Hk + kv) * D;
      for (int d = 0; d < D; ++d) dp += ds_do[d] * vr[d];
    }
    drow[j] = dp;
    acc += prow[j] * dp;
  }
  acc = block_sum(acc);
  for (int j = threadIdx.x; j < R; j += blockDim.x) drow[j] = j <= i ? prow[j] * (drow[j] - acc) : 0.f;
}

// dq[i][h][d] = scale * sum_j dS[h][i][j] k[j][kv][d]; grid (Hq, R), block over d.
extern "C" __global__ void fn_train_attn_dq(const float* __restrict__ dS, const float* __restrict__ k,
                                            float* __restrict__ dq, int R, int Hq, int Hk, int D, float scale) {
  const int h = blockIdx.x, i = blockIdx.y, kv = h / (Hq / Hk);
  const float* row = dS + ((long long)h * R + i) * R;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float s = 0.f;
    for (int j = 0; j <= i; ++j) s += row[j] * k[((long long)j * Hk + kv) * D + d];
    dq[((long long)i * Hq + h) * D + d] = scale * s;
  }
}

// dk[j][kv][d] = scale * sum_{h in kv's group, i >= j} dS[h][i][j] q[i][h][d];
// dv[j][kv][d] = sum_{h, i >= j} P[h][i][j] dO[i][h][d]; grid (Hk, R), block over d.
extern "C" __global__ void fn_train_attn_dkv(const float* __restrict__ P, const float* __restrict__ dS,
                                             const float* __restrict__ q, const float* __restrict__ dO,
                                             float* __restrict__ dk, float* __restrict__ dv, int R, int Hq, int Hk,
                                             int D, float scale) {
  const int kv = blockIdx.x, j = blockIdx.y, g = Hq / Hk;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float sk = 0.f, sv = 0.f;
    for (int h = kv * g; h < (kv + 1) * g; ++h)
      for (int i = j; i < R; ++i) {
        const long long pi = ((long long)h * R + i) * R + j;
        const long long qi = ((long long)i * Hq + h) * D + d;
        sk += dS[pi] * q[qi];
        sv += P[pi] * dO[qi];
      }
    dk[((long long)j * Hk + kv) * D + d] = scale * sk;
    dv[((long long)j * Hk + kv) * D + d] = sv;
  }
}

// Depthwise causal conv over rows: y[t][c] = sum_j w[c][j] * x[t - (K-1-j)*dil][c] (zero before row 0).
extern "C" __global__ void fn_train_conv_fwd(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                             float* __restrict__ y, int R, int C, int K, int dil) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * C) return;
  const int t = (int)(i / C), c = (int)(i % C);
  float s = 0.f;
  for (int j = 0; j < K; ++j) {
    const int src = t - (K - 1 - j) * dil;
    if (src >= 0) s += bf2f(w[(long long)c * K + j]) * x[(long long)src * C + c];
  }
  y[i] = s;
}
// dx[t][c] (+)= sum_j w[c][j] * dy[t + (K-1-j)*dil][c].
extern "C" __global__ void fn_train_conv_bwd(const float* __restrict__ dy, const uint16_t* __restrict__ w,
                                             float* __restrict__ dx, int R, int C, int K, int dil, int add) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * C) return;
  const int t = (int)(i / C), c = (int)(i % C);
  float s = 0.f;
  for (int j = 0; j < K; ++j) {
    const int dst = t + (K - 1 - j) * dil;
    if (dst < R) s += bf2f(w[(long long)c * K + j]) * dy[(long long)dst * C + c];
  }
  dx[i] = add ? dx[i] + s : s;
}

// GatedDeltaNet gates, per row and value head: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b).
extern "C" __global__ void fn_train_gdn_gates(const float* __restrict__ a, const float* __restrict__ b,
                                              const uint16_t* __restrict__ a_log, const uint16_t* __restrict__ dt_bias,
                                              float* __restrict__ g, float* __restrict__ beta, int R, int H) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * H) return;
  const int h = (int)(i % H);
  const float u = a[i] + bf2f(dt_bias[h]);
  g[i] = -__expf(bf2f(a_log[h])) * (u > 20.f ? u : log1pf(__expf(u)));
  beta[i] = sigm(b[i]);
}
// da = dg * -exp(A_log) * sigmoid(a + dt_bias); db = dbeta * beta * (1 - beta).
extern "C" __global__ void fn_train_gdn_gates_bwd(const float* __restrict__ a, const float* __restrict__ beta,
                                                  const uint16_t* __restrict__ a_log,
                                                  const uint16_t* __restrict__ dt_bias, const float* __restrict__ dg,
                                                  const float* __restrict__ dbeta, float* __restrict__ da,
                                                  float* __restrict__ db, int R, int H) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * H) return;
  const int h = (int)(i % H);
  da[i] = dg[i] * -__expf(bf2f(a_log[h])) * sigm(a[i] + bf2f(dt_bias[h]));
  db[i] = dbeta[i] * beta[i] * (1.f - beta[i]);
}

// Heads repeated: y[r][h][d] = x[r][h / rep][d] (fwd), or summed back: x[r][hk][d] = sum over its rep heads (bwd).
extern "C" __global__ void fn_train_heads(const float* __restrict__ in, float* __restrict__ out, int R, int Hk, int rep,
                                          int D, int backward) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (!backward) {
    if (i >= (long long)R * Hk * rep * D) return;
    const long long d = i % D, h = (i / D) % ((long long)Hk * rep), r = i / ((long long)D * Hk * rep);
    out[i] = in[(r * Hk + h / rep) * D + d];
  } else {
    if (i >= (long long)R * Hk * D) return;
    const long long d = i % D, hk = (i / D) % Hk, r = i / ((long long)D * Hk);
    float s = 0.f;
    for (int j = 0; j < rep; ++j) s += in[(r * Hk * rep + hk * rep + j) * D + d];
    out[i] = s;
  }
}

// Gated delta rule, one block per value head, one thread per value column j (blockDim = dv, dk <= 128):
// S <- e^g S;  m = S^T k;  delta = beta (v - m);  S <- S + k delta^T;  o = S^T q.
// States after each step go to Sall [H][R + 1][dk][dv] (step 0 = zeros), for the backward.
extern "C" __global__ void fn_train_delta_fwd(const float* __restrict__ q, const float* __restrict__ k,
                                              const float* __restrict__ v, const float* __restrict__ g,
                                              const float* __restrict__ beta, float* __restrict__ o,
                                              float* __restrict__ Sall, int R, int H, int dk, int dv) {
  const int h = blockIdx.x, j = threadIdx.x;
  __shared__ float ks[128], qs[128];
  float s[128];
  for (int i = 0; i < dk; ++i) s[i] = 0.f;
  float* base = Sall + (long long)h * (R + 1) * dk * dv;
  for (int i = 0; i < dk; ++i) base[(long long)i * dv + j] = 0.f;
  for (int t = 0; t < R; ++t) {
    __syncthreads();
    for (int i = j; i < dk; i += blockDim.x) {
      ks[i] = k[((long long)t * H + h) * dk + i];
      qs[i] = q[((long long)t * H + h) * dk + i];
    }
    __syncthreads();
    const float e = __expf(g[(long long)t * H + h]), b = beta[(long long)t * H + h];
    float m = 0.f;
    for (int i = 0; i < dk; ++i) { s[i] *= e; m += s[i] * ks[i]; }
    const float delta = b * (v[((long long)t * H + h) * dv + j] - m);
    float out = 0.f;
    float* st = base + (long long)(t + 1) * dk * dv;
    for (int i = 0; i < dk; ++i) {
      s[i] += ks[i] * delta;
      out += s[i] * qs[i];
      st[(long long)i * dv + j] = s[i];
    }
    o[((long long)t * H + h) * dv + j] = out;
  }
}

// Its backward, walking t down: dq, dk [R][H][dk], dv [R][H][dv], dg, dbeta [R][H].
extern "C" __global__ void fn_train_delta_bwd(const float* __restrict__ q, const float* __restrict__ k,
                                              const float* __restrict__ v, const float* __restrict__ g,
                                              const float* __restrict__ beta, const float* __restrict__ dO,
                                              const float* __restrict__ Sall, float* __restrict__ dq,
                                              float* __restrict__ dkk, float* __restrict__ dvv,
                                              float* __restrict__ dg, float* __restrict__ dbeta, int R, int H, int dk,
                                              int dv) {
  const int h = blockIdx.x, j = threadIdx.x;
  __shared__ float ks[128], qs[128], dqs[128], dks[128], red[2];
  float ds[128];
  for (int i = 0; i < dk; ++i) ds[i] = 0.f;
  const float* base = Sall + (long long)h * (R + 1) * dk * dv;
  for (int t = R - 1; t >= 0; --t) {
    __syncthreads();
    for (int i = j; i < dk; i += blockDim.x) {
      ks[i] = k[((long long)t * H + h) * dk + i];
      qs[i] = q[((long long)t * H + h) * dk + i];
      dqs[i] = 0.f;
      dks[i] = 0.f;
    }
    if (j < 2) red[j] = 0.f;
    __syncthreads();
    const long long th = (long long)t * H + h;
    const float e = __expf(g[th]), b = beta[th];
    const float d_o = dO[th * dv + j];
    const float* st = base + (long long)(t + 1) * dk * dv;
    const float* sp = base + (long long)t * dk * dv;
    // o = S_t^T q: dS_t += q do^T; dq_i = sum_j S_t[i][j] do_j
    float m = 0.f;
    for (int i = 0; i < dk; ++i) {
      ds[i] += qs[i] * d_o;
      atomicAdd(&dqs[i], st[(long long)i * dv + j] * d_o);
      m += e * sp[(long long)i * dv + j] * ks[i];
    }
    const float vj = v[th * dv + j];
    const float delta = b * (vj - m);
    // S_t = A + k delta^T: dA = dS; d delta_j = sum_i dS[i][j] k_i; dk_i += sum_j dS[i][j] delta_j
    float ddelta = 0.f;
    for (int i = 0; i < dk; ++i) {
      ddelta += ds[i] * ks[i];
      atomicAdd(&dks[i], ds[i] * delta);
    }
    dvv[th * dv + j] = b * ddelta;
    atomicAdd(&red[0], ddelta * (vj - m));  // dbeta
    const float dm = -b * ddelta;
    // m = A^T k: dA[i][j] += k_i dm_j; dk_i += sum_j A[i][j] dm_j; then A = e^g S_{t-1}: dg += <dA, A>, dS_{t-1} = e dA
    float dga = 0.f;
    for (int i = 0; i < dk; ++i) {
      const float a = e * sp[(long long)i * dv + j];
      const float da = ds[i] + ks[i] * dm;
      atomicAdd(&dks[i], a * dm);
      dga += da * a;
      ds[i] = e * da;
    }
    atomicAdd(&red[1], dga);
    __syncthreads();
    for (int i = j; i < dk; i += blockDim.x) {
      dq[th * dk + i] = dqs[i];
      dkk[th * dk + i] = dks[i];
    }
    if (j == 0) {
      dbeta[th] = red[0];
      dg[th] = red[1];
    }
  }
}

// Per (row, stream) dot product: out[r][s] = scale * sum_d a[r][s][d] * b[r][s][d]; grid R*S, block 256.
extern "C" __global__ void fn_train_sdot(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out,
                                         int D, float scale) {
  const long long rs = blockIdx.x;
  float s = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) s += a[rs * D + d] * b[rs * D + d];
  s = block_sum(s);
  if (threadIdx.x == 0) out[rs] = scale * s;
}
// Its backward: da (+)= scale * dout * b, db (+)= scale * dout * a (either may be null).
extern "C" __global__ void fn_train_sdot_bwd(const float* __restrict__ dout, const float* __restrict__ a,
                                             const float* __restrict__ b, float* __restrict__ da, float* __restrict__ db,
                                             long long RS, int D, float scale, int add) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= RS * D) return;
  const float g = scale * dout[i / D];
  if (da) da[i] = add ? da[i] + g * b[i] : g * b[i];
  if (db) db[i] = add ? db[i] + g * a[i] : g * a[i];
}

// MoE routing: top-k of each row's E logits (ties to the lower id), weights = softmax over the chosen logits.
// One warp a row (E <= 1024): idx [R][K] int32, w [R][K].
extern "C" __global__ void fn_train_topk_softmax(const float* __restrict__ logits, int* __restrict__ idx,
                                                 float* __restrict__ w, int R, int E, int K) {
  const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (r >= R) return;
  const float* l = logits + (long long)r * E;
  float chosen[16];
  int ids[16];
  for (int k = 0; k < K; ++k) {
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int e = lane; e < E; e += 32) {
      bool used = false;
      for (int p = 0; p < k; ++p) used |= ids[p] == e;
      const float v = l[e];
      if (!used && (v > best || (v == best && e < at))) { best = v; at = e; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oa = __shfl_xor_sync(0xffffffffu, at, o);
      if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    }
    chosen[k] = best;
    ids[k] = at;
  }
  if (lane == 0) {
    float sum = 0.f;
    for (int k = 0; k < K; ++k) sum += __expf(chosen[k] - chosen[0]);
    for (int k = 0; k < K; ++k) {
      idx[(long long)r * K + k] = ids[k];
      w[(long long)r * K + k] = __expf(chosen[k] - chosen[0]) / sum;
    }
  }
}
// Its backward: dlogits[r][e] = w_k (dw_k - sum_j w_j dw_j) at the chosen e, zero elsewhere (dlogits pre-zeroed).
extern "C" __global__ void fn_train_topk_softmax_bwd(const int* __restrict__ idx, const float* __restrict__ w,
                                                     const float* __restrict__ dw, float* __restrict__ dlogits, int R,
                                                     int E, int K) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  float dot = 0.f;
  for (int k = 0; k < K; ++k) dot += w[(long long)r * K + k] * dw[(long long)r * K + k];
  for (int k = 0; k < K; ++k)
    dlogits[(long long)r * E + idx[(long long)r * K + k]] = w[(long long)r * K + k] * (dw[(long long)r * K + k] - dot);
}

// dst[n][:] = src[rows[n]][:] (D floats a row).
extern "C" __global__ void fn_train_gather(const float* __restrict__ src, const int* __restrict__ rows,
                                           float* __restrict__ dst, int N, int D) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)N * D) return;
  dst[i] = src[(long long)rows[i / D] * D + i % D];
}
// dst[rows[n]][:] += scale[n] * src[n][:] (scale null: 1), atomically.
extern "C" __global__ void fn_train_scatter_add(const float* __restrict__ src, const int* __restrict__ rows,
                                                const float* __restrict__ scale, float* __restrict__ dst, int N, int D) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)N * D) return;
  const long long n = i / D;
  atomicAdd(&dst[(long long)rows[n] * D + i % D], (scale ? scale[n] : 1.f) * src[i]);
}

// Cross-entropy on rows of V logits: loss[r] = lse - logit[target]; dlogits = (softmax - onehot) * gscale.
// One block a row.
extern "C" __global__ void fn_train_xent(const float* __restrict__ logits, const int* __restrict__ target,
                                         float* __restrict__ loss, float* __restrict__ dlogits, int V, float gscale) {
  const long long r = blockIdx.x;
  const float* l = logits + r * V;
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < V; i += blockDim.x) mx = fmaxf(mx, l[i]);
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  mx = red[0];
  __syncthreads();
  float s = 0.f;
  for (int i = threadIdx.x; i < V; i += blockDim.x) s += __expf(l[i] - mx);
  s = block_sum(s);
  const float lse = mx + logf(s);
  const int t = target[r];
  if (threadIdx.x == 0) loss[r] = lse - l[t];
  if (dlogits)
    for (int i = threadIdx.x; i < V; i += blockDim.x)
      dlogits[r * V + i] = gscale * (__expf(l[i] - lse) - (i == t ? 1.f : 0.f));
}
