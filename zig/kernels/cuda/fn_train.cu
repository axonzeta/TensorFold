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
__device__ __forceinline__ float ew_f(int op, float u) {
  switch (op) {
    case 1: return u * sigm(u);
    case 2: return sigm(u);
    case 3: return u > 20.f ? u : log1pf(__expf(u));
    case 4: return __expf(u);
    default: return u;
  }
}
__device__ __forceinline__ float ew_df(int op, float u) {
  switch (op) {
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
