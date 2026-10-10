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
