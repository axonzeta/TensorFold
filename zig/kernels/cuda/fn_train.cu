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
