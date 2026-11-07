#include "kernel.h"
#include <math.h>
#include <string.h>

#if !defined(CL_FORCE_SCALAR) && defined(__ARM_NEON)
#include <arm_neon.h>
#define CL_PATH_NEON 1
#elif !defined(CL_FORCE_SCALAR) && defined(__AVX2__)
#include <immintrin.h>
#define CL_PATH_AVX2 1
#endif

const char *cl_kernel_name(void) {
#if defined(CL_PATH_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(CL_NO_DOTPROD)
  return "neon+dotprod";
#elif defined(CL_PATH_NEON)
  return "neon";
#elif defined(CL_PATH_AVX2)
  return "avx2";
#else
  return "scalar";
#endif
}

float cl_f16_to_f32_soft(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ffu, bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else { /* subnormal: normalise */
      int e = -1;
      do { e++; man <<= 1; } while (!(man & 0x400u));
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3ffu) << 13);
    }
  } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
  else bits = sign | ((exp + 112) << 23) | (man << 13);
  float f;
  memcpy(&f, &bits, sizeof f);
  return f;
}

void cl_quantize_q8_0(const float *x, cl_block_q8_0 *y, int n) {
  for (int b = 0; b < n / CL_QK; b++) {
    float amax = 0;
    for (int j = 0; j < CL_QK; j++) { float a = fabsf(x[b * CL_QK + j]); if (a > amax) amax = a; }
    float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    y[b].d = d;
    for (int j = 0; j < CL_QK; j++) y[b].qs[j] = (int8_t)roundf(x[b * CL_QK + j] * id);
  }
}

float cl_dot_q4_0_q8_0_scalar(int nblocks, const cl_block_q4_0 *w, const cl_block_q8_0 *a) {
  float sum = 0;
  for (int b = 0; b < nblocks; b++) {
    int s = 0;
    for (int j = 0; j < CL_QK / 2; j++) {
      int lo = (w[b].qs[j] & 0x0f) - 8, hi = (w[b].qs[j] >> 4) - 8;
      s += lo * a[b].qs[j] + hi * a[b].qs[j + CL_QK / 2];
    }
    sum += (float)s * (cl_f16_to_f32(w[b].d) * a[b].d);
  }
  return sum;
}

float cl_dot_q4_0_q8_0(int nblocks, const cl_block_q4_0 *w, const cl_block_q8_0 *a) {
#if defined(CL_PATH_NEON)
  const uint8x16_t m4 = vdupq_n_u8(0x0f);
  const int8x16_t eight = vdupq_n_s8(8);
  float sum = 0;
  for (int b = 0; b < nblocks; b++) {
    uint8x16_t raw = vld1q_u8(w[b].qs);
    /* unpack: low nibbles are weights 0..15, high nibbles are weights 16..31; both shifted from 0..15 to -8..7 */
    int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, m4)), eight);
    int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
    int8x16_t a0 = vld1q_s8(a[b].qs), a1 = vld1q_s8(a[b].qs + 16);
#if defined(__ARM_FEATURE_DOTPROD) && !defined(CL_NO_DOTPROD)
    int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, a0), hi, a1);
#else
    /* |weight| <= 8 and |activation| <= 127, so two products added in 16 bits cannot overflow (at most 2032) */
    int16x8_t p0 = vmlal_s8(vmull_s8(vget_low_s8(lo), vget_low_s8(a0)), vget_high_s8(lo), vget_high_s8(a0));
    int16x8_t p1 = vmlal_s8(vmull_s8(vget_low_s8(hi), vget_low_s8(a1)), vget_high_s8(hi), vget_high_s8(a1));
    int32x4_t p = vpadalq_s16(vpaddlq_s16(p0), p1);
#endif
    sum += (float)vaddvq_s32(p) * (cl_f16_to_f32(w[b].d) * a[b].d);
  }
  return sum;
#elif defined(CL_PATH_AVX2)
  const __m128i m4 = _mm_set1_epi8(0x0f);
  const __m256i eight = _mm256_set1_epi8(8), ones = _mm256_set1_epi16(1);
  float sum = 0;
  for (int b = 0; b < nblocks; b++) {
    __m128i raw = _mm_loadu_si128((const __m128i *)(const void *)w[b].qs);
    __m128i lo = _mm_and_si128(raw, m4);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
    __m256i q = _mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1), eight); /* weights 0..31 as -8..7 */
    __m256i av = _mm256_loadu_si256((const __m256i *)(const void *)a[b].qs);
    /* maddubs wants unsigned x signed: use |q| and move q's sign onto the activation */
    __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(av, q));
    __m256i p32 = _mm256_madd_epi16(p16, ones);
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p32), _mm256_extracti128_si256(p32, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    sum += (float)_mm_cvtsi128_si32(s) * (cl_f16_to_f32(w[b].d) * a[b].d);
  }
  return sum;
#else
  return cl_dot_q4_0_q8_0_scalar(nblocks, w, a);
#endif
}

void cl_gemv_q4_0(const cl_block_q4_0 *w, int rows, int blocks_per_row, const cl_block_q8_0 *a, float *out) {
  for (int r = 0; r < rows; r++) out[r] = cl_dot_q4_0_q8_0(blocks_per_row, w + (size_t)r * blocks_per_row, a);
}

uint64_t cl_q4_0_pass(const void *bytes, uint64_t n, const cl_block_q8_0 *act, int blocks_per_row) {
  uint64_t nblocks = n / sizeof(cl_block_q4_0), rows = nblocks / (uint64_t)blocks_per_row, digest = 0;
  const cl_block_q4_0 *w = (const cl_block_q4_0 *)bytes;
  for (uint64_t r = 0; r < rows; r++) {
    float v = cl_dot_q4_0_q8_0(blocks_per_row, w + r * (uint64_t)blocks_per_row, act);
    if (!isfinite(v)) v = 0.0f; /* arbitrary bytes can hold an Inf or NaN scale */
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    digest = digest * 1099511628211ull + bits;
  }
  return digest;
}
