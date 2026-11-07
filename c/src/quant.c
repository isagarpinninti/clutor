#define _GNU_SOURCE
#include "quant.h"
#include "kernel.h"
#include <math.h>
#include "cl_thread.h"
#include <stdlib.h>
#include <string.h>
#include "plat.h"

/* Block layouts are ggml's (llama.cpp/ggml-common.h); the maths below follows its dequantize_row_* functions. */
#define QK_K 256
typedef struct { uint16_t d; uint8_t qs[16]; } b_q4_0;
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } b_q5_0;
typedef struct { uint16_t d; int8_t qs[32]; } b_q8_0;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } b_q4_K;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[QK_K / 8]; uint8_t qs[QK_K / 2]; } b_q5_K;
typedef struct { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; uint16_t d; } b_q6_K;
_Static_assert(sizeof(b_q4_K) == 144 && sizeof(b_q5_K) == 176 && sizeof(b_q6_K) == 210 && sizeof(b_q5_0) == 22 && sizeof(b_q8_0) == 34, "block sizes");

int cl_type_supported(uint32_t t) { return cl_type_block(t) != 0; }
uint32_t cl_type_block(uint32_t t) {
  switch (t) { case CL_T_F32: case CL_T_F16: case CL_T_BF16: return 1; case CL_T_Q4_0: case CL_T_Q5_0: case CL_T_Q8_0: return 32; case CL_T_Q4_K: case CL_T_Q5_K: case CL_T_Q6_K: return 256; default: return 0; }
}
uint32_t cl_type_bytes(uint32_t t) {
  switch (t) { case CL_T_F32: return 4; case CL_T_F16: case CL_T_BF16: return 2; case CL_T_Q4_0: return 18; case CL_T_Q5_0: return 22; case CL_T_Q8_0: return 34; case CL_T_Q4_K: return 144; case CL_T_Q5_K: return 176; case CL_T_Q6_K: return 210; default: return 0; }
}
uint64_t cl_row_bytes(uint32_t t, uint64_t n) { return n / cl_type_block(t) * cl_type_bytes(t); }

static float bf16(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f; }

/* Q4_K / Q5_K store 8 six-bit (scale, min) pairs packed into 12 bytes. */
static void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
  else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}


/* The eight 6-bit scales and mins of a Q4_K / Q5_K block, unpacked with a few word operations (the layout ggml uses) instead of a branch per value. */
static inline void unpack_scales(const uint8_t *s12, uint8_t sc[8], uint8_t mn[8]) {
  uint32_t u[4];
  memcpy(u, s12, 12);
  u[3] = ((u[2] >> 4) & 0x0f0f0f0fu) | (((u[1] >> 6) & 0x03030303u) << 4);
  const uint32_t aux = u[1] & 0x3f3f3f3fu;
  u[1] = (u[2] & 0x0f0f0f0fu) | (((u[0] >> 6) & 0x03030303u) << 4);
  u[2] = aux;
  u[0] &= 0x3f3f3f3fu;
  memcpy(sc, &u[0], 8); memcpy(mn, &u[2], 8);   /* little-endian: byte j of u[0..1] is scale j, byte j of u[2..3] is min j */
}

static void deq_q4_K(const b_q4_K *x, float *y) {
  const float d = cl_f16_to_f32(x->d), min = cl_f16_to_f32(x->dmin);
  const uint8_t *q = x->qs;
  int is = 0;
  for (int j = 0; j < QK_K; j += 64) {
    uint8_t sc, m;
    scale_min_k4(is + 0, x->scales, &sc, &m);
    float d1 = d * sc, m1 = min * m;
    scale_min_k4(is + 1, x->scales, &sc, &m);
    float d2 = d * sc, m2 = min * m;
    for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
    q += 32; is += 2;
  }
}

static void deq_q5_K(const b_q5_K *x, float *y) {
  const float d = cl_f16_to_f32(x->d), min = cl_f16_to_f32(x->dmin);
  const uint8_t *ql = x->qs, *qh = x->qh;
  int is = 0;
  uint8_t u1 = 1, u2 = 2;
  for (int j = 0; j < QK_K; j += 64) {
    uint8_t sc, m;
    scale_min_k4(is + 0, x->scales, &sc, &m);
    float d1 = d * sc, m1 = min * m;
    scale_min_k4(is + 1, x->scales, &sc, &m);
    float d2 = d * sc, m2 = min * m;
    for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
    for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
    ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
  }
}

static void deq_q6_K(const b_q6_K *x, float *y) {
  const float d = cl_f16_to_f32(x->d);
  const uint8_t *ql = x->ql, *qh = x->qh;
  const int8_t *sc = x->scales;
  for (int n = 0; n < QK_K; n += 128) {
    for (int l = 0; l < 32; l++) {
      int is = l / 16;
      int q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      int q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      int q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l] = d * sc[is + 0] * q1;
      y[l + 32] = d * sc[is + 2] * q2;
      y[l + 64] = d * sc[is + 4] * q3;
      y[l + 96] = d * sc[is + 6] * q4;
    }
    y += 128; ql += 64; qh += 32; sc += 8;
  }
}

void cl_dequant_row(uint32_t type, const void *src, float *dst, int n) {
  const uint8_t *p = src;
  switch (type) {
    case CL_T_F32: memcpy(dst, src, (size_t)n * 4); return;
    case CL_T_F16: for (int i = 0; i < n; i++) { uint16_t h; memcpy(&h, p + 2 * i, 2); dst[i] = cl_f16_to_f32(h); } return;
    case CL_T_BF16: for (int i = 0; i < n; i++) { uint16_t h; memcpy(&h, p + 2 * i, 2); dst[i] = bf16(h); } return;
    case CL_T_Q4_0: for (int b = 0; b < n / 32; b++) {
      const b_q4_0 *x = (const b_q4_0 *)(const void *)(p + (size_t)b * 18); float d = cl_f16_to_f32(x->d);
      for (int j = 0; j < 16; j++) { dst[b * 32 + j] = ((x->qs[j] & 0xF) - 8) * d; dst[b * 32 + j + 16] = ((x->qs[j] >> 4) - 8) * d; }
    } return;
    case CL_T_Q5_0: for (int b = 0; b < n / 32; b++) {
      const b_q5_0 *x = (const b_q5_0 *)(const void *)(p + (size_t)b * 22); float d = cl_f16_to_f32(x->d);
      uint32_t qh; memcpy(&qh, x->qh, 4);
      for (int j = 0; j < 16; j++) {
        int x0 = ((x->qs[j] & 0xF) | (((qh >> j) & 1) << 4)) - 16, x1 = ((x->qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16;
        dst[b * 32 + j] = x0 * d; dst[b * 32 + j + 16] = x1 * d;
      }
    } return;
    case CL_T_Q8_0: for (int b = 0; b < n / 32; b++) {
      const b_q8_0 *x = (const b_q8_0 *)(const void *)(p + (size_t)b * 34); float d = cl_f16_to_f32(x->d);
      for (int j = 0; j < 32; j++) dst[b * 32 + j] = x->qs[j] * d;
    } return;
    case CL_T_Q4_K: for (int b = 0; b < n / QK_K; b++) deq_q4_K((const b_q4_K *)(const void *)(p + (size_t)b * 144), dst + b * QK_K); return;
    case CL_T_Q5_K: for (int b = 0; b < n / QK_K; b++) deq_q5_K((const b_q5_K *)(const void *)(p + (size_t)b * 176), dst + b * QK_K); return;
    case CL_T_Q6_K: for (int b = 0; b < n / QK_K; b++) deq_q6_K((const b_q6_K *)(const void *)(p + (size_t)b * 210), dst + b * QK_K); return;
    default: memset(dst, 0, (size_t)n * 4);
  }
}

/* ---- integer dot products: K-quant weights against 8-bit activations (the maths of ggml's vec_dot for these types) ---- */
#if !defined(CL_FORCE_SCALAR) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define CL_Q_NEON 1
#elif !defined(CL_FORCE_SCALAR) && defined(__AVX2__)
#include <immintrin.h>
#define CL_Q_AVX2 1
#endif
const char *cl_quant_path(void) {
#ifdef CL_Q_NEON
  return "neon+dotprod";
#elif defined(CL_Q_AVX2)
  return "avx2";
#else
  return "scalar";
#endif
}
int cl_type_integer_dot(uint32_t t) { return t == CL_T_Q4_K || t == CL_T_Q5_K || t == CL_T_Q6_K; }

int cl_xq_init(cl_xq *q, int n) { q->n = n; q->b = n % 256 == 0 ? malloc((size_t)(n / 256) * sizeof(cl_q8k)) : NULL; return q->b != NULL; }
void cl_xq_free(cl_xq *q) { free(q->b); q->b = NULL; q->n = 0; }
void cl_xq_quantize(cl_xq *q, const float *x) {
  for (int i = 0; i < q->n / 256; i++) {
    const float *xs = x + i * 256;
    float amax = 0;
    for (int j = 0; j < 256; j++) { float a = fabsf(xs[j]); if (a > amax) amax = a; }
    cl_q8k *b = &q->b[i];
    float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    b->d = d;
    for (int j = 0; j < 256; j++) b->qs[j] = (int8_t)roundf(xs[j] * id);
    for (int g = 0; g < 16; g++) { int s = 0; for (int j = 0; j < 16; j++) s += b->qs[g * 16 + j]; b->bsums[g] = (int16_t)s; }
  }
}

#if !defined(CL_Q_NEON) && !defined(CL_Q_AVX2)
static inline int32_t dot16_s(const int8_t *a, const int8_t *b) { int32_t s = 0; for (int i = 0; i < 16; i++) s += a[i] * b[i]; return s; }
#endif

#ifdef CL_Q_NEON
/* ---- NEON: unpack a block's weights to int8 once with vector operations, so that a batch of tokens can reuse it ---- */
static inline void unpack_q5_K(const b_q5_K *x, int8_t *q) {
  const uint8x16_t m4 = vdupq_n_u8(0xf), one = vdupq_n_u8(1);
  const uint8x16_t qh0 = vld1q_u8(x->qh), qh1 = vld1q_u8(x->qh + 16);
  const uint8_t *ql = x->qs;
  for (int c = 0; c < 4; c++) {   /* chunk c holds elements 64c..64c+63: low nibbles with bit 2c of qh, then high nibbles with bit 2c+1 */
    uint8x16_t a0 = vld1q_u8(ql + 32 * c), a1 = vld1q_u8(ql + 32 * c + 16);
    int s0 = 2 * c, s1 = 2 * c + 1;
    uint8x16_t lo0 = vorrq_u8(vandq_u8(a0, m4), vshlq_n_u8(vandq_u8(vshlq_u8(qh0, vdupq_n_s8((int8_t)-s0)), one), 4));
    uint8x16_t lo1 = vorrq_u8(vandq_u8(a1, m4), vshlq_n_u8(vandq_u8(vshlq_u8(qh1, vdupq_n_s8((int8_t)-s0)), one), 4));
    uint8x16_t hi0 = vorrq_u8(vshrq_n_u8(a0, 4), vshlq_n_u8(vandq_u8(vshlq_u8(qh0, vdupq_n_s8((int8_t)-s1)), one), 4));
    uint8x16_t hi1 = vorrq_u8(vshrq_n_u8(a1, 4), vshlq_n_u8(vandq_u8(vshlq_u8(qh1, vdupq_n_s8((int8_t)-s1)), one), 4));
    vst1q_s8(q + 64 * c, vreinterpretq_s8_u8(lo0)); vst1q_s8(q + 64 * c + 16, vreinterpretq_s8_u8(lo1));
    vst1q_s8(q + 64 * c + 32, vreinterpretq_s8_u8(hi0)); vst1q_s8(q + 64 * c + 48, vreinterpretq_s8_u8(hi1));
  }
}
static inline void unpack_q6_K(const b_q6_K *x, int8_t *q) {
  const uint8x16_t m4 = vdupq_n_u8(0xf), m3 = vdupq_n_u8(3);
  const int8x16_t off = vdupq_n_s8(32);
  const uint8_t *ql = x->ql, *qh = x->qh;
  for (int n = 0; n < 256; n += 128) {
    for (int h = 0; h < 32; h += 16) {
      uint8x16_t l0 = vld1q_u8(ql + h), l1 = vld1q_u8(ql + 32 + h), hh = vld1q_u8(qh + h);
      uint8x16_t v0 = vorrq_u8(vandq_u8(l0, m4), vshlq_n_u8(vandq_u8(hh, m3), 4));
      uint8x16_t v1 = vorrq_u8(vandq_u8(l1, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 2), m3), 4));
      uint8x16_t v2 = vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 4), m3), 4));
      uint8x16_t v3 = vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 6), m3), 4));
      vst1q_s8(q + n + h, vsubq_s8(vreinterpretq_s8_u8(v0), off)); vst1q_s8(q + n + 32 + h, vsubq_s8(vreinterpretq_s8_u8(v1), off));
      vst1q_s8(q + n + 64 + h, vsubq_s8(vreinterpretq_s8_u8(v2), off)); vst1q_s8(q + n + 96 + h, vsubq_s8(vreinterpretq_s8_u8(v3), off));
    }
    ql += 64; qh += 32;
  }
}
/* sum of mins[j] * (bsums[2j] + bsums[2j+1]) */
static inline int32_t mins_dot(const uint8_t *mn, const int16_t *bsums) {
  int16x8_t p = vpaddq_s16(vld1q_s16(bsums), vld1q_s16(bsums + 8));
  int16x8_t m = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mn)));
  return vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(p), vget_low_s16(m)), vmull_high_s16(p, m)));
}

#endif

static float dot_q4_K(int nb, const b_q4_K *x, const cl_q8k *y) {
  float sum = 0;
  for (int i = 0; i < nb; i++) {
    uint8_t sc[8], mn[8];
    unpack_scales(x[i].scales, sc, mn);
    const uint8_t *q4 = x[i].qs;
    const int8_t *q8 = y[i].qs;
    int32_t sumi = 0;
#ifdef CL_Q_NEON
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    int32x4_t acc = vdupq_n_s32(0);   /* scaled partial sums stay in the vector; one horizontal add per block (integer sums are exact in any order) */
    for (int g = 0; g < 4; g++) {
      uint8x16_t v0 = vld1q_u8(q4 + 32 * g), v1 = vld1q_u8(q4 + 32 * g + 16);
      int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(v0, m4)), lo1 = vreinterpretq_s8_u8(vandq_u8(v1, m4));
      int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), hi1 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
      int32x4_t a = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, vld1q_s8(q8 + 64 * g)), lo1, vld1q_s8(q8 + 64 * g + 16));
      int32x4_t b = vdotq_s32(vdotq_s32(vdupq_n_s32(0), hi0, vld1q_s8(q8 + 64 * g + 32)), hi1, vld1q_s8(q8 + 64 * g + 48));
      acc = vmlaq_n_s32(vmlaq_n_s32(acc, a, sc[2 * g]), b, sc[2 * g + 1]);
    }
    sumi = vaddvq_s32(acc);
#elif defined(CL_Q_AVX2)
    {   /* nibbles are unsigned 0..15 and the activations signed, which is exactly what maddubs wants; each pair-sum fits int16 (at most 15*127*2) */
      const __m256i m4 = _mm256_set1_epi8(0x0f);
      __m256i acc = _mm256_setzero_si256();
      for (int g = 0; g < 4; g++) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(q4 + 32 * g));
        __m256i lo = _mm256_and_si256(v, m4), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(const void *)(q8 + 64 * g)), a1 = _mm256_loadu_si256((const __m256i *)(const void *)(q8 + 64 * g + 32));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, a0), _mm256_set1_epi16((short)sc[2 * g])));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, a1), _mm256_set1_epi16((short)sc[2 * g + 1])));
      }
      __m128i t = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
      t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
      t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
      sumi = _mm_cvtsi128_si32(t);
    }
#else
    for (int g = 0; g < 4; g++) {
      int32_t a = 0, b = 0;
      for (int l = 0; l < 32; l++) { a += (q4[32 * g + l] & 0xF) * q8[64 * g + l]; b += (q4[32 * g + l] >> 4) * q8[64 * g + 32 + l]; }
      sumi += sc[2 * g] * a + sc[2 * g + 1] * b;
    }
#endif
    int32_t summs = 0;
    for (int j = 0; j < 8; j++) summs += mn[j] * (y[i].bsums[2 * j] + y[i].bsums[2 * j + 1]);
    sum += y[i].d * (cl_f16_to_f32(x[i].d) * (float)sumi - cl_f16_to_f32(x[i].dmin) * (float)summs);
  }
  return sum;
}

/* Q5_K: like Q4_K, but each weight is a 5-bit value (a nibble plus one bit from qh), so the unpacked values are 0..31. ggml's vec_dot_q5_K_q8_K computes the same integer sums. */
static float dot_q5_K(int nb, const b_q5_K *x, const cl_q8k *y) {
  float sum = 0;
  for (int i = 0; i < nb; i++) {
    uint8_t sc[8], mn[8], q[256];
    unpack_scales(x[i].scales, sc, mn);
#ifdef CL_Q_NEON
    unpack_q5_K(&x[i], (int8_t *)q);
#else
    const uint8_t *ql = x[i].qs, *qh = x[i].qh;
    uint8_t u1 = 1, u2 = 2;
    for (int c = 0; c < 4; c++) {
      for (int l = 0; l < 32; l++) {
        q[64 * c + l] = (uint8_t)((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0));
        q[64 * c + 32 + l] = (uint8_t)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0));
      }
      ql += 32; u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
    }
#endif
    const int8_t *q8 = y[i].qs;
    int32_t sumi = 0;
#ifdef CL_Q_NEON
    {
      int32x4_t acc = vdupq_n_s32(0);
      for (int g = 0; g < 8; g++) {
        int32x4_t a = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8((const int8_t *)q + 32 * g), vld1q_s8(q8 + 32 * g)), vld1q_s8((const int8_t *)q + 32 * g + 16), vld1q_s8(q8 + 32 * g + 16));
        acc = vmlaq_n_s32(acc, a, sc[g]);
      }
      sumi = vaddvq_s32(acc);
    }
#elif defined(CL_Q_AVX2)
    {   /* the values are unsigned 0..31 and the activations signed: maddubs pair-sums are at most 31*127*2 and fit int16 */
      __m256i acc = _mm256_setzero_si256();
      for (int g = 0; g < 8; g++) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(q + 32 * g)), a = _mm256_loadu_si256((const __m256i *)(const void *)(q8 + 32 * g));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(v, a), _mm256_set1_epi16((short)sc[g])));
      }
      __m128i t = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
      t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
      t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
      sumi = _mm_cvtsi128_si32(t);
    }
#else
    for (int g = 0; g < 8; g++) { int32_t a = 0; for (int l = 0; l < 32; l++) a += q[32 * g + l] * q8[32 * g + l]; sumi += sc[g] * a; }
#endif
    int32_t summs = 0;
    for (int j = 0; j < 8; j++) summs += mn[j] * (y[i].bsums[2 * j] + y[i].bsums[2 * j + 1]);
    sum += y[i].d * (cl_f16_to_f32(x[i].d) * (float)sumi - cl_f16_to_f32(x[i].dmin) * (float)summs);
  }
  return sum;
}

static float dot_q6_K(int nb, const b_q6_K *x, const cl_q8k *y) {
  float sum = 0;
  for (int i = 0; i < nb; i++) {
    int8_t q[256];
#ifdef CL_Q_NEON
    unpack_q6_K(&x[i], q);
#else
    const uint8_t *ql = x[i].ql, *qh = x[i].qh;
    for (int n = 0; n < 256; n += 128) {
      for (int l = 0; l < 32; l++) {
        q[n + l] = (int8_t)(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
        q[n + l + 32] = (int8_t)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
        q[n + l + 64] = (int8_t)(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
        q[n + l + 96] = (int8_t)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
      }
      ql += 64; qh += 32;
    }
#endif
    int32_t sumi = 0;
#if defined(CL_Q_AVX2)
    {   /* |q| <= 32 and |a| <= 127, so abs(q) x sign-moved a pair-sums to at most 8128 and fits int16 */
      __m256i acc = _mm256_setzero_si256();
      for (int j = 0; j < 16; j += 2) {
        __m256i qv = _mm256_loadu_si256((const __m256i *)(const void *)(q + 16 * j)), av = _mm256_loadu_si256((const __m256i *)(const void *)(y[i].qs + 16 * j));
        __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(qv, qv), _mm256_sign_epi8(av, qv));
        __m256i scv = _mm256_setr_epi16((short)x[i].scales[j], (short)x[i].scales[j], (short)x[i].scales[j], (short)x[i].scales[j], (short)x[i].scales[j], (short)x[i].scales[j], (short)x[i].scales[j], (short)x[i].scales[j],
                                        (short)x[i].scales[j + 1], (short)x[i].scales[j + 1], (short)x[i].scales[j + 1], (short)x[i].scales[j + 1], (short)x[i].scales[j + 1], (short)x[i].scales[j + 1], (short)x[i].scales[j + 1], (short)x[i].scales[j + 1]);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, scv));
      }
      __m128i t = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
      t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
      t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
      sumi = _mm_cvtsi128_si32(t);
    }
#elif defined(CL_Q_NEON)
    {
      int32x4_t acc = vdupq_n_s32(0);
      for (int j = 0; j < 16; j++) acc = vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), vld1q_s8(q + 16 * j), vld1q_s8(y[i].qs + 16 * j)), x[i].scales[j]);
      sumi = vaddvq_s32(acc);
    }
#else
    for (int j = 0; j < 16; j++) sumi += x[i].scales[j] * dot16_s(q + 16 * j, y[i].qs + 16 * j);
#endif
    sum += cl_f16_to_f32(x[i].d) * y[i].d * (float)sumi;
  }
  return sum;
}


#ifdef CL_Q_NEON
/* nb inputs against one weight row: each block is unpacked once, then multiplied with every input. sum[b] receives the dot product with input b. */
static void dotb_q4_K(int nblk, const b_q4_K *x, const cl_xq *xq, int nb, float *sum) {
  const uint8x16_t m4 = vdupq_n_u8(0x0f);
  for (int b = 0; b < nb; b++) sum[b] = 0;
  for (int i = 0; i < nblk; i++) {
    uint8_t sc[8], mn[8];
    unpack_scales(x[i].scales, sc, mn);
    int8x16_t w[16];
    for (int g = 0; g < 4; g++) {
      uint8x16_t v0 = vld1q_u8(x[i].qs + 32 * g), v1 = vld1q_u8(x[i].qs + 32 * g + 16);
      w[4 * g] = vreinterpretq_s8_u8(vandq_u8(v0, m4)); w[4 * g + 1] = vreinterpretq_s8_u8(vandq_u8(v1, m4));
      w[4 * g + 2] = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)); w[4 * g + 3] = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
    }
    const float d = cl_f16_to_f32(x[i].d), dm = cl_f16_to_f32(x[i].dmin);
    for (int b = 0; b < nb; b++) {
      const cl_q8k *y = &xq[b].b[i];
      const int8_t *q8 = y->qs;
      int32x4_t acc = vdupq_n_s32(0);
      for (int g = 0; g < 4; g++) {
        int32x4_t a = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w[4 * g], vld1q_s8(q8 + 64 * g)), w[4 * g + 1], vld1q_s8(q8 + 64 * g + 16));
        int32x4_t c = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w[4 * g + 2], vld1q_s8(q8 + 64 * g + 32)), w[4 * g + 3], vld1q_s8(q8 + 64 * g + 48));
        acc = vmlaq_n_s32(vmlaq_n_s32(acc, a, sc[2 * g]), c, sc[2 * g + 1]);
      }
      sum[b] += y->d * (d * (float)vaddvq_s32(acc) - dm * (float)mins_dot(mn, y->bsums));
    }
  }
}
static void dotb_q5_K(int nblk, const b_q5_K *x, const cl_xq *xq, int nb, float *sum) {
  for (int b = 0; b < nb; b++) sum[b] = 0;
  for (int i = 0; i < nblk; i++) {
    uint8_t sc[8], mn[8];
    int8_t q[256];
    unpack_scales(x[i].scales, sc, mn);
    unpack_q5_K(&x[i], q);
    const float d = cl_f16_to_f32(x[i].d), dm = cl_f16_to_f32(x[i].dmin);
    for (int b = 0; b < nb; b++) {
      const cl_q8k *y = &xq[b].b[i];
      int32x4_t acc = vdupq_n_s32(0);
      for (int g = 0; g < 8; g++) {
        int32x4_t a = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(q + 32 * g), vld1q_s8(y->qs + 32 * g)), vld1q_s8(q + 32 * g + 16), vld1q_s8(y->qs + 32 * g + 16));
        acc = vmlaq_n_s32(acc, a, sc[g]);
      }
      sum[b] += y->d * (d * (float)vaddvq_s32(acc) - dm * (float)mins_dot(mn, y->bsums));
    }
  }
}
static void dotb_q6_K(int nblk, const b_q6_K *x, const cl_xq *xq, int nb, float *sum) {
  for (int b = 0; b < nb; b++) sum[b] = 0;
  for (int i = 0; i < nblk; i++) {
    int8_t q[256];
    unpack_q6_K(&x[i], q);
    const float d = cl_f16_to_f32(x[i].d);
    for (int b = 0; b < nb; b++) {
      const cl_q8k *y = &xq[b].b[i];
      int32x4_t acc = vdupq_n_s32(0);
      for (int j = 0; j < 16; j++) acc = vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), vld1q_s8(q + 16 * j), vld1q_s8(y->qs + 16 * j)), x[i].scales[j]);
      sum[b] += d * y->d * (float)vaddvq_s32(acc);
    }
  }
}
#endif

/* ---- thread pool: persistent workers claim chunks of a batch; the caller works too ---- */
#include <stdatomic.h>
#define CHUNK 16
static int g_threads = 0;
void cl_set_threads(int n) { g_threads = n < 1 ? 1 : n > 16 ? 16 : n; }
int cl_get_threads(void) {
  if (!g_threads) { cl_set_threads(cl_ncpu()); }
  return g_threads;
}

typedef struct { const cl_mv *jobs; int n; int *first; int total; } batch_t;   /* first[j] = index of job j's first chunk */
static pthread_mutex_t pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pool_cv = PTHREAD_COND_INITIALIZER, pool_done = PTHREAD_COND_INITIALIZER;
static _Atomic(const batch_t *) cur_batch;
static atomic_int next_chunk, pending_workers, generation;
static int pool_started, pool_size;
#define SPIN 20000   /* a worker polls this many times for the next batch before it sleeps; a batch arrives within microseconds of the last */

/* CLUTOR_FLOAT_MATH=1 skips the 8-bit activations and multiplies dequantized weights in float: slower, and the reference the integer kernels are measured against. */
static int float_math(void) {
  static int v = -1;
  if (v < 0) { const char *e = getenv("CLUTOR_FLOAT_MATH"); v = e && *e == '1'; }
  return v;
}

static void do_rows(const cl_mv *j, int r0, int r1) {
  uint32_t t = j->type;
  const cl_xq *xq = float_math() ? NULL : j->xq;
  const int nb = j->nb > 1 ? (j->nb > 64 ? 64 : j->nb) : 1;
  uint64_t rb = cl_row_bytes(t, (uint64_t)j->cols);
  const uint8_t *w = j->w;
  float *out = j->out;
  const size_t os = j->out_stride;
#ifdef CL_Q_NEON
  if (nb > 1 && xq && (t == CL_T_Q4_K || t == CL_T_Q5_K || t == CL_T_Q6_K)) {
    float sums[64];
    for (int r = r0; r < r1; r++) {
      const void *row = w + (size_t)r * rb;
      if (t == CL_T_Q4_K) dotb_q4_K(j->cols / 256, (const b_q4_K *)row, xq, nb, sums);
      else if (t == CL_T_Q5_K) dotb_q5_K(j->cols / 256, (const b_q5_K *)row, xq, nb, sums);
      else dotb_q6_K(j->cols / 256, (const b_q6_K *)row, xq, nb, sums);
      for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = sums[b];
    }
    return;
  }
#endif
  if (t == CL_T_Q4_K && xq) for (int r = r0; r < r1; r++) { const b_q4_K *row = (const b_q4_K *)(const void *)(w + (size_t)r * rb); for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = dot_q4_K(j->cols / 256, row, xq[b].b); }
  else if (t == CL_T_Q5_K && xq) for (int r = r0; r < r1; r++) { const b_q5_K *row = (const b_q5_K *)(const void *)(w + (size_t)r * rb); for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = dot_q5_K(j->cols / 256, row, xq[b].b); }
  else if (t == CL_T_Q6_K && xq) for (int r = r0; r < r1; r++) { const b_q6_K *row = (const b_q6_K *)(const void *)(w + (size_t)r * rb); for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = dot_q6_K(j->cols / 256, row, xq[b].b); }
  else {   /* any other type (or exact float math): dequantize a block at a time and use the float vectors */
    uint32_t bl = cl_type_block(t), bb = cl_type_bytes(t);
    float buf[256], acc[64];
    for (int r = r0; r < r1; r++) {
      const uint8_t *row = w + (size_t)r * rb;
      for (int b = 0; b < nb; b++) acc[b] = 0;
      if (t == CL_T_F32) { const float *wf = (const float *)(const void *)row; for (int b = 0; b < nb; b++) { const float *x = j->xf + (size_t)b * j->xf_stride; float s = 0; for (int i = 0; i < j->cols; i++) s += wf[i] * x[i]; acc[b] = s; } }
      else for (int c = 0; c < j->cols; c += (int)bl) {
        cl_dequant_row(t, row + (size_t)(c / (int)bl) * bb, buf, (int)bl);
        for (int b = 0; b < nb; b++) { const float *x = j->xf + (size_t)b * j->xf_stride + c; float s = 0; for (uint32_t i = 0; i < bl; i++) s += buf[i] * x[i]; acc[b] += s; }
      }
      for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = acc[b];
    }
  }
}

static void run_chunks(const batch_t *b) {
  for (;;) {
    int c = atomic_fetch_add(&next_chunk, 1);
    if (c >= b->total) return;
    int j = 0;
    while (j + 1 < b->n && b->first[j + 1] <= c) j++;
    int r0 = (c - b->first[j]) * CHUNK, r1 = r0 + CHUNK;
    if (r1 > b->jobs[j].rows) r1 = b->jobs[j].rows;
    do_rows(&b->jobs[j], r0, r1);
  }
}

static void *pool_worker(void *arg) {
  (void)arg;
  int seen = 0;   /* the pool is started before the first batch, so every worker begins at generation 0 */
  for (;;) {
    int g = atomic_load(&generation);
    for (int i = 0; i < SPIN && g == seen; i++) g = atomic_load(&generation);
    if (g == seen) {
      pthread_mutex_lock(&pool_mu);
      while ((g = atomic_load(&generation)) == seen) pthread_cond_wait(&pool_cv, &pool_mu);
      pthread_mutex_unlock(&pool_mu);
    }
    seen = g;
    run_chunks(atomic_load(&cur_batch));
    if (atomic_fetch_sub(&pending_workers, 1) == 1) { pthread_mutex_lock(&pool_mu); pthread_cond_signal(&pool_done); pthread_mutex_unlock(&pool_mu); }
  }
  return NULL;
}

void cl_matvec_batch(const cl_mv *jobs, int n) {
  int first[64], total = 0;
  if (n > 64) n = 64;
  for (int j = 0; j < n; j++) { first[j] = total; total += (jobs[j].rows + CHUNK - 1) / CHUNK; }
  batch_t b = {jobs, n, first, total};
  int nt = cl_get_threads();
  if (nt == 1 || total < 4) { atomic_store(&next_chunk, 0); run_chunks(&b); return; }
  if (!pool_started) {
    pool_started = 1; pool_size = nt - 1;
    for (int i = 0; i < pool_size; i++) { pthread_t t; if (pthread_create(&t, NULL, pool_worker, NULL) == 0) pthread_detach(t); else { pool_size = i; break; } }
  }
  atomic_store(&next_chunk, 0);
  atomic_store(&cur_batch, &b);
  atomic_store(&pending_workers, pool_size);
  pthread_mutex_lock(&pool_mu);
  atomic_fetch_add(&generation, 1);
  pthread_cond_broadcast(&pool_cv);
  pthread_mutex_unlock(&pool_mu);
  run_chunks(&b);
  for (int i = 0; i < SPIN && atomic_load(&pending_workers) > 0; i++) { }
  if (atomic_load(&pending_workers) > 0) {
    pthread_mutex_lock(&pool_mu);
    while (atomic_load(&pending_workers) > 0) pthread_cond_wait(&pool_done, &pool_mu);
    pthread_mutex_unlock(&pool_mu);
  }
}

void cl_matvec(uint32_t type, const void *w, int rows, int cols, const float *x, float *out) {
  cl_xq q = {0};
  int use_q = cl_type_integer_dot(type) && cl_xq_init(&q, cols);
  if (use_q) cl_xq_quantize(&q, x);
  cl_mv j = {type, w, rows, cols, x, use_q ? &q : NULL, out, 1, 0, 0};
  if (cl_type_integer_dot(type) && !use_q) { /* odd width: cannot happen for real K-quant tensors */ return; }
  cl_matvec_batch(&j, 1);
  cl_xq_free(&q);
}
