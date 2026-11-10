#define _GNU_SOURCE
#include "quant.h"
#include "kernel.h"
#include "iq.h"
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
typedef struct { uint8_t scales[QK_K / 16]; uint8_t qs[QK_K / 4]; uint16_t d, dmin; } b_q2_K;   /* 2-bit values, 16 groups of 16 each with a 4-bit scale and a 4-bit minimum */
typedef struct { uint8_t hmask[QK_K / 8]; uint8_t qs[QK_K / 4]; uint8_t scales[12]; uint16_t d; } b_q3_K;   /* 3-bit values (2 low bits + one bit of a mask, set means "not negative"), 16 six-bit scales */
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } b_q4_K;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[QK_K / 8]; uint8_t qs[QK_K / 2]; } b_q5_K;
typedef struct { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; uint16_t d; } b_q6_K;
_Static_assert(sizeof(b_q2_K) == 84 && sizeof(b_q3_K) == 110 && sizeof(b_q4_K) == 144 && sizeof(b_q5_K) == 176 && sizeof(b_q6_K) == 210 && sizeof(b_q5_0) == 22 && sizeof(b_q8_0) == 34, "block sizes");

int cl_type_supported(uint32_t t) { return cl_type_block(t) != 0; }
uint32_t cl_type_block(uint32_t t) {
  switch (t) { case CL_T_F32: case CL_T_F16: case CL_T_BF16: return 1; case CL_T_Q4_0: case CL_T_Q5_0: case CL_T_Q8_0: return 32; case CL_T_Q2_K: case CL_T_Q3_K: case CL_T_Q4_K: case CL_T_Q5_K: case CL_T_Q6_K: return 256; default: return cl_iq_block(t); }
}
uint32_t cl_type_bytes(uint32_t t) {
  switch (t) { case CL_T_F32: return 4; case CL_T_F16: case CL_T_BF16: return 2; case CL_T_Q4_0: return 18; case CL_T_Q5_0: return 22; case CL_T_Q8_0: return 34; case CL_T_Q2_K: return 84; case CL_T_Q3_K: return 110; case CL_T_Q4_K: return 144; case CL_T_Q5_K: return 176; case CL_T_Q6_K: return 210; default: return cl_iq_bytes(t); }
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

/* Q2_K / Q3_K. Element e of a block (0..255) sits in group g = e / 16 (a scale each); within the block, 128-element halves n hold four 32-element runs j, run j using bits 2j and 2j+1 of 32 bytes. */
static inline int q2_val(const b_q2_K *x, int g, int l) { const int n = g / 8, j = (g % 8) / 2, half = g % 2; return (x->qs[32 * n + 16 * half + l] >> (2 * j)) & 3; }
static inline int q3_val(const b_q3_K *x, int g, int l) {
  const int n = g / 8, j = (g % 8) / 2, half = g % 2;
  const int low = (x->qs[32 * n + 16 * half + l] >> (2 * j)) & 3;
  return low - ((x->hmask[16 * half + l] & (1u << (4 * n + j))) ? 0 : 4);
}
/* The sixteen 6-bit scales of a Q3_K block, minus 32 (ggml packs them into 12 bytes). */
static inline void q3_scales(const uint8_t *s, int8_t *out) {
  uint32_t a[4];
  memcpy(a, s, 12);
  const uint32_t k1 = 0x03030303u, k2 = 0x0f0f0f0fu, tmp = a[2];
  a[2] = ((a[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
  a[3] = ((a[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
  a[0] = (a[0] & k2) | (((tmp >> 0) & k1) << 4);
  a[1] = (a[1] & k2) | (((tmp >> 2) & k1) << 4);
  uint8_t raw[16];
  memcpy(raw, a, 16);
  for (int i = 0; i < 16; i++) out[i] = (int8_t)((int)raw[i] - 32);
}
static void deq_q2_K(const b_q2_K *x, float *y) {
  const float d = cl_f16_to_f32(x->d), mn = cl_f16_to_f32(x->dmin);
  for (int g = 0; g < 16; g++) { const float dl = d * (float)(x->scales[g] & 0xF), ml = mn * (float)(x->scales[g] >> 4); for (int l = 0; l < 16; l++) y[16 * g + l] = dl * (float)q2_val(x, g, l) - ml; }
}
static void deq_q3_K(const b_q3_K *x, float *y) {
  const float d = cl_f16_to_f32(x->d);
  int8_t sc[16];
  q3_scales(x->scales, sc);
  for (int g = 0; g < 16; g++) for (int l = 0; l < 16; l++) y[16 * g + l] = d * (float)sc[g] * (float)q3_val(x, g, l);
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
    case CL_T_Q2_K: for (int b = 0; b < n / QK_K; b++) deq_q2_K((const b_q2_K *)(const void *)(p + (size_t)b * 84), dst + b * QK_K); return;
    case CL_T_Q3_K: for (int b = 0; b < n / QK_K; b++) deq_q3_K((const b_q3_K *)(const void *)(p + (size_t)b * 110), dst + b * QK_K); return;
    case CL_T_Q4_K: for (int b = 0; b < n / QK_K; b++) deq_q4_K((const b_q4_K *)(const void *)(p + (size_t)b * 144), dst + b * QK_K); return;
    case CL_T_Q5_K: for (int b = 0; b < n / QK_K; b++) deq_q5_K((const b_q5_K *)(const void *)(p + (size_t)b * 176), dst + b * QK_K); return;
    case CL_T_Q6_K: for (int b = 0; b < n / QK_K; b++) deq_q6_K((const b_q6_K *)(const void *)(p + (size_t)b * 210), dst + b * QK_K); return;
    default:
      if (cl_iq_is(type)) { const uint32_t bl = cl_iq_block(type), bb = cl_iq_bytes(type); for (int b = 0; b < n / (int)bl; b++) cl_iq_dequant(type, p + (size_t)b * bb, dst + (size_t)b * bl); return; }
      memset(dst, 0, (size_t)n * 4);
  }
}

/* ---- integer dot products: K-quant weights against 8-bit activations (the maths of ggml's vec_dot for these types) ---- */
#if !defined(CL_FORCE_SCALAR) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define CL_Q_NEON 1
#elif !defined(CL_FORCE_SCALAR) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
/* x86-64: the AVX2 and AVX-512 VNNI kernels are compiled with per-function target attributes, so one binary runs on any x86-64 processor
 * and picks the best kernels at start-up from what the processor reports (CLUTOR_ISA=scalar|avx2 forces a lower one, for measuring and tests). */
#include <immintrin.h>
#include <cpuid.h>
#define CL_X86 1
#endif
static int isa_level(void);
const char *cl_quant_path(void) {
#ifdef CL_Q_NEON
  return "neon+dotprod";
#elif defined(CL_X86)
  return isa_level() >= 2 ? "avx512vnni" : isa_level() == 1 ? "avx2" : "scalar";
#else
  return "scalar";
#endif
}
int cl_type_integer_dot(uint32_t t) { return (cl_iq_is(t) && t != CL_T_IQ4_NL) || t == CL_T_Q2_K || t == CL_T_Q3_K || t == CL_T_Q4_K || t == CL_T_Q5_K || t == CL_T_Q6_K; }

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

#if !defined(CL_Q_NEON)
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
#if defined(CL_Q_NEON)
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
/* Q2_K / Q3_K, batched: a block's 256 weights are unpacked to int8 once and multiplied with every token of the tile. */
static inline void unpack_q2_K(const b_q2_K *x, int8_t *q) {
  const uint8x16_t m3 = vdupq_n_u8(3);
  for (int n = 0; n < 2; n++) {
    const uint8x16_t q0 = vld1q_u8(x->qs + 32 * n), q1 = vld1q_u8(x->qs + 32 * n + 16);
    for (int j = 0; j < 4; j++) {
      const int8x16_t sh = vdupq_n_s8((int8_t)(-2 * j));
      vst1q_s8(q + 128 * n + 32 * j, vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q0, sh), m3)));
      vst1q_s8(q + 128 * n + 32 * j + 16, vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q1, sh), m3)));
    }
  }
}
static inline void unpack_q3_K(const b_q3_K *x, int8_t *q) {
  const uint8x16_t m3 = vdupq_n_u8(3), one = vdupq_n_u8(1);
  const int8x16_t four = vdupq_n_s8(4);
  const uint8x16_t h0 = vld1q_u8(x->hmask), h1 = vld1q_u8(x->hmask + 16);
  for (int n = 0; n < 2; n++) {
    const uint8x16_t q0 = vld1q_u8(x->qs + 32 * n), q1 = vld1q_u8(x->qs + 32 * n + 16);
    for (int j = 0; j < 4; j++) {
      const int8x16_t sh = vdupq_n_s8((int8_t)(-2 * j)), shb = vdupq_n_s8((int8_t)(-(4 * n + j)));
      const uint8x16_t b0 = vshlq_n_u8(vandq_u8(vshlq_u8(h0, shb), one), 2), b1 = vshlq_n_u8(vandq_u8(vshlq_u8(h1, shb), one), 2);
      vst1q_s8(q + 128 * n + 32 * j, vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(vandq_u8(vshlq_u8(q0, sh), m3), b0)), four));
      vst1q_s8(q + 128 * n + 32 * j + 16, vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(vandq_u8(vshlq_u8(q1, sh), m3), b1)), four));
    }
  }
}
static void dotb_q2_K(int nblk, const b_q2_K *x, const cl_xq *xq, int nb, float *sum) {
  for (int b = 0; b < nb; b++) sum[b] = 0;
  for (int i = 0; i < nblk; i++) {
    int8_t q[256];
    unpack_q2_K(&x[i], q);
    uint8_t sc[16], mn[16];
    for (int g = 0; g < 16; g++) { sc[g] = x[i].scales[g] & 0xF; mn[g] = x[i].scales[g] >> 4; }
    const float d = cl_f16_to_f32(x[i].d), dm = cl_f16_to_f32(x[i].dmin);
    for (int b = 0; b < nb; b++) {
      const cl_q8k *y = &xq[b].b[i];
      int32x4_t acc = vdupq_n_s32(0);
      for (int g = 0; g < 16; g++) acc = vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), vld1q_s8(q + 16 * g), vld1q_s8(y->qs + 16 * g)), sc[g]);
      int32_t summs = 0;
      for (int g = 0; g < 16; g++) summs += mn[g] * y->bsums[g];
      sum[b] += y->d * (d * (float)vaddvq_s32(acc) - dm * (float)summs);
    }
  }
}
static void dotb_q3_K(int nblk, const b_q3_K *x, const cl_xq *xq, int nb, float *sum) {
  for (int b = 0; b < nb; b++) sum[b] = 0;
  for (int i = 0; i < nblk; i++) {
    int8_t q[256], sc[16];
    unpack_q3_K(&x[i], q);
    q3_scales(x[i].scales, sc);
    const float d = cl_f16_to_f32(x[i].d);
    for (int b = 0; b < nb; b++) {
      const cl_q8k *y = &xq[b].b[i];
      int32x4_t acc = vdupq_n_s32(0);
      for (int g = 0; g < 16; g++) acc = vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), vld1q_s8(q + 16 * g), vld1q_s8(y->qs + 16 * g)), sc[g]);
      sum[b] += d * y->d * (float)vaddvq_s32(acc);
    }
  }
}
#endif

/* The integer dot products with a block of Q8_K activations (ggml's vec_dot_q2_K_q8_K and vec_dot_q3_K_q8_K): exact integer sums per group, then one scaling per block. */
static float dot_q2_K(int nb, const b_q2_K *x, const cl_q8k *y) {
  float sum = 0;
  for (int i = 0; i < nb; i++) {
    int32_t isum = 0, summs = 0;
#ifdef CL_Q_NEON
    {   /* two 128-value halves; in each, run j takes bits 2j and 2j+1 of 32 bytes, as two 16-value groups with a scale each */
      const uint8x16_t m3 = vdupq_n_u8(3);
      int32x4_t acc = vdupq_n_s32(0);
      for (int n = 0; n < 2; n++) {
        const uint8x16_t q0 = vld1q_u8(x[i].qs + 32 * n), q1 = vld1q_u8(x[i].qs + 32 * n + 16);
        for (int j = 0; j < 4; j++) {
          const int g0 = 8 * n + 2 * j, g1 = g0 + 1;
          const int8x16_t v0 = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q0, vdupq_n_s8((int8_t)(-2 * j))), m3)), v1 = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q1, vdupq_n_s8((int8_t)(-2 * j))), m3));
          acc = vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), v0, vld1q_s8(y[i].qs + 16 * g0)), x[i].scales[g0] & 0xF);
          acc = vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), v1, vld1q_s8(y[i].qs + 16 * g1)), x[i].scales[g1] & 0xF);
        }
      }
      isum = vaddvq_s32(acc);
      for (int g = 0; g < 16; g++) summs += (x[i].scales[g] >> 4) * y[i].bsums[g];
    }
#else
    for (int g = 0; g < 16; g++) {
      int32_t a = 0;
      const int8_t *q8 = y[i].qs + 16 * g;
      const int n = g / 8, j = (g % 8) / 2, half = g % 2, sh = 2 * j;
      const uint8_t *q = x[i].qs + 32 * n + 16 * half;
      for (int l = 0; l < 16; l++) a += q8[l] * ((q[l] >> sh) & 3);
      isum += (x[i].scales[g] & 0xF) * a;
      summs += (x[i].scales[g] >> 4) * y[i].bsums[g];
    }
#endif
    sum += y[i].d * (cl_f16_to_f32(x[i].d) * (float)isum - cl_f16_to_f32(x[i].dmin) * (float)summs);
  }
  return sum;
}
static float dot_q3_K(int nb, const b_q3_K *x, const cl_q8k *y) {
  float sum = 0;
  for (int i = 0; i < nb; i++) {
    int8_t sc[16];
    q3_scales(x[i].scales, sc);
    int32_t isum = 0;
#ifdef CL_Q_NEON
    {   /* value = low two bits + 4 * (mask bit) - 4, so a group's dot product is dot(low + 4 * bit, q8) - 4 * (sum of its q8), and the sums are already known */
      const uint8x16_t m3 = vdupq_n_u8(3), one = vdupq_n_u8(1);
      const uint8x16_t h0 = vld1q_u8(x[i].hmask), h1 = vld1q_u8(x[i].hmask + 16);
      int32x4_t acc = vdupq_n_s32(0);
      for (int n = 0; n < 2; n++) {
        const uint8x16_t q0 = vld1q_u8(x[i].qs + 32 * n), q1 = vld1q_u8(x[i].qs + 32 * n + 16);
        for (int j = 0; j < 4; j++) {
          const int g0 = 8 * n + 2 * j, g1 = g0 + 1, bit = 4 * n + j;
          const int8x16_t sh2 = vdupq_n_s8((int8_t)(-2 * j)), shb = vdupq_n_s8((int8_t)(-bit));
          const uint8x16_t b0 = vshlq_n_u8(vandq_u8(vshlq_u8(h0, shb), one), 2), b1 = vshlq_n_u8(vandq_u8(vshlq_u8(h1, shb), one), 2);
          const int8x16_t v0 = vreinterpretq_s8_u8(vaddq_u8(vandq_u8(vshlq_u8(q0, sh2), m3), b0)), v1 = vreinterpretq_s8_u8(vaddq_u8(vandq_u8(vshlq_u8(q1, sh2), m3), b1));
          acc = vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), v0, vld1q_s8(y[i].qs + 16 * g0)), sc[g0]);
          acc = vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), v1, vld1q_s8(y[i].qs + 16 * g1)), sc[g1]);
        }
      }
      isum = vaddvq_s32(acc);
      for (int g = 0; g < 16; g++) isum -= 4 * sc[g] * y[i].bsums[g];
    }
#else
    for (int g = 0; g < 16; g++) {
      int32_t a = 0;
      const int8_t *q8 = y[i].qs + 16 * g;
      const int n = g / 8, j = (g % 8) / 2, half = g % 2, sh = 2 * j;
      const uint8_t *q = x[i].qs + 32 * n + 16 * half, *hm = x[i].hmask + 16 * half;
      const uint32_t bit = 1u << (4 * n + j);
      for (int l = 0; l < 16; l++) a += q8[l] * (((q[l] >> sh) & 3) - ((hm[l] & bit) ? 0 : 4));
      isum += sc[g] * a;
    }
#endif
    sum += y[i].d * cl_f16_to_f32(x[i].d) * (float)isum;
  }
  return sum;
}

/* ---- thread pool: persistent workers claim chunks of a batch; the caller works too ---- */
#include <stdatomic.h>
#define CHUNK 16
#ifndef CL_TILE
#define CL_TILE 8   /* tokens per tile in a batched product; measured: 8 is as fast as any, and far faster than 32 or 64 */
#endif
static int g_threads = 0;
void cl_set_threads(int n) { g_threads = n < 1 ? 1 : n > 16 ? 16 : n; }
int cl_get_threads(void) {
  if (!g_threads) { const char *e = getenv("CLUTOR_THREADS"); cl_set_threads(e && atoi(e) > 0 ? atoi(e) : cl_ncpu()); }   /* CLUTOR_THREADS=n: for measuring how many cores help */
  return g_threads;
}

typedef struct { const cl_mv *jobs; int n; int *first; int total; void (*fn)(void *, int); void *ctx; } batch_t;   /* first[j] = index of job j's first chunk; with fn set, chunk c is fn(ctx, c) */
static pthread_mutex_t pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pool_cv = PTHREAD_COND_INITIALIZER, pool_done = PTHREAD_COND_INITIALIZER;
static _Atomic(const batch_t *) cur_batch;
static atomic_int next_chunk, pending_workers, generation;
static int pool_started, pool_size;
#define SPIN 20000   /* a worker polls this many times for the next batch before it sleeps; a batch arrives within microseconds of the last */

/* CLUTOR_FLOAT_MATH=1 skips the 8-bit activations and multiplies dequantized weights in float: slower, and the reference the integer kernels are measured against. */
static int float_math(void) {
  static atomic_int v = -1;   /* read by every pool thread: atomic, so the first reads racing to set it are well defined */
  int x = atomic_load_explicit(&v, memory_order_relaxed);
  if (x < 0) { const char *e = getenv("CLUTOR_FLOAT_MATH"); x = e && *e == '1'; atomic_store_explicit(&v, x, memory_order_relaxed); }
  return x;
}

/* ---- the instruction set in use (x86-64): picked once from what the processor reports ---- */
#ifdef CL_X86
static atomic_int g_isa = -1;   /* 0 portable, 1 AVX2, 2 AVX-512 VNNI */
static int isa_detect(void) {   /* cpuid directly (no compiler runtime needed), and the operating system must save the wide registers too */
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d) || !((c >> 27) & 1) || !((c >> 28) & 1)) return 0;
  unsigned xlo, xhi;
  __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
  if ((xlo & 6) != 6 || __get_cpuid_max(0, 0) < 7) return 0;
  __cpuid_count(7, 0, a, b, c, d);
  if (!((b >> 5) & 1)) return 0;   /* AVX2 */
  int v = 1;
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 9)
  if ((xlo & 0xe0) == 0xe0 && ((b >> 16) & 1) && ((b >> 31) & 1) && ((c >> 11) & 1)) v = 2;   /* AVX-512 F and VL, VNNI, with the 512-bit state enabled */
#endif
  return v;
}
static int isa_max(void) {   /* read by every pool thread: atomic, so the first reads racing to set it are well defined */
  static atomic_int m = -1;
  int v = atomic_load_explicit(&m, memory_order_relaxed);
  if (v < 0) { v = isa_detect(); atomic_store_explicit(&m, v, memory_order_relaxed); }
  return v;
}
static int isa_level(void) {
  int v = atomic_load_explicit(&g_isa, memory_order_relaxed);
  if (v < 0) {
    v = isa_max();
    const char *e = getenv("CLUTOR_ISA");
    if (e && !strcmp(e, "scalar")) v = 0;
    else if (e && !strcmp(e, "avx2") && v > 1) v = 1;
    atomic_store_explicit(&g_isa, v, memory_order_relaxed);
  }
  return v;
}
int cl_quant_set_isa(const char *name) {
  int want = !strcmp(name, "scalar") ? 0 : !strcmp(name, "avx2") ? 1 : !strcmp(name, "avx512vnni") ? 2 : !strcmp(name, "auto") ? isa_max() : -1;
  if (want < 0 || want > isa_max()) return 0;
  atomic_store_explicit(&g_isa, want, memory_order_relaxed);
  return 1;
}
#else
static int isa_level(void) __attribute__((unused));
static int isa_level(void) { return 0; }
int cl_quant_set_isa(const char *name) { (void)name; return 0; }
#endif

#ifndef CL_Q_NEON
/* ---- K-quant products without NEON (x86 and portable C), in the same shape for every type: a block's 256 weights are unpacked to int8 once, a per-16-group scale
 * vector is made, and each activation block is multiplied with them by dot256. The integer sums are exact in any order, so every path gives the same bits. ---- */
static void upk_q2(const b_q2_K *x, int8_t *q) { for (int n = 0; n < 2; n++) for (int j = 0; j < 4; j++) for (int l = 0; l < 32; l++) q[128 * n + 32 * j + l] = (int8_t)((x->qs[32 * n + l] >> (2 * j)) & 3); }
static void upk_q3(const b_q3_K *x, int8_t *q) {
  for (int n = 0; n < 2; n++) for (int j = 0; j < 4; j++) for (int l = 0; l < 32; l++)
    q[128 * n + 32 * j + l] = (int8_t)(((x->qs[32 * n + l] >> (2 * j)) & 3) + 4 * ((x->hmask[l] >> (4 * n + j)) & 1) - 4);
}
static void upk_q4(const b_q4_K *x, int8_t *q) { for (int g = 0; g < 4; g++) for (int l = 0; l < 32; l++) { q[64 * g + l] = (int8_t)(x->qs[32 * g + l] & 0xF); q[64 * g + 32 + l] = (int8_t)(x->qs[32 * g + l] >> 4); } }
static void upk_q5(const b_q5_K *x, int8_t *q) {
  for (int c = 0; c < 4; c++) for (int l = 0; l < 32; l++) {
    q[64 * c + l] = (int8_t)((x->qs[32 * c + l] & 0xF) + ((x->qh[l] >> (2 * c)) & 1) * 16);
    q[64 * c + 32 + l] = (int8_t)((x->qs[32 * c + l] >> 4) + ((x->qh[l] >> (2 * c + 1)) & 1) * 16);
  }
}
static void upk_q6(const b_q6_K *x, int8_t *q) {
  const uint8_t *ql = x->ql, *qh = x->qh;
  for (int n = 0; n < 256; n += 128) {
    for (int l = 0; l < 32; l++) {
      q[n + l] = (int8_t)(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
      q[n + l + 32] = (int8_t)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
      q[n + l + 64] = (int8_t)(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
      q[n + l + 96] = (int8_t)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
    }
    ql += 64; qh += 32;
  }
}
#endif /* !CL_Q_NEON (the K-quant unpackers above are for the other instruction sets; NEON has its own) */
__attribute__((unused)) static int32_t dot256_s(const int8_t *q, const int8_t *a, const int16_t *sc) {
  int32_t s = 0;
  for (int g = 0; g < 16; g++) { int32_t t = 0; for (int i = 0; i < 16; i++) t += q[16 * g + i] * a[16 * g + i]; s += sc[g] * t; }
  return s;
}
#ifdef CL_Q_NEON
static int32_t dot256_neon(const int8_t *q, const int8_t *a, const int16_t *sc) {
  int32x4_t acc = vdupq_n_s32(0);
  for (int g = 0; g < 16; g++) acc = vmlaq_n_s32(acc, vdotq_s32(vdupq_n_s32(0), vld1q_s8(q + 16 * g), vld1q_s8(a + 16 * g)), sc[g]);
  return vaddvq_s32(acc);
}
#endif
#ifndef CL_Q_NEON

#ifdef CL_X86
#define TGT_AVX2 __attribute__((target("avx2")))
static TGT_AVX2 inline __m256i srl8(__m256i v, int s) { return _mm256_srl_epi16(v, _mm_cvtsi32_si128(s)); }   /* 16-bit shift by a variable count; callers mask the bits that cross a byte */
static TGT_AVX2 void upk_q2_avx2(const b_q2_K *x, int8_t *q) {
  const __m256i m3 = _mm256_set1_epi8(3);
  for (int n = 0; n < 2; n++) {
    __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(x->qs + 32 * n));
    for (int j = 0; j < 4; j++) _mm256_storeu_si256((__m256i *)(void *)(q + 128 * n + 32 * j), _mm256_and_si256(srl8(v, 2 * j), m3));
  }
}
static TGT_AVX2 void upk_q3_avx2(const b_q3_K *x, int8_t *q) {
  const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1), four = _mm256_set1_epi8(4);
  const __m256i h = _mm256_loadu_si256((const __m256i *)(const void *)x->hmask);
  for (int n = 0; n < 2; n++) {
    __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(x->qs + 32 * n));
    for (int j = 0; j < 4; j++) {
      __m256i lo = _mm256_and_si256(srl8(v, 2 * j), m3), hb = _mm256_slli_epi16(_mm256_and_si256(srl8(h, 4 * n + j), one), 2);
      _mm256_storeu_si256((__m256i *)(void *)(q + 128 * n + 32 * j), _mm256_sub_epi8(_mm256_add_epi8(lo, hb), four));
    }
  }
}
static TGT_AVX2 void upk_q4_avx2(const b_q4_K *x, int8_t *q) {
  const __m256i m4 = _mm256_set1_epi8(0x0f);
  for (int g = 0; g < 4; g++) {
    __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(x->qs + 32 * g));
    _mm256_storeu_si256((__m256i *)(void *)(q + 64 * g), _mm256_and_si256(v, m4));
    _mm256_storeu_si256((__m256i *)(void *)(q + 64 * g + 32), _mm256_and_si256(srl8(v, 4), m4));
  }
}
static TGT_AVX2 void upk_q5_avx2(const b_q5_K *x, int8_t *q) {
  const __m256i m4 = _mm256_set1_epi8(0x0f), one = _mm256_set1_epi8(1);
  const __m256i qh = _mm256_loadu_si256((const __m256i *)(const void *)x->qh);
  for (int c = 0; c < 4; c++) {
    __m256i a = _mm256_loadu_si256((const __m256i *)(const void *)(x->qs + 32 * c));
    __m256i lo = _mm256_or_si256(_mm256_and_si256(a, m4), _mm256_slli_epi16(_mm256_and_si256(srl8(qh, 2 * c), one), 4));
    __m256i hi = _mm256_or_si256(_mm256_and_si256(srl8(a, 4), m4), _mm256_slli_epi16(_mm256_and_si256(srl8(qh, 2 * c + 1), one), 4));
    _mm256_storeu_si256((__m256i *)(void *)(q + 64 * c), lo);
    _mm256_storeu_si256((__m256i *)(void *)(q + 64 * c + 32), hi);
  }
}
static TGT_AVX2 void upk_q6_avx2(const b_q6_K *x, int8_t *q) {
  const __m256i m4 = _mm256_set1_epi8(0x0f), m3 = _mm256_set1_epi8(3), off = _mm256_set1_epi8(32);
  for (int n = 0; n < 2; n++) {
    const uint8_t *ql = x->ql + 64 * n, *qh = x->qh + 32 * n;
    __m256i l0 = _mm256_loadu_si256((const __m256i *)(const void *)ql), l1 = _mm256_loadu_si256((const __m256i *)(const void *)(ql + 32)), h = _mm256_loadu_si256((const __m256i *)(const void *)qh);
    __m256i v0 = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_slli_epi16(_mm256_and_si256(h, m3), 4));
    __m256i v1 = _mm256_or_si256(_mm256_and_si256(l1, m4), _mm256_slli_epi16(_mm256_and_si256(srl8(h, 2), m3), 4));
    __m256i v2 = _mm256_or_si256(_mm256_and_si256(srl8(l0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(srl8(h, 4), m3), 4));
    __m256i v3 = _mm256_or_si256(_mm256_and_si256(srl8(l1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(srl8(h, 6), m3), 4));
    int8_t *o = q + 128 * n;
    _mm256_storeu_si256((__m256i *)(void *)o, _mm256_sub_epi8(v0, off)); _mm256_storeu_si256((__m256i *)(void *)(o + 32), _mm256_sub_epi8(v1, off));
    _mm256_storeu_si256((__m256i *)(void *)(o + 64), _mm256_sub_epi8(v2, off)); _mm256_storeu_si256((__m256i *)(void *)(o + 96), _mm256_sub_epi8(v3, off));
  }
}
static TGT_AVX2 inline int32_t hsum8(__m256i acc) {
  __m128i t = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
  t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(t);
}
/* The values are small (|q| <= 32), so abs(q) times the activation moved to q's sign pair-sums to at most 8128 and fits int16; the scales then widen it. */
static TGT_AVX2 int32_t dot256_avx2(const int8_t *q, const int8_t *a, const int16_t *sc) {
  __m256i acc = _mm256_setzero_si256();
  for (int j = 0; j < 16; j += 2) {
    __m256i qv = _mm256_loadu_si256((const __m256i *)(const void *)(q + 16 * j)), av = _mm256_loadu_si256((const __m256i *)(const void *)(a + 16 * j));
    __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(qv, qv), _mm256_sign_epi8(av, qv));
    __m256i scv = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16(sc[j])), _mm_set1_epi16(sc[j + 1]), 1);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, scv));
  }
  return hsum8(acc);
}
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 9)
#define CL_X86_VNNI 1
/* AVX-512 VNNI: one instruction multiplies unsigned by signed bytes and adds four products into each 32-bit lane (no 16-bit step), 16 values per group, two groups per register. */
static __attribute__((target("avx2,avx512vnni,avx512vl"))) int32_t dot256_vnni(const int8_t *q, const int8_t *a, const int16_t *sc) {
  __m256i acc = _mm256_setzero_si256();
  for (int j = 0; j < 16; j += 2) {
    __m256i qv = _mm256_loadu_si256((const __m256i *)(const void *)(q + 16 * j)), av = _mm256_loadu_si256((const __m256i *)(const void *)(a + 16 * j));
    __m256i p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(qv, qv), _mm256_sign_epi8(av, qv));
    __m256i scv = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi32(sc[j])), _mm_set1_epi32(sc[j + 1]), 1);
    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(p, scv));
  }
  return hsum8(acc);
}
#endif
#endif /* CL_X86 */
#endif /* !CL_Q_NEON */

/* One block of 256 weights as int8 values q, a scale per 16 values (sc), a minimum per 16 (mn), and the block scales d and dm (dm is 0 for types without minimums).
 * Returns 1 when the type has minimums. The IQ types come out of iq.c with ready-made scales. */
static int gblock(int isa, uint32_t t, const void *blk, int8_t *q, int16_t *sc, int16_t *mn, float *d, float *dm) {
  if (cl_iq_is(t)) { cl_iq_unpack(t, blk, q, sc, d); for (int g = 0; g < 16; g++) mn[g] = 0; *dm = 0; return 0; }
#ifndef CL_Q_NEON
#ifdef CL_X86
  if (isa > 0) {
    switch (t) {
      case CL_T_Q2_K: upk_q2_avx2((const b_q2_K *)blk, q); break;
      case CL_T_Q3_K: upk_q3_avx2((const b_q3_K *)blk, q); break;
      case CL_T_Q4_K: upk_q4_avx2((const b_q4_K *)blk, q); break;
      case CL_T_Q5_K: upk_q5_avx2((const b_q5_K *)blk, q); break;
      default: upk_q6_avx2((const b_q6_K *)blk, q); break;
    }
  } else
#endif
  switch (t) {
    case CL_T_Q2_K: upk_q2((const b_q2_K *)blk, q); break;
    case CL_T_Q3_K: upk_q3((const b_q3_K *)blk, q); break;
    case CL_T_Q4_K: upk_q4((const b_q4_K *)blk, q); break;
    case CL_T_Q5_K: upk_q5((const b_q5_K *)blk, q); break;
    default: upk_q6((const b_q6_K *)blk, q); break;
  }
#endif
  (void)isa;
  if (t == CL_T_Q2_K) {
    const b_q2_K *x = (const b_q2_K *)blk;
    for (int g = 0; g < 16; g++) { sc[g] = x->scales[g] & 0xF; mn[g] = x->scales[g] >> 4; }
    *d = cl_f16_to_f32(x->d); *dm = cl_f16_to_f32(x->dmin); return 1;
  }
  if (t == CL_T_Q3_K) {
    const b_q3_K *x = (const b_q3_K *)blk;
    int8_t s8[16]; q3_scales(x->scales, s8);
    for (int g = 0; g < 16; g++) { sc[g] = s8[g]; mn[g] = 0; }
    *d = cl_f16_to_f32(x->d); *dm = 0; return 0;
  }
  if (t == CL_T_Q6_K) {
    const b_q6_K *x = (const b_q6_K *)blk;
    for (int g = 0; g < 16; g++) { sc[g] = x->scales[g]; mn[g] = 0; }
    *d = cl_f16_to_f32(x->d); *dm = 0; return 0;
  }
  const uint8_t *scales = t == CL_T_Q4_K ? ((const b_q4_K *)blk)->scales : ((const b_q5_K *)blk)->scales;
  uint8_t s8[8], m8[8];
  unpack_scales(scales, s8, m8);
  for (int k = 0; k < 8; k++) { sc[2 * k] = sc[2 * k + 1] = s8[k]; mn[2 * k] = mn[2 * k + 1] = m8[k]; }
  *d = cl_f16_to_f32(t == CL_T_Q4_K ? ((const b_q4_K *)blk)->d : ((const b_q5_K *)blk)->d);
  *dm = cl_f16_to_f32(t == CL_T_Q4_K ? ((const b_q4_K *)blk)->dmin : ((const b_q5_K *)blk)->dmin);
  return 1;
}
static inline int32_t dot256(int isa, const int8_t *q, const int8_t *a, const int16_t *sc) {
#ifdef CL_Q_NEON
  (void)isa; return dot256_neon(q, a, sc);
#else
#ifdef CL_X86
#ifdef CL_X86_VNNI
  if (isa >= 2) return dot256_vnni(q, a, sc);
#endif
  if (isa == 1) return dot256_avx2(q, a, sc);
#else
  (void)isa;
#endif
  return dot256_s(q, a, sc);
#endif
}
/* nb inputs against one weight row of nblk blocks: sum[b] = the dot product with input b. */
static void dotb_gen(uint32_t t, int nblk, const uint8_t *row, const cl_xq *xq, int nb, float *sum) {
  const int isa = isa_level();
  const size_t bb = cl_type_bytes(t);
  for (int b = 0; b < nb; b++) sum[b] = 0;
  for (int i = 0; i < nblk; i++) {
    int8_t q[256]; int16_t sc[16], mn[16]; float d, dm;
    const int has_min = gblock(isa, t, row + (size_t)i * bb, q, sc, mn, &d, &dm);
    for (int b = 0; b < nb; b++) {
      const cl_q8k *y = &xq[b].b[i];
      const int32_t isum = dot256(isa, q, y->qs, sc);
      if (has_min) { int32_t m = 0; for (int g = 0; g < 16; g++) m += mn[g] * y->bsums[g]; sum[b] += y->d * (d * (float)isum - dm * (float)m); }
      else sum[b] += d * y->d * (float)isum;
    }
  }
}

static void do_rows(const cl_mv *j, int r0, int r1) {
  uint32_t t = j->type;
  const cl_xq *xq = float_math() ? NULL : j->xq;
  const int nb = j->nb > 1 ? (j->nb > CL_MV_MAX ? CL_MV_MAX : j->nb) : 1;   /* callers never ask for more (forward.c asserts it); the clamp only keeps a bad batch from overrunning acc[] */
  uint64_t rb = cl_row_bytes(t, (uint64_t)j->cols);
  const uint8_t *w = j->w;
  float *out = j->out;
  const size_t os = j->out_stride;
#ifdef CL_Q_NEON
  const int tiled = nb > 1 || (cl_iq_is(t) && t != CL_T_IQ4_NL);
#else
  const int tiled = nb > 1 || isa_level() > 0 || (cl_iq_is(t) && t != CL_T_IQ4_NL);   /* without NEON, a lone token takes this path too when the processor has AVX2 */
#endif
  if (tiled && xq && ((cl_iq_is(t) && t != CL_T_IQ4_NL) || t == CL_T_Q2_K || t == CL_T_Q3_K || t == CL_T_Q4_K || t == CL_T_Q5_K || t == CL_T_Q6_K)) {
    /* The batch is walked in tiles of CL_TILE tokens: a weight row is re-read for each tile (from the processor's cache, it is only a few KB), but the tile's activations stay in
     * the fastest cache, which is where a large batch used to lose most of its speed. */
    float sums[CL_TILE];
    for (int b0 = 0; b0 < nb; b0 += CL_TILE) {
      const int nt = nb - b0 < CL_TILE ? nb - b0 : CL_TILE;
      for (int r = r0; r < r1; r++) {
        const void *row = w + (size_t)r * rb;
#ifdef CL_Q_NEON
        if (cl_iq_is(t)) dotb_gen(t, j->cols / 256, (const uint8_t *)row, xq + b0, nt, sums);
        else if (t == CL_T_Q2_K) dotb_q2_K(j->cols / 256, (const b_q2_K *)row, xq + b0, nt, sums);
        else if (t == CL_T_Q3_K) dotb_q3_K(j->cols / 256, (const b_q3_K *)row, xq + b0, nt, sums);
        else if (t == CL_T_Q4_K) dotb_q4_K(j->cols / 256, (const b_q4_K *)row, xq + b0, nt, sums);
        else if (t == CL_T_Q5_K) dotb_q5_K(j->cols / 256, (const b_q5_K *)row, xq + b0, nt, sums);
        else dotb_q6_K(j->cols / 256, (const b_q6_K *)row, xq + b0, nt, sums);
#else
        dotb_gen(t, j->cols / 256, (const uint8_t *)row, xq + b0, nt, sums);
#endif
        for (int b = 0; b < nt; b++) out[(size_t)(b0 + b) * os + r] = sums[b];
      }
    }
    return;
  }
  if (t == CL_T_Q2_K && xq) for (int r = r0; r < r1; r++) { const b_q2_K *row = (const b_q2_K *)(const void *)(w + (size_t)r * rb); for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = dot_q2_K(j->cols / 256, row, xq[b].b); }
  else if (t == CL_T_Q3_K && xq) for (int r = r0; r < r1; r++) { const b_q3_K *row = (const b_q3_K *)(const void *)(w + (size_t)r * rb); for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = dot_q3_K(j->cols / 256, row, xq[b].b); }
  else if (t == CL_T_Q4_K && xq) for (int r = r0; r < r1; r++) { const b_q4_K *row = (const b_q4_K *)(const void *)(w + (size_t)r * rb); for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = dot_q4_K(j->cols / 256, row, xq[b].b); }
  else if (t == CL_T_Q5_K && xq) for (int r = r0; r < r1; r++) { const b_q5_K *row = (const b_q5_K *)(const void *)(w + (size_t)r * rb); for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = dot_q5_K(j->cols / 256, row, xq[b].b); }
  else if (t == CL_T_Q6_K && xq) for (int r = r0; r < r1; r++) { const b_q6_K *row = (const b_q6_K *)(const void *)(w + (size_t)r * rb); for (int b = 0; b < nb; b++) out[(size_t)b * os + r] = dot_q6_K(j->cols / 256, row, xq[b].b); }
  else {   /* any other type (or exact float math): dequantize a block at a time and use the float vectors */
    uint32_t bl = cl_type_block(t), bb = cl_type_bytes(t);
    float buf[256], acc[CL_MV_MAX];
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
    if (b->fn) { b->fn(b->ctx, c); continue; }
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

/* Runs the batch on the pool (the caller works too) and returns when every chunk is done. */
static void dispatch(const batch_t *bp) {
  batch_t b = *bp;
  int nt = cl_get_threads();
  if (nt == 1 || b.total < (b.fn ? 2 : 4)) { atomic_store(&next_chunk, 0); run_chunks(&b); return; }
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

void cl_matvec_batch(const cl_mv *jobs, int n) {
  int first[64], total = 0;
  if (n > 64) n = 64;
  for (int j = 0; j < n; j++) { first[j] = total; total += (jobs[j].rows + CHUNK - 1) / CHUNK; }
  batch_t b = {jobs, n, first, total, NULL, NULL};
  dispatch(&b);
}

void cl_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx) {
  if (n <= 0) return;
  batch_t b = {NULL, 0, NULL, n, fn, ctx};
  dispatch(&b);
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
