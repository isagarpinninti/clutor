#include "iq.h"
#include "kernel.h"
#include "iq_tables.h"
#include <stdatomic.h>
#include <string.h>

int cl_iq_is(uint32_t t) { return cl_iq_block(t) != 0; }
uint32_t cl_iq_block(uint32_t t) {
  switch (t) { case CL_T_IQ4_NL: return 32; case CL_T_IQ2_XXS: case CL_T_IQ2_XS: case CL_T_IQ2_S: case CL_T_IQ3_XXS: case CL_T_IQ3_S: case CL_T_IQ1_S: case CL_T_IQ1_M: case CL_T_IQ4_XS: return 256; default: return 0; }
}
uint32_t cl_iq_bytes(uint32_t t) {
  switch (t) { case CL_T_IQ2_XXS: return 66; case CL_T_IQ2_XS: return 74; case CL_T_IQ2_S: return 82; case CL_T_IQ3_XXS: return 98; case CL_T_IQ3_S: return 110;
    case CL_T_IQ1_S: return 50; case CL_T_IQ1_M: return 56; case CL_T_IQ4_NL: return 18; case CL_T_IQ4_XS: return 136; default: return 0; }
}

static const int8_t KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};   /* the non-linear 4-bit values of IQ4_NL and IQ4_XS */

#if defined(__ARM_NEON) && !defined(CL_FORCE_SCALAR)
#include <arm_neon.h>
#elif defined(__AVX2__) && !defined(CL_FORCE_SCALAR)
#include <immintrin.h>
#endif
/* 16 bytes of packed nibbles to 32 values through the 16-entry table (a byte shuffle does 16 lookups at once). */
static void kv16(const uint8_t *qs, int8_t *out) {
#if defined(__ARM_NEON) && !defined(CL_FORCE_SCALAR)
  const uint8x16_t tab = vld1q_u8((const uint8_t *)KV), v = vld1q_u8(qs);
  vst1q_u8((uint8_t *)out, vqtbl1q_u8(tab, vandq_u8(v, vdupq_n_u8(0xf))));
  vst1q_u8((uint8_t *)out + 16, vqtbl1q_u8(tab, vshrq_n_u8(v, 4)));
#elif defined(__AVX2__) && !defined(CL_FORCE_SCALAR)
  const __m128i tab = _mm_loadu_si128((const __m128i *)(const void *)KV), v = _mm_loadu_si128((const __m128i *)(const void *)qs), m = _mm_set1_epi8(0xf);
  _mm_storeu_si128((__m128i *)(void *)out, _mm_shuffle_epi8(tab, _mm_and_si128(v, m)));
  _mm_storeu_si128((__m128i *)(void *)(out + 16), _mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(v, 4), m)));
#else
  for (int j = 0; j < 16; j++) { out[j] = KV[qs[j] & 0xf]; out[j + 16] = KV[qs[j] >> 4]; }
#endif
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static float f16at(const uint8_t *p) { return cl_f16_to_f32(rd16(p)); }

/* Eight weights at a time in one 64-bit word (little-endian, as the file). The grid values are never zero (checked on the tables in the tests), so negating the bytes whose
 * sign bit is set is (v ^ mask) + (mask & 1) with no carry into the next byte. */
#define ONES 0x0101010101010101ull
static uint64_t negate_masked(uint64_t v, uint64_t m) { return (v ^ m) + (m & ONES); }
static void put8(int8_t *q, uint64_t v) { memcpy(q, &v, 8); }
/* IQ1: weight = 8 * g + dl (g is -1, 0 or 1, dl is -1 or +1), eight bytes per table entry. Both versions of the whole grid (16 KB each) are made once, on first use;
 * a lookup is then as cheap as the other formats'. The tables are filled by one thread while the others wait, and read-only afterwards. */
static uint64_t iq1_plus[2048], iq1_minus[2048];
static atomic_int iq1_state;   /* 0 not started, 1 being made, 2 ready */
static uint64_t iq1_make(uint64_t g, int dl) {
  uint64_t r = 0;
  for (int j = 0; j < 8; j++) r |= (uint64_t)(uint8_t)(int8_t)(8 * (int8_t)(uint8_t)(g >> (8 * j)) + dl) << (8 * j);
  return r;
}
static void iq1_ready(void) {
  if (atomic_load_explicit(&iq1_state, memory_order_acquire) == 2) return;
  int e = 0;
  if (atomic_compare_exchange_strong(&iq1_state, &e, 1)) {
    for (int i = 0; i < 2048; i++) { iq1_plus[i] = iq1_make(iq1s_grid[i], 1); iq1_minus[i] = iq1_make(iq1s_grid[i], -1); }
    atomic_store_explicit(&iq1_state, 2, memory_order_release);
  } else while (atomic_load_explicit(&iq1_state, memory_order_acquire) != 2) { }
}


int cl_iq_unpack(uint32_t t, const void *vb, int8_t *q, int16_t *sc, float *d) {
  const uint8_t *b = vb;
  switch (t) {
    case CL_T_IQ2_XXS: {   /* eight groups of 32: four grid indices, then four 7-bit sign codes and a 4-bit scale in one 32-bit word */
      *d = f16at(b) / 8.0f;
      for (int ib = 0; ib < 8; ib++) {
        const uint32_t a0 = rd32(b + 2 + 8 * ib), a1 = rd32(b + 2 + 8 * ib + 4);
        sc[2 * ib] = sc[2 * ib + 1] = (int16_t)(2 * (a1 >> 28) + 1);
        for (int l = 0; l < 4; l++) {
          const uint64_t g = iq2xxs_grid[(a0 >> (8 * l)) & 0xff];
          const uint64_t m = ksigns64[(a1 >> (7 * l)) & 127];
          put8(q + 32 * ib + 8 * l, negate_masked(g, m));
        }
      }
      return 1;
    }
    case CL_T_IQ2_XS: {
      *d = f16at(b) / 8.0f;
      const uint8_t *scales = b + 66;
      for (int ib = 0; ib < 8; ib++) {
        sc[2 * ib] = (int16_t)(2 * (scales[ib] & 0xf) + 1); sc[2 * ib + 1] = (int16_t)(2 * (scales[ib] >> 4) + 1);
        for (int l = 0; l < 4; l++) {
          const uint16_t v = rd16(b + 2 + 2 * (4 * ib + l));
          const uint64_t g = iq2xs_grid[v & 511];
          const uint64_t m = ksigns64[v >> 9];
          put8(q + 32 * ib + 8 * l, negate_masked(g, m));
        }
      }
      return 1;
    }
    case CL_T_IQ2_S: {
      *d = f16at(b) / 8.0f;
      const uint8_t *qs = b + 2, *signs = b + 34, *qh = b + 66, *scales = b + 74;
      for (int ib = 0; ib < 8; ib++) {
        sc[2 * ib] = (int16_t)(2 * (scales[ib] & 0xf) + 1); sc[2 * ib + 1] = (int16_t)(2 * (scales[ib] >> 4) + 1);
        for (int l = 0; l < 4; l++) {
          const uint64_t g = iq2s_grid[qs[4 * ib + l] | ((qh[ib] >> (2 * l)) & 3) << 8];
          const uint8_t s = signs[4 * ib + l];
          put8(q + 32 * ib + 8 * l, negate_masked(g, smask64[s]));
        }
      }
      return 1;
    }
    case CL_T_IQ3_XXS: {
      *d = f16at(b) / 4.0f;
      const uint8_t *qs = b + 2, *gas = b + 66;
      for (int ib = 0; ib < 8; ib++) {
        const uint32_t a = rd32(gas + 4 * ib);
        sc[2 * ib] = sc[2 * ib + 1] = (int16_t)(2 * (a >> 28) + 1);
        for (int l = 0; l < 4; l++) {
          const uint64_t m = ksigns64[(a >> (7 * l)) & 127];
          const uint32_t g1 = iq3xxs_grid[qs[8 * ib + 2 * l]], g2 = iq3xxs_grid[qs[8 * ib + 2 * l + 1]];
          put8(q + 32 * ib + 8 * l, negate_masked((uint64_t)g1 | (uint64_t)g2 << 32, m));
        }
      }
      return 1;
    }
    case CL_T_IQ3_S: {
      *d = f16at(b);
      const uint8_t *qs = b + 2, *qh = b + 66, *signs = b + 74, *scales = b + 106;
      for (int ib = 0; ib < 8; ib++) {
        sc[2 * ib] = sc[2 * ib + 1] = (int16_t)(1 + 2 * ((scales[ib / 2] >> (4 * (ib % 2))) & 0xf));
        for (int r = 0; r < 4; r++) {
          const int m1 = 8 * ib + 2 * r, m2 = m1 + 1;
          const uint32_t g1 = iq3s_grid[qs[m1] | (((qh[ib] >> (2 * r)) & 1) << 8)], g2 = iq3s_grid[qs[m2] | (((qh[ib] >> (2 * r + 1)) & 1) << 8)];
          const uint8_t s = signs[4 * ib + r];
          put8(q + 32 * ib + 8 * r, negate_masked((uint64_t)g1 | (uint64_t)g2 << 32, smask64[s]));
        }
      }
      return 1;
    }
    case CL_T_IQ1_S: {
      iq1_ready();   /* grid entries are -1, 0 or 1 per weight, shifted by +-1/8; scaled by 8 they are integers, 8*g +- 1 */
      *d = f16at(b) / 8.0f;
      for (int ib = 0; ib < 8; ib++) {
        const uint16_t qh = rd16(b + 34 + 2 * ib);
        sc[2 * ib] = sc[2 * ib + 1] = (int16_t)(2 * ((qh >> 12) & 7) + 1);
        const int dl = (qh & 0x8000) ? -1 : 1;
        for (int l = 0; l < 4; l++) {
          const int gi = b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
          put8(q + 32 * ib + 8 * l, dl > 0 ? iq1_plus[gi] : iq1_minus[gi]);
        }
      }
      return 1;
    }
    case CL_T_IQ1_M: {
      iq1_ready();
      const uint8_t *qs = b, *qh = b + 32, *scp = b + 48;
      uint16_t s16[4];
      for (int k = 0; k < 4; k++) s16[k] = rd16(scp + 2 * k);
      const uint16_t d16 = (uint16_t)((s16[0] >> 12) | ((s16[1] >> 12) << 4) | ((s16[2] >> 12) << 8) | ((s16[3] >> 12) << 12));
      *d = cl_f16_to_f32(d16) / 8.0f;
      for (int ib = 0; ib < 8; ib++) {
        const int sh = 6 * (ib % 2);
        sc[2 * ib] = (int16_t)(2 * ((s16[ib / 2] >> sh) & 7) + 1); sc[2 * ib + 1] = (int16_t)(2 * ((s16[ib / 2] >> (sh + 3)) & 7) + 1);
        for (int l = 0; l < 4; l++) {
          const int nib = (qh[2 * ib + l / 2] >> (4 * (l % 2))) & 0xf;
          const int gi = qs[4 * ib + l] | ((nib & 7) << 8);
          const int dl = (nib & 8) ? -1 : 1;
          put8(q + 32 * ib + 8 * l, dl > 0 ? iq1_plus[gi] : iq1_minus[gi]);
        }
      }
      return 1;
    }
    case CL_T_IQ4_XS: {
      *d = f16at(b);
      const uint16_t sh = rd16(b + 2);
      const uint8_t *sl = b + 4, *qs = b + 8;
      for (int ib = 0; ib < 8; ib++) {
        const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        sc[2 * ib] = sc[2 * ib + 1] = (int16_t)(ls - 32);
        kv16(qs + 16 * ib, q + 32 * ib);
      }
      return 1;
    }
    default: return 0;
  }
}

void cl_iq_dequant(uint32_t t, const void *vb, float *y) {
  if (t == CL_T_IQ4_NL) {
    const uint8_t *b = vb;
    const float d = f16at(b);
    for (int j = 0; j < 16; j++) { y[j] = d * (float)KV[b[2 + j] & 0xf]; y[j + 16] = d * (float)KV[b[2 + j] >> 4]; }
    return;
  }
  int8_t q[256]; int16_t sc[16]; float d;
  if (!cl_iq_unpack(t, vb, q, sc, &d)) { memset(y, 0, 256 * sizeof *y); return; }
  for (int g = 0; g < 16; g++) for (int l = 0; l < 16; l++) y[16 * g + l] = d * (float)sc[g] * (float)q[16 * g + l];
}
