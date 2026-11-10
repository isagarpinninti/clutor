#include "iq.h"
#include "kernel.h"
#include "iq_tables.h"
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

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static float f16at(const uint8_t *p) { return cl_f16_to_f32(rd16(p)); }
/* Byte j of a table entry, as a signed value (the tables are little-endian packed vectors). */
static int8_t gb64(uint64_t g, int j) { return (int8_t)(uint8_t)(g >> (8 * j)); }
static int8_t gb32(uint32_t g, int j) { return (int8_t)(uint8_t)(g >> (8 * j)); }
static int8_t sgn(int8_t v, int neg) { return neg ? (int8_t)-v : v; }

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
          const uint8_t s = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
          for (int j = 0; j < 8; j++) q[32 * ib + 8 * l + j] = sgn(gb64(g, j), (s >> j) & 1);
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
          const uint8_t s = ksigns_iq2xs[v >> 9];
          for (int j = 0; j < 8; j++) q[32 * ib + 8 * l + j] = sgn(gb64(g, j), (s >> j) & 1);
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
          for (int j = 0; j < 8; j++) q[32 * ib + 8 * l + j] = sgn(gb64(g, j), (s >> j) & 1);
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
          const uint8_t s = ksigns_iq2xs[(a >> (7 * l)) & 127];
          const uint32_t g1 = iq3xxs_grid[qs[8 * ib + 2 * l]], g2 = iq3xxs_grid[qs[8 * ib + 2 * l + 1]];
          for (int j = 0; j < 4; j++) { q[32 * ib + 8 * l + j] = sgn(gb32(g1, j), (s >> j) & 1); q[32 * ib + 8 * l + 4 + j] = sgn(gb32(g2, j), (s >> (j + 4)) & 1); }
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
          for (int j = 0; j < 4; j++) { q[32 * ib + 8 * r + j] = sgn(gb32(g1, j), (s >> j) & 1); q[32 * ib + 8 * r + 4 + j] = sgn(gb32(g2, j), (s >> (j + 4)) & 1); }
        }
      }
      return 1;
    }
    case CL_T_IQ1_S: {   /* grid entries are -1, 0 or 1 per weight, shifted by +-1/8; scaled by 8 they are integers, 8*g +- 1 */
      *d = f16at(b) / 8.0f;
      for (int ib = 0; ib < 8; ib++) {
        const uint16_t qh = rd16(b + 34 + 2 * ib);
        sc[2 * ib] = sc[2 * ib + 1] = (int16_t)(2 * ((qh >> 12) & 7) + 1);
        const int dl = (qh & 0x8000) ? -1 : 1;
        for (int l = 0; l < 4; l++) {
          const uint64_t g = iq1s_grid[b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
          for (int j = 0; j < 8; j++) q[32 * ib + 8 * l + j] = (int8_t)(8 * gb64(g, j) + dl);
        }
      }
      return 1;
    }
    case CL_T_IQ1_M: {
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
          const uint64_t g = iq1s_grid[qs[4 * ib + l] | ((nib & 7) << 8)];
          const int dl = (nib & 8) ? -1 : 1;
          for (int j = 0; j < 8; j++) q[32 * ib + 8 * l + j] = (int8_t)(8 * gb64(g, j) + dl);
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
        for (int j = 0; j < 16; j++) { q[32 * ib + j] = KV[qs[16 * ib + j] & 0xf]; q[32 * ib + 16 + j] = KV[qs[16 * ib + j] >> 4]; }
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
