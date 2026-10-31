#include "quant.h"
#include "kernel.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

/* ---- thread pool: a matvec is split into row ranges, one per worker; the caller works too ---- */
typedef struct { uint32_t type; const uint8_t *w; int rows, cols; const float *x; float *out; uint64_t rowb; } job_t;
static int g_threads = 0;

static void run_rows(const job_t *j, int r0, int r1) {
  float *tmp = malloc((size_t)j->cols * sizeof(float));
  if (!tmp) return;
  for (int r = r0; r < r1; r++) {
    const uint8_t *row = j->w + (size_t)r * j->rowb;
    float acc = 0;
    if (j->type == CL_T_F32) {
      const float *wf = (const float *)(const void *)row;
      for (int i = 0; i < j->cols; i++) acc += wf[i] * j->x[i];
    } else {
      /* dequantize a block-sized slice at a time, so the scratch row stays in cache */
      uint32_t bl = cl_type_block(j->type), bb = cl_type_bytes(j->type);
      float buf[256];
      for (int c = 0; c < j->cols; c += (int)bl) {
        cl_dequant_row(j->type, row + (size_t)(c / (int)bl) * bb, buf, (int)bl);
        for (uint32_t i = 0; i < bl; i++) acc += buf[i] * j->x[c + (int)i];
      }
    }
    j->out[r] = acc;
  }
  free(tmp);
}

typedef struct { const job_t *job; int r0, r1; } slice_t;
static void *worker(void *p) { slice_t *s = p; run_rows(s->job, s->r0, s->r1); return NULL; }

void cl_set_threads(int n) { g_threads = n < 1 ? 1 : n > 16 ? 16 : n; }
int cl_get_threads(void) {
  if (!g_threads) { long c = sysconf(_SC_NPROCESSORS_ONLN); cl_set_threads(c > 0 ? (int)c : 1); }
  return g_threads;
}

void cl_matvec(uint32_t type, const void *w, int rows, int cols, const float *x, float *out) {
  job_t j = {type, w, rows, cols, x, out, cl_row_bytes(type, (uint64_t)cols)};
  int nt = cl_get_threads();
  if (rows < 64 || nt == 1 || (uint64_t)rows * (uint64_t)cols < (1u << 18)) { run_rows(&j, 0, rows); return; }
  if (nt > rows) nt = rows;
  pthread_t th[16];
  slice_t sl[16];
  int per = (rows + nt - 1) / nt, started = 0;
  for (int t = 1; t < nt; t++) {
    sl[t] = (slice_t){&j, t * per, (t + 1) * per > rows ? rows : (t + 1) * per};
    if (sl[t].r0 >= rows) break;
    if (pthread_create(&th[t], NULL, worker, &sl[t])) { run_rows(&j, sl[t].r0, sl[t].r1); continue; }
    started = t;
  }
  run_rows(&j, 0, per < rows ? per : rows);
  for (int t = 1; t <= started; t++) pthread_join(th[t], NULL);
}
