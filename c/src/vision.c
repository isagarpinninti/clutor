#include "vision.h"
#include "gguf.h"
#include "meta.h"
#include "quant.h"
#include "plat.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__aarch64__) && !defined(CL_FORCE_SCALAR)
#include <arm_neon.h>
#endif

typedef struct { const uint8_t *p; uint32_t type; int rows, cols; } vw;   /* rows x cols weights of `type`; a row holds `cols` inputs */
typedef struct { vw qkv, out, up, down; float *qkv_b, *out_b, *up_b, *down_b, *ln1_w, *ln1_b, *ln2_w, *ln2_b; int deepstack; } vblk;
typedef struct { vw fc1, fc2; float *fc1_b, *fc2_b, *norm_w, *norm_b; } vmerge;

struct cl_vision {
  cl_model model; cl_meta meta; uint8_t *map; size_t map_size;
  int embd, ffn, depth, heads, hd, patch, merge, out_embd, grid_side;
  float eps;
  float *patch_w, *patch_b;      /* the two temporal kernels of the patch convolution added together (a still picture is shown to both): embd x (3 * patch * patch) */
  const float *pos;              /* grid_side^2 x embd learned position embeddings (float32 in the file) */
  float *pos_owned;
  vblk *blk;
  vmerge final_m, ds_m[3];       /* the final merger (after the post-norm) and the three DeepStack mergers */
  float *post_w, *post_b;
  int n_ds, ds_layer[3];
};

static const cl_tensor *vfind(const cl_model *m, const char *name) { for (uint64_t i = 0; i < m->n_tensors; i++) if (!strcmp(m->tensors[i].name, name)) return &m->tensors[i]; return NULL; }
static float *vfloat(cl_vision *v, const char *name, size_t n) {
  const cl_tensor *t = vfind(&v->model, name);
  if (!t || (size_t)t->dims[0] * (t->ndims > 1 ? (size_t)t->dims[1] : 1) != n || !cl_type_supported(t->type)) return NULL;
  float *o = malloc(n * sizeof(float));
  if (o) cl_dequant_row(t->type, v->map + t->offset, o, (int)n);
  return o;
}
static int vweight(cl_vision *v, vw *w, const char *name, int rows, int cols) {
  const cl_tensor *t = vfind(&v->model, name);
  if (!t || t->ndims != 2 || (int)t->dims[0] != cols || (int)t->dims[1] != rows || (t->type != CL_T_F16 && t->type != CL_T_F32 && t->type != CL_T_BF16 && t->type != CL_T_Q8_0)) return 0;
  w->p = v->map + t->offset; w->type = t->type; w->rows = rows; w->cols = cols;
  return 1;
}

static void free_merge(vmerge *m) { free(m->fc1_b); free(m->fc2_b); free(m->norm_w); free(m->norm_b); }

void cl_vision_close(cl_vision *v) {
  if (!v) return;
  if (v->blk) for (int i = 0; i < v->depth; i++) { vblk *b = &v->blk[i]; free(b->qkv_b); free(b->out_b); free(b->up_b); free(b->down_b); free(b->ln1_w); free(b->ln1_b); free(b->ln2_w); free(b->ln2_b); }
  free(v->blk); free(v->patch_w); free(v->patch_b); free(v->pos_owned); free(v->post_w); free(v->post_b);
  free_merge(&v->final_m); for (int i = 0; i < 3; i++) free_merge(&v->ds_m[i]);
  cl_meta_free(&v->meta); cl_model_close(&v->model);
  if (v->map) cl_unmap(v->map, v->map_size);
  free(v);
}

static int load_merge(cl_vision *v, vmerge *m, const char *p1, const char *p2, const char *pn, int fc_in, int fc_out, int norm_n, char *err, size_t errlen) {
  char nm[160];
  snprintf(nm, sizeof nm, "%s.weight", p1); if (!vweight(v, &m->fc1, nm, fc_in, fc_in)) { snprintf(err, errlen, "the projector has no usable %s", nm); return 0; }
  snprintf(nm, sizeof nm, "%s.weight", p2); if (!vweight(v, &m->fc2, nm, fc_out, fc_in)) { snprintf(err, errlen, "the projector has no usable %s", nm); return 0; }
  snprintf(nm, sizeof nm, "%s.bias", p1); m->fc1_b = vfloat(v, nm, (size_t)fc_in);
  snprintf(nm, sizeof nm, "%s.bias", p2); m->fc2_b = vfloat(v, nm, (size_t)fc_out);
  if (pn) { snprintf(nm, sizeof nm, "%s.weight", pn); m->norm_w = vfloat(v, nm, (size_t)norm_n); snprintf(nm, sizeof nm, "%s.bias", pn); m->norm_b = vfloat(v, nm, (size_t)norm_n); if (!m->norm_w || !m->norm_b) { snprintf(err, errlen, "the projector has no usable %s", pn); return 0; } }
  if (!m->fc1_b || !m->fc2_b) { snprintf(err, errlen, "the projector has no usable biases for %s", p1); return 0; }
  return 1;
}

cl_vision *cl_vision_open(const char *path, int llm_embd, char *err, size_t errlen) {
  cl_vision *v = calloc(1, sizeof *v);
  if (!v) { snprintf(err, errlen, "out of memory"); return NULL; }
  if (cl_model_open(&v->model, path, err, errlen)) { free(v); return NULL; }
  int fd = cl_open_ro(path);
  if (fd < 0) { snprintf(err, errlen, "cannot open %s", path); cl_model_close(&v->model); free(v); return NULL; }
  v->map_size = (size_t)cl_fd_size(fd);
  v->map = cl_map_ro(fd, v->map_size);
  cl_close(fd);
  if (!v->map) { snprintf(err, errlen, "cannot map %s", path); cl_model_close(&v->model); free(v); return NULL; }
  if (cl_meta_parse(&v->meta, v->map, v->map_size, err, errlen)) { cl_unmap(v->map, v->map_size); cl_model_close(&v->model); free(v); return NULL; }
  cl_str s;
  if (strcmp(v->model.arch, "clip") || !cl_meta_str(&v->meta, "clip.projector_type", &s) || s.len != 14 || strncmp(s.p, "qwen3vl_merger", 14)) { snprintf(err, errlen, "this is not a Qwen3-VL projector file (architecture \"%s\")", v->model.arch); goto fail; }
  uint64_t u; double d;
#define NEED(key, dst) do { if (!cl_meta_u64(&v->meta, key, &u)) { snprintf(err, errlen, "the projector does not state %s", key); goto fail; } dst = (int)u; } while (0)
  NEED("clip.vision.embedding_length", v->embd); NEED("clip.vision.feed_forward_length", v->ffn); NEED("clip.vision.block_count", v->depth); NEED("clip.vision.attention.head_count", v->heads);
  NEED("clip.vision.patch_size", v->patch); NEED("clip.vision.spatial_merge_size", v->merge); NEED("clip.vision.projection_dim", v->out_embd);
  v->eps = cl_meta_f64(&v->meta, "clip.vision.attention.layer_norm_epsilon", &d) ? (float)d : 1e-6f;
  if (v->out_embd != llm_embd) { snprintf(err, errlen, "the projector produces %d-wide vectors but the language model is %d wide: they belong to different models", v->out_embd, llm_embd); goto fail; }
  if (v->embd <= 0 || v->heads <= 0 || v->embd % v->heads || v->depth <= 0 || v->depth > 64 || v->patch != 16 || v->merge != 2 || (v->embd / v->heads) % 4 || v->embd % 4 || v->ffn % 4) { snprintf(err, errlen, "unsupported projector shape"); goto fail; }
  v->hd = v->embd / v->heads;
  const int P3 = 3 * v->patch * v->patch, E = v->embd, M = E * v->merge * v->merge;
  v->blk = calloc((size_t)v->depth, sizeof *v->blk);
  if (!v->blk) { snprintf(err, errlen, "out of memory"); goto fail; }
  /* which blocks feed a DeepStack merger */
  const cl_kv *kv = cl_meta_get(&v->meta, "clip.vision.is_deepstack_layers");
  for (int i = 0; i < v->depth; i++) { v->blk[i].deepstack = kv && kv->type == 9 && (uint64_t)i < kv->count && v->map[kv->off + (size_t)i] ? 1 : 0; if (v->blk[i].deepstack && v->n_ds < 3) v->ds_layer[v->n_ds++] = i; }
  if (v->n_ds != 3) { snprintf(err, errlen, "expected three DeepStack layers, found %d", v->n_ds); goto fail; }
  /* the patch convolution: two temporal kernels [16,16,3,E]; a still picture is shown to both, so their sum is applied once */
  {
    const cl_tensor *w0 = vfind(&v->model, "v.patch_embd.weight"), *w1 = vfind(&v->model, "v.patch_embd.weight.1");
    if (!w0 || !w1 || w0->type != CL_T_F16 || w1->type != CL_T_F16 || (size_t)w0->dims[0] * w0->dims[1] * w0->dims[2] != (size_t)P3 || (int)w0->dims[3] != E) { snprintf(err, errlen, "the projector's patch embedding is not the expected two float16 kernels"); goto fail; }
    v->patch_w = malloc((size_t)E * P3 * sizeof(float)); float *tmp = malloc((size_t)E * P3 * sizeof(float));
    if (!v->patch_w || !tmp) { free(tmp); snprintf(err, errlen, "out of memory"); goto fail; }
    cl_dequant_row(CL_T_F16, v->map + w0->offset, v->patch_w, E * P3); cl_dequant_row(CL_T_F16, v->map + w1->offset, tmp, E * P3);
    for (size_t i = 0; i < (size_t)E * P3; i++) v->patch_w[i] += tmp[i];
    free(tmp);
    v->patch_b = vfloat(v, "v.patch_embd.bias", (size_t)E);
    if (!v->patch_b) { snprintf(err, errlen, "the projector has no patch embedding bias"); goto fail; }
  }
  {
    const cl_tensor *pe = vfind(&v->model, "v.position_embd.weight");
    if (!pe || pe->ndims != 2 || (int)pe->dims[0] != E) { snprintf(err, errlen, "the projector has no position embedding"); goto fail; }
    v->grid_side = (int)lround(sqrt((double)pe->dims[1]));
    if (v->grid_side * v->grid_side != (int)pe->dims[1]) { snprintf(err, errlen, "the position embedding is not a square grid"); goto fail; }
    if (pe->type == CL_T_F32) v->pos = (const float *)(const void *)(v->map + pe->offset);
    else { v->pos_owned = malloc((size_t)pe->dims[1] * E * sizeof(float)); if (!v->pos_owned) { snprintf(err, errlen, "out of memory"); goto fail; } cl_dequant_row(pe->type, v->map + pe->offset, v->pos_owned, (int)(pe->dims[1] * E)); v->pos = v->pos_owned; }
  }
  for (int i = 0; i < v->depth; i++) {
    vblk *b = &v->blk[i]; char nm[160];
#define W(field, suffix, r, c) do { snprintf(nm, sizeof nm, "v.blk.%d." suffix ".weight", i); if (!vweight(v, &b->field, nm, r, c)) { snprintf(err, errlen, "the projector has no usable %s", nm); goto fail; } } while (0)
#define F(field, suffix, n) do { snprintf(nm, sizeof nm, "v.blk.%d." suffix, i); b->field = vfloat(v, nm, (size_t)(n)); if (!b->field) { snprintf(err, errlen, "the projector has no usable %s", nm); goto fail; } } while (0)
    W(qkv, "attn_qkv", 3 * E, E); W(out, "attn_out", E, E); W(up, "ffn_up", v->ffn, E); W(down, "ffn_down", E, v->ffn);
    F(qkv_b, "attn_qkv.bias", 3 * E); F(out_b, "attn_out.bias", E); F(up_b, "ffn_up.bias", v->ffn); F(down_b, "ffn_down.bias", E);
    F(ln1_w, "ln1.weight", E); F(ln1_b, "ln1.bias", E); F(ln2_w, "ln2.weight", E); F(ln2_b, "ln2.bias", E);
  }
  v->post_w = vfloat(v, "v.post_ln.weight", (size_t)E); v->post_b = vfloat(v, "v.post_ln.bias", (size_t)E);
  if (!v->post_w || !v->post_b) { snprintf(err, errlen, "the projector has no post-norm"); goto fail; }
  if (!load_merge(v, &v->final_m, "mm.0", "mm.2", NULL, M, v->out_embd, 0, err, errlen)) goto fail;
  for (int i = 0; i < 3; i++) {
    char a[64], b[64], c[64];
    snprintf(a, sizeof a, "v.deepstack.%d.fc1", v->ds_layer[i]); snprintf(b, sizeof b, "v.deepstack.%d.fc2", v->ds_layer[i]); snprintf(c, sizeof c, "v.deepstack.%d.norm", v->ds_layer[i]);
    if (!load_merge(v, &v->ds_m[i], a, b, c, M, v->out_embd, M, err, errlen)) goto fail;
  }
  return v;
fail:
  cl_vision_close(v);
  return NULL;
}

void cl_vision_out_free(cl_vision_out *o) { if (!o) return; free(o->emb); for (int i = 0; i < 3; i++) free(o->ds[i]); memset(o, 0, sizeof *o); }

/* ---- arithmetic ---- */
static inline float dotv(const float *a, const float *b, int n) {   /* n is a multiple of 4 */
#if defined(__aarch64__) && !defined(CL_FORCE_SCALAR)
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0), s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
  int i = 0;
  for (; i + 16 <= n; i += 16) { s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i)); s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)); s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8)); s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12)); }
  for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
#else
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (int i = 0; i < n; i += 4) { s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]; }
  return (s0 + s1) + (s2 + s3);
#endif
}

#define RB 16    /* weight rows converted to float together */
#define TB 16    /* tokens multiplied with them */
typedef struct { const vw *w; const float *bias; const float *X; float *Y; int N, nchunks; } gemm_job;
static void gemm_task(void *vc, int c) {
  const gemm_job *g = vc;
  const int K = g->w->cols, R = g->w->rows, r0 = c * RB, r1 = r0 + RB < R ? r0 + RB : R;
  float *buf = malloc((size_t)RB * K * sizeof(float));
  if (!buf) return;
  const size_t rb = cl_row_bytes(g->w->type, (uint64_t)K);
  for (int r = r0; r < r1; r++) cl_dequant_row(g->w->type, g->w->p + (size_t)r * rb, buf + (size_t)(r - r0) * K, K);
  for (int t0 = 0; t0 < g->N; t0 += TB) {
    const int t1 = t0 + TB < g->N ? t0 + TB : g->N;
    for (int t = t0; t < t1; t++) {
      const float *x = g->X + (size_t)t * K; float *y = g->Y + (size_t)t * R;
      for (int r = r0; r < r1; r++) y[r] = dotv(buf + (size_t)(r - r0) * K, x, K) + (g->bias ? g->bias[r] : 0.0f);
    }
  }
  free(buf);
}
/* Y[t][r] = bias[r] + dot(row r of W, X[t]) for t < N */
static void gemm(const vw *w, const float *bias, const float *X, int N, float *Y) {
  gemm_job g = {w, bias, X, Y, N, (w->rows + RB - 1) / RB};
  cl_parallel_for(g.nchunks, gemm_task, &g);
}

static void layernorm(float *out, const float *x, const float *w, const float *b, int n, float eps) {
  double mu = 0, var = 0;
  for (int i = 0; i < n; i++) mu += x[i];
  mu /= n;
  for (int i = 0; i < n; i++) { double d = x[i] - mu; var += d * d; }
  var /= n;
  const float inv = 1.0f / sqrtf((float)var + eps), m = (float)mu;
  for (int i = 0; i < n; i++) out[i] = (x[i] - m) * inv * w[i] + b[i];
}
static inline float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
static inline float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }

typedef struct { float *X, *out; const float *w, *b; int N, n; float eps; } ln_job;
static void ln_task(void *vc, int i) { const ln_job *j = vc; layernorm(j->out + (size_t)i * j->n, j->X + (size_t)i * j->n, j->w, j->b, j->n, j->eps); }

typedef struct { const float *qkv; float *out; const float *cs; int N, heads, hd, E; } attn_job;
static void attn_task(void *vc, int idx) {
  const attn_job *a = vc;
  const int h = idx / a->N, qi = idx % a->N, hd = a->hd, E3 = 3 * a->E;
  const float *q = a->qkv + (size_t)qi * E3 + (size_t)h * hd;
  float *sc = malloc((size_t)a->N * sizeof(float));
  if (!sc) return;
  const float scale = 1.0f / sqrtf((float)hd);
  float mx = -INFINITY;
  for (int k = 0; k < a->N; k++) { float s = 0; const float *kk = a->qkv + (size_t)k * E3 + a->E + (size_t)h * hd; for (int i = 0; i < hd; i++) s += q[i] * kk[i]; s *= scale; sc[k] = s; if (s > mx) mx = s; }
  double z = 0;
  for (int k = 0; k < a->N; k++) { sc[k] = expf(sc[k] - mx); z += sc[k]; }
  float *o = a->out + (size_t)qi * a->E + (size_t)h * hd;
  for (int i = 0; i < hd; i++) o[i] = 0;
  for (int k = 0; k < a->N; k++) { const float wgt = (float)(sc[k] / z); const float *vv = a->qkv + (size_t)k * E3 + 2 * a->E + (size_t)h * hd; for (int i = 0; i < hd; i++) o[i] += wgt * vv[i]; }
  free(sc);
}

typedef struct { float *qkv; const float *cs; int N, E, heads, hd; } rope_job;
/* two-dimensional rotary embedding on queries and keys: the first half of the head's rotation pairs follow the patch's row, the second half its column (Hugging Face's rotate_half layout) */
static void rope_task(void *vc, int t) {
  const rope_job *r = vc;
  const int half = r->hd / 2, E3 = 3 * r->E;
  const float *cs = r->cs + (size_t)t * r->hd;   /* cos[half], sin[half] */
  for (int part = 0; part < 2; part++) for (int h = 0; h < r->heads; h++) {
    float *x = r->qkv + (size_t)t * E3 + (size_t)part * r->E + (size_t)h * r->hd;
    for (int j = 0; j < half; j++) { const float c = cs[j], s = cs[half + j], a = x[j], b = x[j + half]; x[j] = a * c - b * s; x[j + half] = b * c + a * s; }
  }
}

typedef struct { const float *X; float *out; const float *w, *b; float eps; int n; } mnorm_job;

/* merger: (optionally norm the 4*E-wide groups) -> fc1 -> GELU -> fc2. `in` is Nm x M (groups of four consecutive patch vectors). */
static float *merge_forward(const cl_vision *v, const vmerge *m, const float *in, int Nm, int M, int normalize) {
  float *a = malloc((size_t)Nm * M * sizeof(float)), *h = malloc((size_t)Nm * M * sizeof(float)), *o = malloc((size_t)Nm * v->out_embd * sizeof(float));
  if (!a || !h || !o) { free(a); free(h); free(o); return NULL; }
  const float *src = in;
  if (normalize) { ln_job lj = {(float *)(uintptr_t)in, a, m->norm_w, m->norm_b, Nm, M, v->eps}; cl_parallel_for(Nm, ln_task, &lj); src = a; }
  gemm(&m->fc1, m->fc1_b, src, Nm, h);
  for (size_t i = 0; i < (size_t)Nm * M; i++) h[i] = gelu_erf(h[i]);
  gemm(&m->fc2, m->fc2_b, h, Nm, o);
  free(a); free(h);
  return o;
}

typedef struct { const uint8_t *rgb; float *P; int w, gw; } patch_job;
static void patch_task(void *vc, int p) {
  const patch_job *j = vc;
  const int bw = j->gw / 2, blk = p / 4, inner = p % 4;
  const int ph = (blk / bw) * 2 + inner / 2, pw = (blk % bw) * 2 + inner % 2;
  float *P = j->P + (size_t)p * 768;
  for (int c = 0; c < 3; c++) for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++)
    P[(c * 16 + y) * 16 + x] = (float)j->rgb[((size_t)(ph * 16 + y) * (size_t)j->w + (size_t)(pw * 16 + x)) * 3 + (size_t)c] / 127.5f - 1.0f;   /* (value / 255 - 0.5) / 0.5 */
}

int cl_vision_encode(cl_vision *v, const uint8_t *rgb, int w, int h, cl_vision_out *out, char *err, size_t errlen) {
  memset(out, 0, sizeof *out);
  if (w < 64 || h < 64 || w % 32 || h % 32) { snprintf(err, errlen, "the picture must be at least 64 x 64 pixels with both sides a multiple of 32 (it is %d x %d)", w, h); return 1; }
  const int gh = h / 16, gw = w / 16, N = gh * gw, E = v->embd, M = 4 * E, Nm = N / 4;
  if (N > 4096) { snprintf(err, errlen, "the picture is too large (%d patches; at most 4096)", N); return 1; }
  float *X = malloc((size_t)N * E * sizeof(float)), *T = malloc((size_t)N * E * sizeof(float)), *QKV = malloc((size_t)N * 3 * E * sizeof(float)), *F = malloc((size_t)N * v->ffn * sizeof(float));
  float *P = malloc((size_t)N * 768 * sizeof(float)), *CS = malloc((size_t)N * v->hd * sizeof(float));
  if (!X || !T || !QKV || !F || !P || !CS) { free(X); free(T); free(QKV); free(F); free(P); free(CS); snprintf(err, errlen, "out of memory"); return 1; }
  /* 1. patches, in the order the merger wants (each 2 x 2 neighbourhood together), through the patch convolution */
  { patch_job pj = {rgb, P, w, gw}; cl_parallel_for(N, patch_task, &pj); }
  { vw pw_ = {(const uint8_t *)(const void *)v->patch_w, CL_T_F32, E, 768}; gemm(&pw_, v->patch_b, P, N, X); }
  /* 2. learned position embeddings, bilinearly resampled to this picture's grid (corners aligned), and the rotary angles */
  const int S = v->grid_side;
  const int half = v->hd / 2, qd = half / 2;   /* rotation pairs: the first qd follow the row, the next qd the column */
  float *inv = malloc((size_t)qd * sizeof(float));
  if (!inv) { free(X); free(T); free(QKV); free(F); free(P); free(CS); snprintf(err, errlen, "out of memory"); return 1; }
  for (int k = 0; k < qd; k++) inv[k] = powf(10000.0f, -(float)(2 * k) / (float)half);
  for (int p = 0; p < N; p++) {
    const int bw = gw / 2, blk = p / 4, inner = p % 4;
    const int ph = (blk / bw) * 2 + inner / 2, pw = (blk % bw) * 2 + inner % 2;
    const float fh = gh > 1 ? (float)(S - 1) * (float)ph / (float)(gh - 1) : 0.0f, fw = gw > 1 ? (float)(S - 1) * (float)pw / (float)(gw - 1) : 0.0f;
    const int h0 = (int)floorf(fh), w0 = (int)floorf(fw), h1 = h0 + 1 < S ? h0 + 1 : S - 1, w1 = w0 + 1 < S ? w0 + 1 : S - 1;
    const float dh = fh - (float)h0, dw = fw - (float)w0;
    const float c00 = (1 - dh) * (1 - dw), c01 = (1 - dh) * dw, c10 = dh * (1 - dw), c11 = dh * dw;
    const float *e00 = v->pos + (size_t)(h0 * S + w0) * E, *e01 = v->pos + (size_t)(h0 * S + w1) * E, *e10 = v->pos + (size_t)(h1 * S + w0) * E, *e11 = v->pos + (size_t)(h1 * S + w1) * E;
    float *x = X + (size_t)p * E;
    for (int i = 0; i < E; i++) x[i] += c00 * e00[i] + c01 * e01[i] + c10 * e10[i] + c11 * e11[i];
    float *cs = CS + (size_t)p * v->hd;
    for (int j = 0; j < half; j++) { const float ang = j < qd ? (float)ph * inv[j] : (float)pw * inv[j - qd]; cs[j] = cosf(ang); cs[half + j] = sinf(ang); }
  }
  free(inv);
  /* 3. the transformer blocks, collecting the DeepStack states */
  float *ds_state[3] = {0, 0, 0};
  int ds_n = 0, rc = 0;
  for (int l = 0; l < v->depth && !rc; l++) {
    const vblk *b = &v->blk[l];
    { ln_job lj = {X, T, b->ln1_w, b->ln1_b, N, E, v->eps}; cl_parallel_for(N, ln_task, &lj); }
    gemm(&b->qkv, b->qkv_b, T, N, QKV);
    { rope_job rj = {QKV, CS, N, E, v->heads, v->hd}; cl_parallel_for(N, rope_task, &rj); }
    { attn_job aj = {QKV, T, CS, N, v->heads, v->hd, E}; cl_parallel_for(v->heads * N, attn_task, &aj); }
    gemm(&b->out, b->out_b, T, N, QKV);   /* QKV is free to reuse: its first N x E floats hold the projection */
    for (size_t i = 0; i < (size_t)N * E; i++) X[i] += QKV[i];
    { ln_job lj = {X, T, b->ln2_w, b->ln2_b, N, E, v->eps}; cl_parallel_for(N, ln_task, &lj); }
    gemm(&b->up, b->up_b, T, N, F);
    for (size_t i = 0; i < (size_t)N * v->ffn; i++) F[i] = gelu_tanh(F[i]);
    gemm(&b->down, b->down_b, F, N, QKV);
    for (size_t i = 0; i < (size_t)N * E; i++) X[i] += QKV[i];
    if (b->deepstack && ds_n < 3) { ds_state[ds_n] = malloc((size_t)N * E * sizeof(float)); if (!ds_state[ds_n]) rc = 1; else { memcpy(ds_state[ds_n], X, (size_t)N * E * sizeof(float)); ds_n++; } }
  }
  free(QKV); free(F); free(P); free(CS);
  if (rc) { for (int i = 0; i < 3; i++) free(ds_state[i]); free(X); free(T); snprintf(err, errlen, "out of memory"); return 1; }
  /* 4. the mergers */
  out->n = Nm; out->gh = gh / 2; out->gw = gw / 2; out->n_ds = 3;
  for (int i = 0; i < 3; i++) { out->ds[i] = merge_forward(v, &v->ds_m[i], ds_state[i], Nm, M, 1); free(ds_state[i]); if (!out->ds[i]) rc = 1; }
  { ln_job lj = {X, T, v->post_w, v->post_b, N, E, v->eps}; cl_parallel_for(N, ln_task, &lj); }
  out->emb = merge_forward(v, &v->final_m, T, Nm, M, 0);
  free(X); free(T);
  if (rc || !out->emb) { cl_vision_out_free(out); snprintf(err, errlen, "out of memory"); return 1; }
  return 0;
}

/* ---- pictures ---- */
void cl_vision_fit(int w, int h, int max_tokens, int *ow, int *oh) {
  if (w < 1) w = 1;
  if (h < 1) h = 1;
  double s = 1.0;
  const double px = (double)w * h, cap = (double)max_tokens * 1024.0;   /* 32 x 32 pixels per token */
  if (px > cap) s = sqrt(cap / px);
  int nw = (int)lround((double)w * s / 32.0) * 32, nh = (int)lround((double)h * s / 32.0) * 32;
  while ((double)nw * nh > cap && (nw > 64 || nh > 64)) { if (nw >= nh && nw > 64) nw -= 32; else if (nh > 64) nh -= 32; else break; }
  if (nw < 64) nw = 64;
  if (nh < 64) nh = 64;
  *ow = nw; *oh = nh;
}

uint8_t *cl_image_resize(const uint8_t *rgb, int w, int h, int ow, int oh) {
  uint8_t *o = malloc((size_t)ow * oh * 3);
  if (!o) return NULL;
  const double sx = (double)w / ow, sy = (double)h / oh;
  for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
    double x0 = x * sx, x1 = (x + 1) * sx, y0 = y * sy, y1 = (y + 1) * sy;
    if (sx <= 2 && sy <= 2) {   /* bilinear at the pixel's centre */
      const double fx = (x + 0.5) * sx - 0.5, fy = (y + 0.5) * sy - 0.5;
      int ix = (int)floor(fx), iy = (int)floor(fy);
      const double ax = fx - ix, ay = fy - iy;
      for (int c = 0; c < 3; c++) {
        int xa = ix < 0 ? 0 : ix >= w ? w - 1 : ix, xb = ix + 1 < 0 ? 0 : ix + 1 >= w ? w - 1 : ix + 1, ya = iy < 0 ? 0 : iy >= h ? h - 1 : iy, yb = iy + 1 < 0 ? 0 : iy + 1 >= h ? h - 1 : iy + 1;
        double t = (1 - ay) * ((1 - ax) * rgb[((size_t)ya * w + xa) * 3 + c] + ax * rgb[((size_t)ya * w + xb) * 3 + c]) + ay * ((1 - ax) * rgb[((size_t)yb * w + xa) * 3 + c] + ax * rgb[((size_t)yb * w + xb) * 3 + c]);
        o[((size_t)y * ow + x) * 3 + c] = (uint8_t)(t < 0 ? 0 : t > 255 ? 255 : t + 0.5);
      }
    } else {   /* shrinking a lot: average everything under the output pixel */
      int xa = (int)floor(x0), xb = (int)ceil(x1), ya = (int)floor(y0), yb = (int)ceil(y1);
      if (xb > w) xb = w;
      if (yb > h) yb = h;
      for (int c = 0; c < 3; c++) {
        double sum = 0, wsum = 0;
        for (int yy = ya; yy < yb; yy++) for (int xx = xa; xx < xb; xx++) { double wx = (xx + 1 < x1 ? xx + 1 : x1) - (xx > x0 ? xx : x0), wy = (yy + 1 < y1 ? yy + 1 : y1) - (yy > y0 ? yy : y0), ww = wx * wy; sum += ww * rgb[((size_t)yy * w + xx) * 3 + c]; wsum += ww; }
        o[((size_t)y * ow + x) * 3 + c] = (uint8_t)(wsum > 0 ? sum / wsum + 0.5 : 0);
      }
    }
  }
  return o;
}

uint8_t *cl_image_load_ppm(const char *path, int *w, int *h) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return NULL;
  int mx = 0;
  char m[3] = {0};
  if (fscanf(fp, "%2s %d %d %d", m, w, h, &mx) != 4 || strcmp(m, "P6") || mx != 255 || *w < 1 || *h < 1 || (long long)*w * *h > 64000000LL) { fclose(fp); return NULL; }
  fgetc(fp);   /* the single whitespace after the header */
  uint8_t *b = malloc((size_t)*w * *h * 3);
  if (b && fread(b, 1, (size_t)*w * *h * 3, fp) != (size_t)*w * *h * 3) { free(b); b = NULL; }
  fclose(fp);
  return b;
}
