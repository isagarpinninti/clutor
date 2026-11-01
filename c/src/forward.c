#define _GNU_SOURCE
#include "forward.h"
#include "quant.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const cl_tensor *find(const cl_model *m, const char *name) {
  for (uint64_t i = 0; i < m->n_tensors; i++) if (!strcmp(m->tensors[i].name, name)) return &m->tensors[i];
  return NULL;
}

static int bind(cl_fwd *f, cl_w *w, const char *name, int blk, int want_dims, char *err, size_t errlen) {
  char nm[200];
  if (blk >= 0) snprintf(nm, sizeof nm, "blk.%d.%s", blk, name); else snprintf(nm, sizeof nm, "%s", name);
  const cl_tensor *t = find(&f->model, nm);
  if (!t) { snprintf(err, errlen, "the model has no tensor \"%s\"", nm); return 1; }
  if ((int)t->ndims != want_dims) { snprintf(err, errlen, "tensor \"%s\" has %u dimensions, expected %d", nm, t->ndims, want_dims); return 1; }
  if (!t->nbytes || !cl_type_supported(t->type)) { snprintf(err, errlen, "tensor \"%s\" is stored as %s, which this engine cannot compute with yet", nm, cl_type_name(t->type)); return 1; }
  if (t->dims[0] % cl_type_block(t->type)) { snprintf(err, errlen, "tensor \"%s\" row length %llu is not a multiple of its block", nm, (unsigned long long)t->dims[0]); return 1; }
  w->p = f->map + t->offset; w->type = t->type; w->cols = (int)t->dims[0];
  w->rows = want_dims >= 2 ? (int)t->dims[1] : 1;
  return 0;
}

static float *to_float(const cl_w *w, size_t n) {
  float *o = malloc(n * sizeof(float));
  if (o) cl_dequant_row(w->type, w->p, o, (int)n);
  return o;
}

/* A metadata key of this model's architecture, for example "olmoe.block_count". */
typedef struct { const char *name; int qk_mode, norm_topk, rope_pairs, expert_ff_key, scaled; } arch_spec;
static const arch_spec ARCHS[] = {
  /* name        qk-norm  renorm  pair-rope  expert_feed_forward_length  granite multipliers */
  {"olmoe",      1,       0,      0,         0,                          0},
  {"qwen3moe",   2,       1,      0,         1,                          0},
  {"granitemoe", 0,       1,      1,         0,                          1},
};

#define AKEY(buf, suffix) snprintf(buf, sizeof buf, "%s.%s", f->arch, suffix)
#define REQ_U(suffix, dst) do { char k_[96]; uint64_t v_; AKEY(k_, suffix); if (!cl_meta_u64(&f->meta, k_, &v_)) { snprintf(err, errlen, "the model does not state %s", k_); goto fail; } dst = (int)v_; } while (0)

int cl_fwd_open(cl_fwd *f, const char *path, int n_ctx, const cl_fwd_opts *opts, char *err, size_t errlen) {
  memset(f, 0, sizeof *f);
  if (cl_model_open(&f->model, path, err, errlen)) return 1;
  const arch_spec *spec = NULL;
  for (size_t i = 0; i < sizeof ARCHS / sizeof *ARCHS; i++) if (!strcmp(f->model.arch, ARCHS[i].name)) spec = &ARCHS[i];
  if (!spec) { snprintf(err, errlen, "the forward pass supports the \"olmoe\", \"qwen3moe\" and \"granitemoe\" architectures, not \"%s\"", f->model.arch); goto fail; }
  snprintf(f->arch, sizeof f->arch, "%s", spec->name);
  f->qk_mode = spec->qk_mode; f->qk_per_head = spec->qk_mode == 2; f->norm_topk = spec->norm_topk; f->rope_pairs = spec->rope_pairs;
  f->embed_scale = f->resid_scale = f->logit_scale = 1.0f; f->attn_scale = 0;
  int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st)) { snprintf(err, errlen, "cannot open %s", path); if (fd >= 0) close(fd); goto fail; }
  f->map_size = (size_t)st.st_size;
  f->map = mmap(NULL, f->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (f->map == MAP_FAILED) { f->map = NULL; snprintf(err, errlen, "cannot map %s", path); goto fail; }
  if (cl_meta_parse(&f->meta, f->map, f->map_size, err, errlen)) goto fail;
  REQ_U("embedding_length", f->n_embd);
  REQ_U("attention.head_count", f->n_head);
  REQ_U("attention.head_count_kv", f->n_head_kv);
  { char k1[96], k2[96]; uint64_t v_; AKEY(k1, "expert_feed_forward_length"); AKEY(k2, "feed_forward_length");
    if ((spec->expert_ff_key && cl_meta_u64(&f->meta, k1, &v_)) || (!spec->expert_ff_key && cl_meta_u64(&f->meta, k2, &v_))) f->n_ff_exp = (int)v_; else { snprintf(err, errlen, "the model does not state its expert feed-forward length"); goto fail; } }   /* olmoe files store the per-expert width under the plain key */
  REQ_U("context_length", f->n_ctx_max);
  f->n_layer = (int)f->model.n_layers; f->n_expert = (int)f->model.n_experts; f->n_used = (int)f->model.n_used;
  double d = 0;
  { char k[96];
    AKEY(k, "attention.layer_norm_rms_epsilon"); f->rms_eps = cl_meta_f64(&f->meta, k, &d) ? (float)d : 1e-5f;
    AKEY(k, "rope.freq_base"); f->rope_base = cl_meta_f64(&f->meta, k, &d) ? (float)d : 10000.0f; }
  if (f->n_embd < 256 || f->n_head < 1 || f->n_head > f->n_embd || f->n_embd % f->n_head || f->n_layer < 1 || f->n_expert < 1 || f->n_used < 1 || f->n_used > f->n_expert || f->n_ff_exp < 1) {
    snprintf(err, errlen, "implausible hyper-parameters (embedding %d, heads %d, layers %d, experts %d, used %d, expert width %d)", f->n_embd, f->n_head, f->n_layer, f->n_expert, f->n_used, f->n_ff_exp); goto fail;
  }
  f->head_dim = f->n_embd / f->n_head;
  { char k[96]; uint64_t v_; AKEY(k, "attention.key_length"); if (cl_meta_u64(&f->meta, k, &v_) && v_ >= 2 && v_ <= 1024 && v_ % 2 == 0) f->head_dim = (int)v_; }   /* qwen3moe: 128, not n_embd / n_head */
  if (f->n_head_kv < 1 || f->n_head_kv > f->n_head || f->n_head % f->n_head_kv) { snprintf(err, errlen, "implausible head counts (%d query heads, %d key/value heads)", f->n_head, f->n_head_kv); goto fail; }
  { char k[96]; uint64_t v_; AKEY(k, "rope.dimension_count");
    if (cl_meta_u64(&f->meta, k, &v_) && (int)v_ != f->head_dim) { snprintf(err, errlen, "partial rotary embeddings (%d of %d dimensions) are not implemented", (int)v_, f->head_dim); goto fail; } }
  if (spec->scaled) {
    char k[96]; double dv;
    AKEY(k, "embedding_scale"); if (cl_meta_f64(&f->meta, k, &dv)) f->embed_scale = (float)dv;
    AKEY(k, "residual_scale"); if (cl_meta_f64(&f->meta, k, &dv)) f->resid_scale = (float)dv;
    AKEY(k, "logit_scale"); if (cl_meta_f64(&f->meta, k, &dv) && dv > 0) f->logit_scale = (float)dv;
    AKEY(k, "attention.scale"); if (cl_meta_f64(&f->meta, k, &dv) && dv > 0) f->attn_scale = (float)dv;
  }
  if (f->head_dim * f->n_head != f->n_embd && !f->qk_per_head) { snprintf(err, errlen, "head_dim * heads must equal the embedding width for this architecture"); goto fail; }
  f->n_q = f->n_head * f->head_dim; f->n_kvw = f->n_head_kv * f->head_dim;
  if (f->attn_scale <= 0) f->attn_scale = 1.0f / sqrtf((float)f->head_dim);
  if (f->n_embd % 256 || f->n_ff_exp % 256 || f->n_q % 256) { snprintf(err, errlen, "embedding width %d / expert width %d / query width %d must be multiples of 256 for the K-quant kernels", f->n_embd, f->n_ff_exp, f->n_q); goto fail; }
  if (f->n_used > 64 || f->n_expert > 4096) { snprintf(err, errlen, "unsupported expert counts"); goto fail; }
  if (n_ctx > f->n_ctx_max) n_ctx = f->n_ctx_max;
  if (n_ctx < 1) n_ctx = 512;
  f->n_ctx_max = n_ctx;

  if (bind(f, &f->tok_embd, "token_embd.weight", -1, 2, err, errlen)) goto fail;
  f->n_vocab = f->tok_embd.rows;
  if (f->tok_embd.cols != f->n_embd || f->n_vocab < 1) { snprintf(err, errlen, "token_embd.weight is %d x %d, expected a vocabulary of rows by %d columns", f->tok_embd.rows, f->tok_embd.cols, f->n_embd); goto fail; }
  if (find(&f->model, "output.weight")) { if (bind(f, &f->output, "output.weight", -1, 2, err, errlen)) goto fail; }
  else f->output = f->tok_embd;   /* tied embeddings: the output projection is the embedding matrix */
  cl_w on;
  if (bind(f, &on, "output_norm.weight", -1, 1, err, errlen)) goto fail;
  if (on.cols != f->n_embd || f->output.cols != f->n_embd || f->output.rows != f->n_vocab) { snprintf(err, errlen, "output tensors do not match the embedding width and vocabulary"); goto fail; }
  f->out_norm_f = to_float(&on, (size_t)f->n_embd);
  f->L = calloc((size_t)f->n_layer, sizeof *f->L);
  if (!f->L || !f->out_norm_f) { snprintf(err, errlen, "out of memory"); goto fail; }
  for (int l = 0; l < f->n_layer; l++) {
    cl_fwd_layer *y = &f->L[l];
    if (bind(f, &y->attn_norm, "attn_norm.weight", l, 1, err, errlen) || bind(f, &y->wq, "attn_q.weight", l, 2, err, errlen) || bind(f, &y->wk, "attn_k.weight", l, 2, err, errlen) ||
        bind(f, &y->wv, "attn_v.weight", l, 2, err, errlen) || bind(f, &y->wo, "attn_output.weight", l, 2, err, errlen) || (f->qk_mode && bind(f, &y->qn, "attn_q_norm.weight", l, 1, err, errlen)) ||
        (f->qk_mode && bind(f, &y->kn, "attn_k_norm.weight", l, 1, err, errlen)) || bind(f, &y->ffn_norm, "ffn_norm.weight", l, 1, err, errlen) || bind(f, &y->router, "ffn_gate_inp.weight", l, 2, err, errlen) ||
        bind(f, &y->gate, "ffn_gate_exps.weight", l, 3, err, errlen) || bind(f, &y->up, "ffn_up_exps.weight", l, 3, err, errlen) || bind(f, &y->down, "ffn_down_exps.weight", l, 3, err, errlen)) goto fail;
    const int N = f->n_embd;
    const int QN = f->qk_per_head ? f->head_dim : f->n_q, KN = f->qk_per_head ? f->head_dim : f->n_kvw;
    if (y->attn_norm.cols != N || y->ffn_norm.cols != N || (f->qk_mode && (y->qn.cols != QN || y->kn.cols != KN)) ||
        y->wq.rows != f->n_q || y->wq.cols != N || y->wk.rows != f->n_kvw || y->wk.cols != N || y->wv.rows != f->n_kvw || y->wv.cols != N || y->wo.rows != N || y->wo.cols != f->n_q ||
        y->up.rows != f->n_ff_exp || y->up.cols != N || y->gate.cols != N) {
      snprintf(err, errlen, "layer %d: attention or norm tensor shapes do not match the embedding width %d", l, N); goto fail;
    }
    char nm[200];
    snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", l);
    const cl_tensor *tg = find(&f->model, nm);
    if (y->router.rows != f->n_expert || y->router.cols != f->n_embd || (int)tg->dims[2] != f->n_expert || y->up.cols != N || y->gate.rows != f->n_ff_exp || y->down.rows != f->n_embd || y->down.cols != f->n_ff_exp) {
      snprintf(err, errlen, "layer %d: tensor shapes do not match the hyper-parameters", l); goto fail;
    }
    y->gate_slab = (uint64_t)y->gate.rows * cl_row_bytes(y->gate.type, (uint64_t)y->gate.cols);
    y->up_slab = (uint64_t)y->up.rows * cl_row_bytes(y->up.type, (uint64_t)y->up.cols);
    y->down_slab = (uint64_t)y->down.rows * cl_row_bytes(y->down.type, (uint64_t)y->down.cols);
    y->attn_norm_f = to_float(&y->attn_norm, (size_t)f->n_embd); y->ffn_norm_f = to_float(&y->ffn_norm, (size_t)f->n_embd);
    if (f->qk_mode) { y->qn_f = to_float(&y->qn, (size_t)QN); y->kn_f = to_float(&y->kn, (size_t)KN); }
    y->router_f = malloc((size_t)f->n_expert * f->n_embd * sizeof(float));
    if (!y->attn_norm_f || !y->ffn_norm_f || (f->qk_mode && (!y->qn_f || !y->kn_f)) || !y->router_f) { snprintf(err, errlen, "out of memory"); goto fail; }
    for (int e = 0; e < f->n_expert; e++) cl_dequant_row(y->router.type, y->router.p + (size_t)e * cl_row_bytes(y->router.type, (uint64_t)f->n_embd), y->router_f + (size_t)e * f->n_embd, f->n_embd);
  }
  size_t kvn = (size_t)f->n_layer * f->n_ctx_max * f->n_kvw;
  size_t E = (size_t)f->n_embd;
  f->kc = calloc(kvn, sizeof(float)); f->vc = calloc(kvn, sizeof(float));
  f->x = calloc(E, 4); f->xn = calloc(E, 4); f->q = calloc((size_t)f->n_q, 4); f->k = calloc((size_t)f->n_kvw, 4); f->v = calloc((size_t)f->n_kvw, 4); f->att = calloc((size_t)f->n_q, 4);
  f->scores = calloc((size_t)f->n_ctx_max, 4); f->moe = calloc(E, 4); f->hg = calloc((size_t)f->n_ff_exp * f->n_used, 4); f->hu = calloc((size_t)f->n_ff_exp * f->n_used, 4); f->hd = calloc(E * (size_t)f->n_used, 4);
  f->rl = calloc((size_t)f->n_expert, 4); f->rlogit = calloc((size_t)f->n_expert, 4);
  f->xq_ffk = calloc((size_t)f->n_used, sizeof(cl_xq));
  if (!f->xq_ffk || !cl_xq_init(&f->xq_e, f->n_embd) || !cl_xq_init(&f->xq_q, f->n_q) || !cl_xq_init(&f->xq_ff, f->n_ff_exp)) { snprintf(err, errlen, "out of memory"); goto fail; }
  for (int k = 0; k < f->n_used; k++) if (!cl_xq_init(&f->xq_ffk[k], f->n_ff_exp)) { snprintf(err, errlen, "out of memory"); goto fail; }
  if (!f->kc || !f->vc || !f->x || !f->xn || !f->q || !f->k || !f->v || !f->att || !f->scores || !f->moe || !f->hg || !f->hu || !f->hd || !f->rl || !f->rlogit) { snprintf(err, errlen, "out of memory"); goto fail; }
  { long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE);
    if ((!opts || !opts->cache_bytes) && pages > 0 && psz > 0 && f->map_size > (size_t)pages * (size_t)psz * 8 / 10) {
      snprintf(err, errlen, "the model file (%.1f GiB) is larger than 80%% of this machine's memory (%.1f GiB): mapping it whole would thrash; stream its experts with --cache-mb N", f->map_size / 1073741824.0, (double)pages * (double)psz / 1073741824.0); goto fail;
    } }
  if (opts && opts->smart.enabled && !opts->cache_bytes) { snprintf(err, errlen, "--smart needs --cache-mb: it substitutes experts that are already in the cache"); goto fail; }
  if (opts && opts->cache_bytes) {
    cl_xs_layer *xl = calloc((size_t)f->n_layer, sizeof *xl);
    if (!xl) { snprintf(err, errlen, "out of memory"); goto fail; }
    for (int l = 0; l < f->n_layer; l++) {
      const cl_fwd_layer *y = &f->L[l];
      xl[l].off[0] = (uint64_t)(y->gate.p - f->map); xl[l].slab[0] = y->gate_slab;
      xl[l].off[1] = (uint64_t)(y->up.p - f->map);   xl[l].slab[1] = y->up_slab;
      xl[l].off[2] = (uint64_t)(y->down.p - f->map); xl[l].slab[2] = y->down_slab;
    }
    int stage = opts->stage_slots > 0 ? opts->stage_slots : (opts->prefetch ? 2 * f->n_used : 0);
    f->xs = cl_xs_open(path, xl, f->n_layer, f->n_expert, f->n_used, opts->cache_bytes, stage, opts->policy, opts->io_threads > 0 ? opts->io_threads : 3, opts->os_bypass, err, errlen);
    free(xl);
    if (!f->xs) goto fail;
    f->prefetch = opts->prefetch;
    f->smart = opts->smart;
    f->prefetch_n = opts->prefetch_n > 0 && opts->prefetch_n < f->n_used ? opts->prefetch_n : f->n_used;
    f->prev_sel = malloc((size_t)f->n_layer * f->n_used * sizeof(int));
    if (!f->prev_sel) { snprintf(err, errlen, "out of memory"); goto fail; }
    f->pred_sel = malloc((size_t)f->n_layer * f->n_used * sizeof(int));
    if (!f->pred_sel) { snprintf(err, errlen, "out of memory"); goto fail; }
    for (int i = 0; i < f->n_layer * f->n_used; i++) f->prev_sel[i] = f->pred_sel[i] = -1;
  }
  return 0;
fail:
  cl_fwd_close(f);
  return 1;
}

void cl_fwd_close(cl_fwd *f) {
  if (f->L) for (int l = 0; l < f->n_layer; l++) { free(f->L[l].attn_norm_f); free(f->L[l].ffn_norm_f); free(f->L[l].qn_f); free(f->L[l].kn_f); free(f->L[l].router_f); }
  free(f->L); free(f->out_norm_f);
  free(f->kc); free(f->vc); free(f->x); free(f->xn); free(f->q); free(f->k); free(f->v); free(f->att); free(f->scores); free(f->moe); free(f->hg); free(f->hu); free(f->hd); free(f->rl); free(f->rlogit);
  cl_xq_free(&f->xq_e); cl_xq_free(&f->xq_q); cl_xq_free(&f->xq_ff);
  if (f->xq_ffk) for (int k = 0; k < f->n_used; k++) cl_xq_free(&f->xq_ffk[k]);
  free(f->xq_ffk);
  cl_xs_close(f->xs); free(f->prev_sel); free(f->pred_sel);
  cl_meta_free(&f->meta);
  if (f->map) munmap(f->map, f->map_size);
  cl_model_close(&f->model);
  memset(f, 0, sizeof *f);
}

void cl_fwd_reset(cl_fwd *f) { f->pos = 0; }

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
  double ss = 0;
  for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
  float s = 1.0f / sqrtf((float)(ss / n) + eps);
  for (int i = 0; i < n; i++) out[i] = x[i] * s * w[i];
}

/* llama.cpp's NORM rope: rotate adjacent pairs (v[2i], v[2i+1]) */
static void rope_pairs(float *v, int hd, int pos, float base) {
  for (int i = 0; i < hd / 2; i++) {
    float theta = (float)pos * powf(base, -2.0f * (float)i / (float)hd), c = cosf(theta), s = sinf(theta);
    float a = v[2 * i], b = v[2 * i + 1];
    v[2 * i] = a * c - b * s; v[2 * i + 1] = a * s + b * c;
  }
}

static void rope_neox(float *v, int hd, int pos, float base) {
  int half = hd / 2;
  for (int i = 0; i < half; i++) {
    float theta = (float)pos * powf(base, -2.0f * (float)i / (float)hd), c = cosf(theta), s = sinf(theta);
    float a = v[i], b = v[i + half];
    v[i] = a * c - b * s; v[i + half] = a * s + b * c;
  }
}

static cl_mv mv(const cl_w *w, const float *xf, const cl_xq *xq, float *out) { cl_mv j = {w->type, w->p, w->rows, w->cols, xf, xq, out}; return j; }

static int resident_cb(void *ctx, int expert) {
  struct { cl_xs *xs; int layer; } *c = ctx;
  return cl_xs_resident(c->xs, c->layer, expert);
}

int cl_fwd_step(cl_fwd *f, int token, float *logits, cl_fwd_route *route, char *err, size_t errlen) {
  if (token < 0 || token >= f->n_vocab) { snprintf(err, errlen, "token %d is outside the vocabulary", token); return 1; }
  if (f->pos >= f->n_ctx_max) { snprintf(err, errlen, "the context is full (%d tokens)", f->n_ctx_max); return 1; }
  const int E = f->n_embd, H = f->n_head, KVH = f->n_head_kv, hd = f->head_dim, pos = f->pos, KW = f->n_kvw, GROUP = f->n_head / f->n_head_kv;
  cl_dequant_row(f->tok_embd.type, f->tok_embd.p + (size_t)token * cl_row_bytes(f->tok_embd.type, (uint64_t)E), f->x, E);
  if (f->embed_scale != 1.0f) for (int i = 0; i < E; i++) f->x[i] *= f->embed_scale;
  if (f->pred_sel) for (int i = 0; i < f->n_layer * f->n_used; i++) f->pred_sel[i] = -1;
  if (f->xs && f->prefetch) for (int k = 0; k < f->prefetch_n; k++) { cl_xs_prefetch(f->xs, 0, f->prev_sel[k]); f->pred_sel[k] = f->prev_sel[k]; }   /* layer 0, as the previous token chose it (nothing else is known yet) */
  for (int l = 0; l < f->n_layer; l++) {
    cl_fwd_layer *y = &f->L[l];
    /* while this layer's attention and experts compute, read the experts the NEXT layer chose for the previous token */
    if (f->xs && f->prefetch == 1 && l + 1 < f->n_layer) for (int k = 0; k < f->prefetch_n; k++) { int e = f->prev_sel[(l + 1) * f->n_used + k]; cl_xs_prefetch(f->xs, l + 1, e); f->pred_sel[(l + 1) * f->n_used + k] = e; }
    rmsnorm(f->xn, f->x, y->attn_norm_f, E, f->rms_eps);
    cl_xq_quantize(&f->xq_e, f->xn);
    { cl_mv j[3] = {mv(&y->wq, f->xn, &f->xq_e, f->q), mv(&y->wk, f->xn, &f->xq_e, f->k), mv(&y->wv, f->xn, &f->xq_e, f->v)}; cl_matvec_batch(j, 3); }
    if (f->qk_mode == 0) { /* granitemoe: no QK-norm */ }
    else if (f->qk_per_head) {   /* qwen3moe: each head's vector is normalised on its own, with one weight shared by all heads */
      for (int h = 0; h < H; h++) rmsnorm(f->q + h * hd, f->q + h * hd, y->qn_f, hd, f->rms_eps);
      for (int h = 0; h < KVH; h++) rmsnorm(f->k + h * hd, f->k + h * hd, y->kn_f, hd, f->rms_eps);
    } else {   /* olmoe: over the whole projected vector, before it is split into heads */
      rmsnorm(f->q, f->q, y->qn_f, f->n_q, f->rms_eps);
      rmsnorm(f->k, f->k, y->kn_f, f->n_kvw, f->rms_eps);
    }
    for (int h = 0; h < H; h++) (f->rope_pairs ? rope_pairs : rope_neox)(f->q + h * hd, hd, pos, f->rope_base);
    for (int h = 0; h < KVH; h++) (f->rope_pairs ? rope_pairs : rope_neox)(f->k + h * hd, hd, pos, f->rope_base);
    float *kc = f->kc + ((size_t)l * f->n_ctx_max + pos) * KW, *vc = f->vc + ((size_t)l * f->n_ctx_max + pos) * KW;
    memcpy(kc, f->k, (size_t)KW * 4); memcpy(vc, f->v, (size_t)KW * 4);
    const float scale = f->attn_scale;
    for (int h = 0; h < H; h++) {
      const float *qh = f->q + h * hd;
      const int kvo = (h / GROUP) * hd;   /* grouped-query attention: GROUP query heads share one key/value head */
      float mx = -INFINITY;
      for (int t = 0; t <= pos; t++) {
        const float *kt = f->kc + ((size_t)l * f->n_ctx_max + t) * KW + kvo;
        float s = 0;
        for (int i = 0; i < hd; i++) s += qh[i] * kt[i];
        f->scores[t] = s * scale;
        if (f->scores[t] > mx) mx = f->scores[t];
      }
      float sum = 0;
      for (int t = 0; t <= pos; t++) { f->scores[t] = expf(f->scores[t] - mx); sum += f->scores[t]; }
      float *oh = f->att + h * hd;
      memset(oh, 0, (size_t)hd * 4);
      for (int t = 0; t <= pos; t++) {
        const float *vt = f->vc + ((size_t)l * f->n_ctx_max + t) * KW + kvo;
        float p = f->scores[t] / sum;
        for (int i = 0; i < hd; i++) oh[i] += p * vt[i];
      }
    }
    cl_xq_quantize(&f->xq_q, f->att);
    { cl_mv j = mv(&y->wo, f->att, &f->xq_q, f->hd); cl_matvec_batch(&j, 1); }
    for (int i = 0; i < E; i++) f->x[i] += f->resid_scale * f->hd[i];

    /* router-ahead: apply the NEXT layer's router to this layer's hidden state (the residual stream changes slowly) and start reading what it picks,
     * so the reads overlap this layer's experts */
    if (f->xs && f->prefetch == 2 && l + 1 < f->n_layer) {
      const cl_fwd_layer *nx = &f->L[l + 1];
      rmsnorm(f->xn, f->x, nx->ffn_norm_f, E, f->rms_eps);
      for (int e = 0; e < f->n_expert; e++) { const float *r = nx->router_f + (size_t)e * E; float a = 0; for (int i = 0; i < E; i++) a += r[i] * f->xn[i]; f->rl[e] = a; }
      char taken[4096] = {0};
      for (int k = 0; k < f->prefetch_n; k++) {
        int best = -1;
        for (int e = 0; e < f->n_expert; e++) if (!taken[e] && (best < 0 || f->rl[e] > f->rl[best])) best = e;
        taken[best] = 1; cl_xs_prefetch(f->xs, l + 1, best); f->pred_sel[(l + 1) * f->n_used + k] = best;
      }
    }

    /* mixture of experts */
    rmsnorm(f->xn, f->x, y->ffn_norm_f, E, f->rms_eps);
    for (int e = 0; e < f->n_expert; e++) {
      const float *r = y->router_f + (size_t)e * E;
      float s = 0;
      for (int i = 0; i < E; i++) s += r[i] * f->xn[i];
      f->rl[e] = s;
    }
    memcpy(f->rlogit, f->rl, (size_t)f->n_expert * sizeof(float));
    if (route && route->logits) memcpy(route->logits + (size_t)l * f->n_expert, f->rl, (size_t)f->n_expert * sizeof(float));
    float mx = f->rl[0];
    for (int e = 1; e < f->n_expert; e++) if (f->rl[e] > mx) mx = f->rl[e];
    double sum = 0;
    for (int e = 0; e < f->n_expert; e++) { f->rl[e] = expf(f->rl[e] - mx); sum += f->rl[e]; }
    for (int e = 0; e < f->n_expert; e++) f->rl[e] = (float)(f->rl[e] / sum);
    int sel[64];
    float wt[64];
    if (f->smart.enabled) {
      /* SMART-ROUTE: the true top-J stay; a slot that would need an expert that is not in memory may take a resident one whose router probability is within tau.
       * A substituted expert is weighted by its OWN router probability. */
      struct { cl_xs *xs; int layer; } rc = {f->xs, l};
      cl_route_result rr;
      if (cl_route(f->rlogit, f->n_expert, f->n_used, &f->smart, resident_cb, &rc, &rr)) { snprintf(err, errlen, "bad SMART-ROUTE settings"); return 1; }
      for (int k = 0; k < f->n_used; k++) { sel[k] = rr.selected[k]; wt[k] = f->rl[sel[k]]; }
      f->subs += (uint64_t)rr.substituted; f->dropped_mass += rr.dropped_mass;
      if (rr.max_delta > f->max_delta) f->max_delta = rr.max_delta;
    } else {
      char used[4096] = {0};
      for (int k = 0; k < f->n_used; k++) {   /* top-k by probability; ties go to the lower expert id */
        int best = -1;
        for (int e = 0; e < f->n_expert; e++) if (!used[e] && (best < 0 || f->rl[e] > f->rl[best])) best = e;
        used[best] = 1; sel[k] = best; wt[k] = f->rl[best];   /* OLMoE does not renormalise the chosen weights */
      }
    }
    if (f->pred_sel) for (int k = 0; k < f->prefetch_n; k++) {
      int pe = f->pred_sel[l * f->n_used + k];
      if (pe < 0) continue;
      f->pred_total++;
      for (int j = 0; j < f->n_used; j++) if (sel[j] == pe) { f->pred_hit++; break; }
    }
    if (f->norm_topk) { float z = 0; for (int k = 0; k < f->n_used; k++) z += wt[k]; if (z > 0) for (int k = 0; k < f->n_used; k++) wt[k] /= z; }   /* qwen3moe: the chosen weights sum to 1 */
    f->expert_slots += (uint64_t)f->n_used;
    if (route) for (int k = 0; k < f->n_used; k++) { route->experts[l * f->n_used + k] = sel[k]; route->weights[l * f->n_used + k] = wt[k]; }
    memset(f->moe, 0, (size_t)E * 4);
    cl_xq_quantize(&f->xq_e, f->xn);   /* every selected expert reads the same normalised input */
    int hnd[64];
    const uint8_t *P[64];
    if (f->xs) {   /* ask for what is already in memory first, so a miss cannot evict an expert this layer is about to use */
      for (int pass = 0; pass < 2; pass++) for (int k = 0; k < f->n_used; k++) {
        int here = cl_xs_resident(f->xs, l, sel[k]);
        if ((pass == 0) == (here != 0)) hnd[k] = cl_xs_request(f->xs, l, sel[k]);
      }
    }
    for (int k = 0; k < f->n_used; k++) {
      if (f->xs) {
        P[k] = cl_xs_wait(f->xs, hnd[k]);
        if (!P[k]) { snprintf(err, errlen, "reading expert %d of layer %d from disk failed", sel[k], l); for (int j = 0; j < f->n_used; j++) cl_xs_release(f->xs, hnd[j]); return 1; }
      } else P[k] = y->gate.p + (size_t)sel[k] * y->gate_slab;   /* mapped file: the three matrices sit in three fused tensors */
    }
    /* all experts' gate and up products in one batch, then all down products in another: two thread hand-offs per layer, not sixteen */
    cl_mv j[64];
    const size_t FF = (size_t)f->n_ff_exp;
    for (int k = 0; k < f->n_used; k++) {
      const uint8_t *g = f->xs ? P[k] : y->gate.p + (size_t)sel[k] * y->gate_slab, *u = f->xs ? P[k] + y->gate_slab : y->up.p + (size_t)sel[k] * y->up_slab;
      j[2 * k] = (cl_mv){y->gate.type, g, f->n_ff_exp, E, f->xn, &f->xq_e, f->hg + k * FF};
      j[2 * k + 1] = (cl_mv){y->up.type, u, f->n_ff_exp, E, f->xn, &f->xq_e, f->hu + k * FF};
    }
    cl_matvec_batch(j, 2 * f->n_used);
    for (int k = 0; k < f->n_used; k++) {
      float *hg = f->hg + k * FF, *hu = f->hu + k * FF;
      for (size_t i = 0; i < FF; i++) hg[i] = hg[i] / (1.0f + expf(-hg[i])) * hu[i];   /* silu(gate) * up */
      cl_xq_quantize(&f->xq_ffk[k], hg);
      const uint8_t *d = f->xs ? P[k] + y->gate_slab + y->up_slab : y->down.p + (size_t)sel[k] * y->down_slab;
      j[k] = (cl_mv){y->down.type, d, E, f->n_ff_exp, hg, &f->xq_ffk[k], f->hd + (size_t)k * E};
    }
    cl_matvec_batch(j, f->n_used);
    for (int k = 0; k < f->n_used; k++) {
      const float *hd = f->hd + (size_t)k * E;
      for (int i = 0; i < E; i++) f->moe[i] += wt[k] * hd[i];
      if (f->xs) cl_xs_release(f->xs, hnd[k]);
    }
    if (f->prev_sel) memcpy(f->prev_sel + l * f->n_used, sel, (size_t)f->n_used * sizeof(int));
    for (int i = 0; i < E; i++) f->x[i] += f->resid_scale * f->moe[i];
  }
  rmsnorm(f->xn, f->x, f->out_norm_f, E, f->rms_eps);
  cl_xq_quantize(&f->xq_e, f->xn);
  { cl_mv j = mv(&f->output, f->xn, &f->xq_e, logits); cl_matvec_batch(&j, 1); }
  if (f->logit_scale != 1.0f) for (int i = 0; i < f->n_vocab; i++) logits[i] /= f->logit_scale;
  f->pos++;
  return 0;
}
