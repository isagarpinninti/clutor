#define _GNU_SOURCE
#include "forward.h"
#include "quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "plat.h"

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
typedef struct { const char *name; int qk_mode, norm_topk, rope_pairs, expert_ff_key, scaled, dense, qkv_bias, hybrid; } arch_spec;
static const arch_spec ARCHS[] = {
  /* name        qk-norm  renorm  pair-rope  expert_feed_forward_length  granite multipliers  dense  qkv bias */
  {"olmoe",      1,       0,      0,         0,                          0,                   0,     0,        0},
  {"qwen3moe",   2,       1,      0,         1,                          0,                   0,     0,        0},
  {"granitemoe", 0,       1,      1,         0,                          1,                   0,     0,        0},
  {"qwen2",      0,       0,      0,         0,                          0,                   1,     1,        0},
  {"llama",      0,       0,      1,         0,                          0,                   1,     0,        0},
  {"qwen35",     2,       0,      0,         0,                          0,                   1,     0,        1},   /* hybrid: Gated DeltaNet + gated attention, dense feed-forward */
};

/* One layer of a qwen35 model: either a Gated DeltaNet layer (a causal convolution, a per-head gated delta rule, a gated norm) or a gated full-attention layer, then a plain SwiGLU feed-forward. */
static int bind_q35_layer(cl_fwd *f, int l, char *err, size_t errlen) {
  cl_fwd_layer *y = &f->L[l];
  const int N = f->n_embd, S = f->ssm_state, HV = f->ssm_heads, HK = f->ssm_groups, CK = f->ssm_conv, CC = 2 * S * HK + S * HV, FF = f->n_ff_exp, hd = f->head_dim;
  y->recurrent = (l + 1) % f->attn_interval != 0;
  if (bind(f, &y->attn_norm, "attn_norm.weight", l, 1, err, errlen) || bind(f, &y->ffn_norm, "post_attention_norm.weight", l, 1, err, errlen) ||
      bind(f, &y->gate, "ffn_gate.weight", l, 2, err, errlen) || bind(f, &y->up, "ffn_up.weight", l, 2, err, errlen) || bind(f, &y->down, "ffn_down.weight", l, 2, err, errlen)) return 1;
  if (y->attn_norm.cols != N || y->ffn_norm.cols != N || y->gate.rows != FF || y->gate.cols != N || y->up.rows != FF || y->up.cols != N || y->down.rows != N || y->down.cols != FF) {
    snprintf(err, errlen, "layer %d: feed-forward or norm shapes do not match the hyper-parameters", l); return 1;
  }
  y->attn_norm_f = to_float(&y->attn_norm, (size_t)N); y->ffn_norm_f = to_float(&y->ffn_norm, (size_t)N);
  if (!y->attn_norm_f || !y->ffn_norm_f) { snprintf(err, errlen, "out of memory"); return 1; }
  if (y->recurrent) {
    cl_w conv, a, dt, nm;
    if (bind(f, &y->attn_qkv, "attn_qkv.weight", l, 2, err, errlen) || bind(f, &y->attn_gate, "attn_gate.weight", l, 2, err, errlen) || bind(f, &y->ssm_alpha, "ssm_alpha.weight", l, 2, err, errlen) ||
        bind(f, &y->ssm_beta, "ssm_beta.weight", l, 2, err, errlen) || bind(f, &y->ssm_out, "ssm_out.weight", l, 2, err, errlen) || bind(f, &conv, "ssm_conv1d.weight", l, 2, err, errlen) ||
        bind(f, &a, "ssm_a", l, 1, err, errlen) || bind(f, &dt, "ssm_dt.bias", l, 1, err, errlen) || bind(f, &nm, "ssm_norm.weight", l, 1, err, errlen)) return 1;
    if (y->attn_qkv.rows != CC || y->attn_qkv.cols != N || y->attn_gate.rows != S * HV || y->attn_gate.cols != N || y->ssm_alpha.rows != HV || y->ssm_alpha.cols != N || y->ssm_beta.rows != HV || y->ssm_beta.cols != N ||
        y->ssm_out.rows != N || y->ssm_out.cols != S * HV || conv.cols != CK || conv.rows != CC || a.cols != HV || dt.cols != HV || nm.cols != S) {
      snprintf(err, errlen, "layer %d: Gated DeltaNet tensor shapes do not match the hyper-parameters", l); return 1;
    }
    if (conv.type != CL_T_F32 || a.type != CL_T_F32 || dt.type != CL_T_F32 || nm.type != CL_T_F32) { snprintf(err, errlen, "layer %d: the convolution, decay and norm tensors must be float32", l); return 1; }
    y->conv_f = to_float(&conv, (size_t)CK * CC); y->ssm_a_f = to_float(&a, (size_t)HV); y->ssm_dt_f = to_float(&dt, (size_t)HV); y->ssm_norm_f = to_float(&nm, (size_t)S);
    if (!y->conv_f || !y->ssm_a_f || !y->ssm_dt_f || !y->ssm_norm_f) { snprintf(err, errlen, "out of memory"); return 1; }
    y->rec_idx = f->n_rec++;
  } else {
    if (bind(f, &y->wq, "attn_q.weight", l, 2, err, errlen) || bind(f, &y->wk, "attn_k.weight", l, 2, err, errlen) || bind(f, &y->wv, "attn_v.weight", l, 2, err, errlen) ||
        bind(f, &y->wo, "attn_output.weight", l, 2, err, errlen) || bind(f, &y->qn, "attn_q_norm.weight", l, 1, err, errlen) || bind(f, &y->kn, "attn_k_norm.weight", l, 1, err, errlen)) return 1;
    if (y->wq.rows != 2 * f->n_q || y->wq.cols != N || y->wk.rows != f->n_kvw || y->wk.cols != N || y->wv.rows != f->n_kvw || y->wv.cols != N || y->wo.rows != N || y->wo.cols != f->n_q || y->qn.cols != hd || y->kn.cols != hd) {
      snprintf(err, errlen, "layer %d: gated attention tensor shapes do not match the hyper-parameters", l); return 1;
    }
    y->qn_f = to_float(&y->qn, (size_t)hd); y->kn_f = to_float(&y->kn, (size_t)hd);
    if (!y->qn_f || !y->kn_f) { snprintf(err, errlen, "out of memory"); return 1; }
    y->kv_idx = f->n_kv_layers++;
  }
  return 0;
}

#define AKEY(buf, suffix) snprintf(buf, sizeof buf, "%s.%s", f->arch, suffix)
#define REQ_U(suffix, dst) do { char k_[96]; uint64_t v_; AKEY(k_, suffix); if (!cl_meta_u64(&f->meta, k_, &v_)) { snprintf(err, errlen, "the model does not state %s", k_); goto fail; } dst = (int)v_; } while (0)

int cl_fwd_open(cl_fwd *f, const char *path, int n_ctx, const cl_fwd_opts *opts, char *err, size_t errlen) {
  memset(f, 0, sizeof *f);
  if (cl_model_open(&f->model, path, err, errlen)) return 1;
  const arch_spec *spec = NULL;
  for (size_t i = 0; i < sizeof ARCHS / sizeof *ARCHS; i++) if (!strcmp(f->model.arch, ARCHS[i].name)) spec = &ARCHS[i];
  if (!spec) { snprintf(err, errlen, "the forward pass supports the \"olmoe\", \"qwen3moe\", \"granitemoe\", \"qwen2\" (dense), \"llama\" (dense) and \"qwen35\" (hybrid) architectures, not \"%s\"", f->model.arch); goto fail; }
  snprintf(f->arch, sizeof f->arch, "%s", spec->name);
  f->qk_mode = spec->qk_mode; f->qk_per_head = spec->qk_mode == 2; f->norm_topk = spec->norm_topk; f->rope_pairs = spec->rope_pairs; f->dense = spec->dense; f->qkv_bias = spec->qkv_bias; f->hybrid = spec->hybrid;
  const int NU = spec->dense ? 1 : 0;   /* a dense model has no experts, but its single feed-forward needs the same scratch as one expert */
  f->embed_scale = f->resid_scale = f->logit_scale = 1.0f; f->attn_scale = 0;
  int fd = cl_open_ro(path);
  if (fd < 0 || cl_fd_size(fd) == 0) { snprintf(err, errlen, "cannot open %s", path); if (fd >= 0) cl_close(fd); goto fail; }
  f->map_size = (size_t)cl_fd_size(fd);
  f->map = cl_map_ro(fd, f->map_size);
  cl_close(fd);
  if (!f->map) { snprintf(err, errlen, "cannot map %s", path); goto fail; }
  if (opts && opts->lock && !opts->cache_bytes) f->locked = cl_lock_memory(f->map, f->map_size);   /* reads the whole file in, so it can take seconds; failing just leaves the model evictable */
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
  if (f->n_embd < 256 || f->n_head < 1 || f->n_head > f->n_embd || f->n_embd % f->n_head || f->n_layer < 1 || (!f->dense && (f->n_expert < 1 || f->n_used < 1 || f->n_used > f->n_expert)) || f->n_ff_exp < 1) {
    snprintf(err, errlen, "implausible hyper-parameters (embedding %d, heads %d, layers %d, experts %d, used %d, expert width %d)", f->n_embd, f->n_head, f->n_layer, f->n_expert, f->n_used, f->n_ff_exp); goto fail;
  }
  f->head_dim = f->n_embd / f->n_head;
  { char k[96]; uint64_t v_; AKEY(k, "attention.key_length"); if (cl_meta_u64(&f->meta, k, &v_) && v_ >= 2 && v_ <= 1024 && v_ % 2 == 0) f->head_dim = (int)v_; }   /* qwen3moe: 128, not n_embd / n_head */
  if (f->n_head_kv < 1 || f->n_head_kv > f->n_head || f->n_head % f->n_head_kv) { snprintf(err, errlen, "implausible head counts (%d query heads, %d key/value heads)", f->n_head, f->n_head_kv); goto fail; }
  f->rope_dim = f->head_dim;
  { char k[96]; uint64_t v_; AKEY(k, "rope.dimension_count");
    if (cl_meta_u64(&f->meta, k, &v_) && (int)v_ != f->head_dim) {
      if (!f->hybrid || v_ < 2 || (int)v_ > f->head_dim || v_ % 2) { snprintf(err, errlen, "partial rotary embeddings (%d of %d dimensions) are not implemented for this architecture", (int)v_, f->head_dim); goto fail; }
      f->rope_dim = (int)v_;   /* qwen35 rotates the first 64 of each head's 256 dimensions */
    } }
  if (f->hybrid) {
    REQ_U("ssm.conv_kernel", f->ssm_conv); REQ_U("ssm.state_size", f->ssm_state); REQ_U("ssm.group_count", f->ssm_groups); REQ_U("ssm.time_step_rank", f->ssm_heads); REQ_U("ssm.inner_size", f->ssm_inner);
    f->attn_interval = 4;
    { char k[96]; uint64_t v_; AKEY(k, "full_attention_interval"); if (cl_meta_u64(&f->meta, k, &v_)) f->attn_interval = (int)v_; }
    if (f->ssm_conv < 2 || f->ssm_conv > 8 || f->ssm_state < 8 || f->ssm_state > 512 || f->ssm_groups < 1 || f->ssm_heads < f->ssm_groups || f->ssm_heads % f->ssm_groups || f->ssm_inner != f->ssm_heads * f->ssm_state || f->attn_interval < 1 || f->attn_interval > f->n_layer) {
      snprintf(err, errlen, "implausible Gated DeltaNet hyper-parameters (conv %d, state %d, groups %d, heads %d, inner %d, attention every %d layers)", f->ssm_conv, f->ssm_state, f->ssm_groups, f->ssm_heads, f->ssm_inner, f->attn_interval); goto fail;
    }
  }
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
  if (!f->dense && (f->n_used > 64 || f->n_expert > 4096)) { snprintf(err, errlen, "unsupported expert counts"); goto fail; }
  f->n_ctx_train = f->n_ctx_max;   /* what the model was trained for; the cache below is sized to what was asked, never more */
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
  if (find(&f->model, "rope_freqs.weight")) {   /* llama 3: per-pair divisors of the rotary frequencies */
    cl_w rf;
    if (bind(f, &rf, "rope_freqs.weight", -1, 1, err, errlen)) goto fail;
    if (rf.type != CL_T_F32 || rf.cols != f->head_dim / 2) { snprintf(err, errlen, "rope_freqs.weight must be %d float32 values", f->head_dim / 2); goto fail; }
    f->rope_factors = to_float(&rf, (size_t)rf.cols);
    if (!f->rope_factors) { snprintf(err, errlen, "out of memory"); goto fail; }
    for (int i = 0; i < rf.cols; i++) if (!(f->rope_factors[i] > 0.0f)) { snprintf(err, errlen, "rope_freqs.weight holds a non-positive factor"); goto fail; }
  }
  f->L = calloc((size_t)f->n_layer, sizeof *f->L);
  if (!f->L || !f->out_norm_f) { snprintf(err, errlen, "out of memory"); goto fail; }
  for (int l = 0; l < f->n_layer; l++) {
    cl_fwd_layer *y = &f->L[l];
    if (f->hybrid) { if (bind_q35_layer(f, l, err, errlen)) goto fail; continue; }
    if (bind(f, &y->attn_norm, "attn_norm.weight", l, 1, err, errlen) || bind(f, &y->wq, "attn_q.weight", l, 2, err, errlen) || bind(f, &y->wk, "attn_k.weight", l, 2, err, errlen) ||
        bind(f, &y->wv, "attn_v.weight", l, 2, err, errlen) || bind(f, &y->wo, "attn_output.weight", l, 2, err, errlen) || (f->qk_mode && bind(f, &y->qn, "attn_q_norm.weight", l, 1, err, errlen)) ||
        (f->qk_mode && bind(f, &y->kn, "attn_k_norm.weight", l, 1, err, errlen)) || bind(f, &y->ffn_norm, "ffn_norm.weight", l, 1, err, errlen)) goto fail;
    if (f->dense) {
      if (bind(f, &y->gate, "ffn_gate.weight", l, 2, err, errlen) || bind(f, &y->up, "ffn_up.weight", l, 2, err, errlen) || bind(f, &y->down, "ffn_down.weight", l, 2, err, errlen)) goto fail;
    } else if (bind(f, &y->router, "ffn_gate_inp.weight", l, 2, err, errlen) || bind(f, &y->gate, "ffn_gate_exps.weight", l, 3, err, errlen) || bind(f, &y->up, "ffn_up_exps.weight", l, 3, err, errlen) || bind(f, &y->down, "ffn_down_exps.weight", l, 3, err, errlen)) goto fail;
    if (f->qkv_bias && (bind(f, &y->bq, "attn_q.bias", l, 1, err, errlen) || bind(f, &y->bk, "attn_k.bias", l, 1, err, errlen) || bind(f, &y->bv, "attn_v.bias", l, 1, err, errlen))) goto fail;
    const int N = f->n_embd;
    const int QN = f->qk_per_head ? f->head_dim : f->n_q, KN = f->qk_per_head ? f->head_dim : f->n_kvw;
    if (y->attn_norm.cols != N || y->ffn_norm.cols != N || (f->qk_mode && (y->qn.cols != QN || y->kn.cols != KN)) ||
        y->wq.rows != f->n_q || y->wq.cols != N || y->wk.rows != f->n_kvw || y->wk.cols != N || y->wv.rows != f->n_kvw || y->wv.cols != N || y->wo.rows != N || y->wo.cols != f->n_q ||
        y->up.rows != f->n_ff_exp || y->up.cols != N || y->gate.cols != N || (f->qkv_bias && (y->bq.cols != f->n_q || y->bk.cols != f->n_kvw || y->bv.cols != f->n_kvw))) {
      snprintf(err, errlen, "layer %d: attention or norm tensor shapes do not match the embedding width %d", l, N); goto fail;
    }
    if (f->dense) {
      if (y->gate.rows != f->n_ff_exp || y->down.rows != f->n_embd || y->down.cols != f->n_ff_exp) { snprintf(err, errlen, "layer %d: feed-forward shapes do not match the hyper-parameters", l); goto fail; }
    } else {
      char nm[200];
      snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", l);
      const cl_tensor *tg = find(&f->model, nm);
      if (y->router.rows != f->n_expert || y->router.cols != f->n_embd || (int)tg->dims[2] != f->n_expert || y->up.cols != N || y->gate.rows != f->n_ff_exp || y->down.rows != f->n_embd || y->down.cols != f->n_ff_exp) {
        snprintf(err, errlen, "layer %d: tensor shapes do not match the hyper-parameters", l); goto fail;
      }
      y->gate_slab = (uint64_t)y->gate.rows * cl_row_bytes(y->gate.type, (uint64_t)y->gate.cols);
      y->up_slab = (uint64_t)y->up.rows * cl_row_bytes(y->up.type, (uint64_t)y->up.cols);
      y->down_slab = (uint64_t)y->down.rows * cl_row_bytes(y->down.type, (uint64_t)y->down.cols);
    }
    y->attn_norm_f = to_float(&y->attn_norm, (size_t)f->n_embd); y->ffn_norm_f = to_float(&y->ffn_norm, (size_t)f->n_embd);
    if (f->qk_mode) { y->qn_f = to_float(&y->qn, (size_t)QN); y->kn_f = to_float(&y->kn, (size_t)KN); }
    if (f->qkv_bias) { y->bq_f = to_float(&y->bq, (size_t)f->n_q); y->bk_f = to_float(&y->bk, (size_t)f->n_kvw); y->bv_f = to_float(&y->bv, (size_t)f->n_kvw); }
    if (!f->dense) y->router_f = malloc((size_t)f->n_expert * f->n_embd * sizeof(float));
    if (!y->attn_norm_f || !y->ffn_norm_f || (f->qk_mode && (!y->qn_f || !y->kn_f)) || (!f->dense && !y->router_f) || (f->qkv_bias && (!y->bq_f || !y->bk_f || !y->bv_f))) { snprintf(err, errlen, "out of memory"); goto fail; }
    for (int e = 0; e < f->n_expert; e++) cl_dequant_row(y->router.type, y->router.p + (size_t)e * cl_row_bytes(y->router.type, (uint64_t)f->n_embd), y->router_f + (size_t)e * f->n_embd, f->n_embd);
  }
  if (!f->hybrid) f->n_kv_layers = f->n_layer;
  size_t kvn = (size_t)f->n_kv_layers * f->n_ctx_max * f->n_kvw;
  size_t E = (size_t)f->n_embd;
  f->kc = calloc(kvn, sizeof(float)); f->vc = calloc(kvn, sizeof(float));
  f->x = calloc(E, 4); f->xn = calloc(E, 4); f->q = calloc((size_t)f->n_q, 4); f->k = calloc((size_t)f->n_kvw, 4); f->v = calloc((size_t)f->n_kvw, 4); f->att = calloc((size_t)f->n_q, 4);
  f->scores = calloc((size_t)f->n_ctx_max, 4); f->moe = calloc(E, 4); f->hg = calloc((size_t)f->n_ff_exp * (size_t)(f->n_used + NU), 4); f->hu = calloc((size_t)f->n_ff_exp * (size_t)(f->n_used + NU), 4); f->hd = calloc(E * (size_t)(f->n_used + NU), 4);
  f->rl = calloc((size_t)f->n_expert + 1, 4); f->rlogit = calloc((size_t)f->n_expert + 1, 4);
  f->xq_ffk = calloc((size_t)f->n_used + 1, sizeof(cl_xq));
  if (!f->xq_ffk || !cl_xq_init(&f->xq_e, f->n_embd) || !cl_xq_init(&f->xq_q, f->n_q) || !cl_xq_init(&f->xq_ff, f->n_ff_exp)) { snprintf(err, errlen, "out of memory"); goto fail; }
  for (int k = 0; k < f->n_used; k++) if (!cl_xq_init(&f->xq_ffk[k], f->n_ff_exp)) { snprintf(err, errlen, "out of memory"); goto fail; }
  if (f->hybrid) {
    const int CC = 2 * f->ssm_state * f->ssm_groups + f->ssm_state * f->ssm_heads;
    f->conv_state = calloc((size_t)f->n_rec * CC * (f->ssm_conv - 1), 4); f->ssm_st = calloc((size_t)f->n_rec * f->ssm_heads * f->ssm_state * f->ssm_state, 4);
    f->qg = calloc((size_t)2 * f->n_q, 4); f->qkv = calloc((size_t)CC, 4); f->cvo = calloc((size_t)CC, 4); f->zbuf = calloc((size_t)f->ssm_inner, 4); f->ab = calloc((size_t)2 * f->ssm_heads, 4);
    { const size_t B = CL_PF_B, AS = (size_t)(f->ssm_inner > f->n_q ? f->ssm_inner : f->n_q);
      f->pf_x = calloc(B * f->n_embd, 4); f->pf_xn = calloc(B * f->n_embd, 4); f->pf_qkv = calloc(B * CC, 4); f->pf_z = calloc(B * f->ssm_inner, 4); f->pf_ab = calloc(B * 2 * f->ssm_heads, 4);
      f->pf_qg = calloc(B * 2 * f->n_q, 4); f->pf_k = calloc(B * f->n_kvw, 4); f->pf_v = calloc(B * f->n_kvw, 4); f->pf_att = calloc(B * AS, 4); f->pf_hd = calloc(B * f->n_embd, 4);
      f->pf_hg = calloc(B * f->n_ff_exp, 4); f->pf_hu = calloc(B * f->n_ff_exp, 4);
      f->pf_xqe = calloc(B, sizeof(cl_xq)); f->pf_xqq = calloc(B, sizeof(cl_xq)); f->pf_xqss = calloc(B, sizeof(cl_xq)); f->pf_xqff = calloc(B, sizeof(cl_xq));
      int ok = f->pf_x && f->pf_xn && f->pf_qkv && f->pf_z && f->pf_ab && f->pf_qg && f->pf_k && f->pf_v && f->pf_att && f->pf_hd && f->pf_hg && f->pf_hu && f->pf_xqe && f->pf_xqq && f->pf_xqss && f->pf_xqff;
      for (size_t i = 0; ok && i < B; i++) ok = cl_xq_init(&f->pf_xqe[i], f->n_embd) && cl_xq_init(&f->pf_xqq[i], f->n_q) && cl_xq_init(&f->pf_xqss[i], f->ssm_inner) && cl_xq_init(&f->pf_xqff[i], f->n_ff_exp);
      if (!ok) { snprintf(err, errlen, "out of memory"); goto fail; } }
    if (!f->conv_state || !f->ssm_st || !f->qg || !f->qkv || !f->cvo || !f->zbuf || !f->ab || !cl_xq_init(&f->xq_ss, f->ssm_inner)) { snprintf(err, errlen, "out of memory"); goto fail; }
  }
  if (!f->kc || !f->vc || !f->x || !f->xn || !f->q || !f->k || !f->v || !f->att || !f->scores || !f->moe || !f->hg || !f->hu || !f->hd || !f->rl || !f->rlogit) { snprintf(err, errlen, "out of memory"); goto fail; }
  if (f->dense && opts && (opts->cache_bytes || opts->smart.enabled)) { snprintf(err, errlen, "this model is dense (it has no experts), so there is nothing to stream or substitute: --cache-mb and --smart apply to mixture-of-experts models only"); goto fail; }
  { const uint64_t ram = cl_phys_mem();
    if ((!opts || !opts->cache_bytes) && ram > 0 && f->map_size > (size_t)(ram / 10 * 8)) {
      snprintf(err, errlen, "the model file (%.1f GiB) is larger than 80%% of this machine's memory (%.1f GiB): mapping it whole would thrash; %s", f->map_size / 1073741824.0, (double)ram / 1073741824.0, f->dense ? "a dense model has no experts to stream, so it needs enough memory for the whole file" : "stream its experts with --cache-mb N"); goto fail;
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
  if (f->L) for (int l = 0; l < f->n_layer; l++) { free(f->L[l].attn_norm_f); free(f->L[l].ffn_norm_f); free(f->L[l].bq_f); free(f->L[l].bk_f); free(f->L[l].bv_f); free(f->L[l].qn_f); free(f->L[l].kn_f); free(f->L[l].router_f); free(f->L[l].ssm_a_f); free(f->L[l].ssm_dt_f); free(f->L[l].ssm_norm_f); free(f->L[l].conv_f); }
  free(f->L); free(f->out_norm_f); free(f->rope_factors); free(f->conv_state); free(f->ssm_st); free(f->qg); free(f->qkv); free(f->cvo); free(f->zbuf); free(f->ab);
  if (f->hybrid) {
    cl_xq_free(&f->xq_ss);
    for (int i = 0; i < CL_PF_B; i++) { if (f->pf_xqe) cl_xq_free(&f->pf_xqe[i]); if (f->pf_xqq) cl_xq_free(&f->pf_xqq[i]); if (f->pf_xqss) cl_xq_free(&f->pf_xqss[i]); if (f->pf_xqff) cl_xq_free(&f->pf_xqff[i]); }
    free(f->pf_xqe); free(f->pf_xqq); free(f->pf_xqss); free(f->pf_xqff);
    free(f->pf_x); free(f->pf_xn); free(f->pf_qkv); free(f->pf_z); free(f->pf_ab); free(f->pf_qg); free(f->pf_k); free(f->pf_v); free(f->pf_att); free(f->pf_hd); free(f->pf_hg); free(f->pf_hu);
  }
  if (f->ps_ready) {
    for (int i = 0; i < f->ps_cap; i++) { cl_xq_free(&f->ps_xqe[i]); cl_xq_free(&f->ps_xqq[i]); cl_xq_free(&f->ps_xqff[i]); }
    free(f->ps_xqe); free(f->ps_xqq); free(f->ps_xqff);
    free(f->ps_x); free(f->ps_xn); free(f->ps_xg); free(f->ps_q); free(f->ps_k); free(f->ps_v); free(f->ps_att); free(f->ps_hd); free(f->ps_moe); free(f->ps_hg); free(f->ps_hu); free(f->ps_rg); free(f->ps_wt); free(f->ps_hde); free(f->ps_sel);
  }
  free(f->logits_buf); free(f->kc); free(f->vc); free(f->x); free(f->xn); free(f->q); free(f->k); free(f->v); free(f->att); free(f->scores); free(f->moe); free(f->hg); free(f->hu); free(f->hd); free(f->rl); free(f->rlogit);
  cl_xq_free(&f->xq_e); cl_xq_free(&f->xq_q); cl_xq_free(&f->xq_ff);
  if (f->xq_ffk) for (int k = 0; k < f->n_used; k++) cl_xq_free(&f->xq_ffk[k]);
  free(f->xq_ffk);
  cl_xs_close(f->xs); free(f->prev_sel); free(f->pred_sel);
  cl_meta_free(&f->meta);
  if (f->map) cl_unmap(f->map, f->map_size);
  cl_model_close(&f->model);
  memset(f, 0, sizeof *f);
}

void cl_fwd_reset(cl_fwd *f) {
  f->pos = 0;
  if (f->hybrid) {   /* the recurrent state is the conversation: forgetting it means zeroing it */
    const size_t CC = (size_t)(2 * f->ssm_state * f->ssm_groups + f->ssm_state * f->ssm_heads);
    memset(f->conv_state, 0, (size_t)f->n_rec * CC * (size_t)(f->ssm_conv - 1) * 4);
    memset(f->ssm_st, 0, (size_t)f->n_rec * f->ssm_heads * f->ssm_state * f->ssm_state * 4);
  }
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
  double ss = 0;
  for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
  float s = 1.0f / sqrtf((float)(ss / n) + eps);
  for (int i = 0; i < n; i++) out[i] = x[i] * s * w[i];
}

/* llama.cpp's NORM rope: rotate adjacent pairs (v[2i], v[2i+1]) */
static void rope_pairs(float *v, int hd, int pos, float base, const float *ff) {
  for (int i = 0; i < hd / 2; i++) {
    float theta = (float)pos * powf(base, -2.0f * (float)i / (float)hd) / (ff ? ff[i] : 1.0f), c = cosf(theta), s = sinf(theta);
    float a = v[2 * i], b = v[2 * i + 1];
    v[2 * i] = a * c - b * s; v[2 * i + 1] = a * s + b * c;
  }
}

static void rope_neox(float *v, int hd, int pos, float base, const float *ff) {
  int half = hd / 2;
  for (int i = 0; i < half; i++) {
    float theta = (float)pos * powf(base, -2.0f * (float)i / (float)hd) / (ff ? ff[i] : 1.0f), c = cosf(theta), s = sinf(theta);
    float a = v[i], b = v[i + half];
    v[i] = a * c - b * s; v[i + half] = a * s + b * c;
  }
}

static cl_mv mv(const cl_w *w, const float *xf, const cl_xq *xq, float *out) { cl_mv j = {w->type, w->p, w->rows, w->cols, xf, xq, out, 1, 0, 0}; return j; }

static int resident_cb(void *ctx, int expert) {
  struct { cl_xs *xs; int layer; } *c = ctx;
  return cl_xs_resident(c->xs, c->layer, expert);
}

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static float siluf(float x) { return x / (1.0f + expf(-x)); }
static float softplusf(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }   /* ggml's softplus */

/* x / sqrt(sum(x^2) + eps): what llama.cpp's build_gdn_l2_norm computes (rms_norm with eps / n, then scaled by 1 / sqrt(n)). */
static void l2norm(float *x, int n, float eps) {
  double ss = 0;
  for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
  float s = 1.0f / sqrtf((float)ss + eps);
  for (int i = 0; i < n; i++) x[i] *= s;
}

/* ---- Attention for one head of one token, as a task for the thread pool ---------------------------------------------------------------------------------------------- */
typedef struct { cl_fwd *f; int l; const float *Q; float *ATT; int pos0; } attn_job;
static _Thread_local float *tl_scores;
static _Thread_local int tl_scores_cap;
static float *scores_for(int need) {
  if (tl_scores_cap < need) {
    free(tl_scores);
    tl_scores = (float *)malloc((size_t)need * sizeof(float));
    if (!tl_scores) { fprintf(stderr, "out of memory (attention scratch)\n"); abort(); }
    tl_scores_cap = need;
  }
  return tl_scores;
}
/* Task i is head i % n_head of token i / n_head (token 0 is the one at position pos0). Reads the key/value cache up to that token's own position, so the tasks do not depend on each other. */
static void attn_task(void *vp, int i) {
  const attn_job *a = (const attn_job *)vp;
  const cl_fwd *f = a->f;
  const int H = f->n_head, hd = f->head_dim, KW = f->n_kvw, GROUP = H / f->n_head_kv, NQ = f->n_q;
  const int b = i / H, h = i % H, pos = a->pos0 + b;
  const float *qh = a->Q + (size_t)b * NQ + (size_t)h * hd;
  float *oh = a->ATT + (size_t)b * NQ + (size_t)h * hd;
  float *sc = scores_for(pos + 1);
  const int kvo = (h / GROUP) * hd;
  float mx = -INFINITY;
  for (int t = 0; t <= pos; t++) {
    const float *kt = f->kc + ((size_t)a->l * f->n_ctx_max + t) * KW + kvo;
    float s = 0;
    for (int d = 0; d < hd; d++) s += qh[d] * kt[d];
    sc[t] = s * f->attn_scale;
    if (sc[t] > mx) mx = sc[t];
  }
  float sum = 0;
  for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
  memset(oh, 0, (size_t)hd * 4);
  for (int t = 0; t <= pos; t++) {
    const float *vt = f->vc + ((size_t)a->l * f->n_ctx_max + t) * KW + kvo;
    const float p = sc[t] / sum;
    for (int d = 0; d < hd; d++) oh[d] += p * vt[d];
  }
}
/* nb tokens' attention, in parallel when there is enough work to be worth waking the threads (a short context with few heads is faster on one thread). */
static void attend(cl_fwd *f, int l, const float *Q, float *ATT, int pos0, int nb) {
  attn_job a = {f, l, Q, ATT, pos0};
  const int n = nb * f->n_head;
  if ((long)(pos0 + nb) * n >= 6000) cl_parallel_for(n, attn_task, &a);
  else for (int i = 0; i < n; i++) attn_task(&a, i);
}

/* A product of one weight matrix with nb inputs at once: xf rows are `in` floats apart, results `rows` apart. */
static cl_mv mvb(const cl_w *w, const float *xf, size_t in, const cl_xq *xq, float *out, int nb) { cl_mv j = {w->type, w->p, w->rows, w->cols, xf, xq, out, nb, in, (size_t)w->rows}; return j; }

/* The sequential part of a Gated DeltaNet layer for ONE token (the state evolves token by token, so a batch walks through here in order): the causal
 * convolution over the concatenated query / key / value projections, the delta rule on every value head's state matrix, and the gated RMS-norm. */
static void deltanet_token(cl_fwd *f, const cl_fwd_layer *y, const float *qkv, const float *z, float *ab, float *out) {
  const int S = f->ssm_state, HK = f->ssm_groups, HV = f->ssm_heads, CK = f->ssm_conv, KEYD = S * HK, VALD = S * HV, CC = 2 * KEYD + VALD;
  float *beta = ab, *g = ab + HV;   /* g currently holds alpha; beta holds the raw beta */
  for (int h = 0; h < HV; h++) { beta[h] = sigmoidf(beta[h]); g[h] = softplusf(g[h] + y->ssm_dt_f[h]) * y->ssm_a_f[h]; }
  /* causal depthwise convolution (kernel CK, oldest tap first), then SiLU; the last CK - 1 inputs stay as the state */
  float *cs = f->conv_state + (size_t)y->rec_idx * CC * (CK - 1), *cvo = f->cvo;
  for (int c = 0; c < CC; c++) {
    float *st = cs + (size_t)c * (CK - 1);
    const float *w = y->conv_f + (size_t)c * CK;
    float acc = 0;
    for (int k = 0; k < CK - 1; k++) acc += st[k] * w[k];
    acc += qkv[c] * w[CK - 1];
    for (int k = 0; k + 1 < CK - 1; k++) st[k] = st[k + 1];
    st[CK - 2] = qkv[c];
    cvo[c] = siluf(acc);
  }
  float *q = cvo, *k = cvo + KEYD, *v = cvo + 2 * KEYD;
  for (int gi = 0; gi < HK; gi++) { l2norm(q + gi * S, S, f->rms_eps); l2norm(k + gi * S, S, f->rms_eps); }
  const float qscale = 1.0f / sqrtf((float)S);
  float *st = f->ssm_st + (size_t)y->rec_idx * HV * S * S;
  for (int h = 0; h < HV; h++) {
    const int kh = h % HK;   /* the key heads are repeated across the value heads in tiles (ggml_repeat), not in pairs */
    const float *kk = k + kh * S, *qq = q + kh * S, *vv = v + h * S;
    float *sm = st + (size_t)h * S * S;
    const float eg = expf(g[h]), bh = beta[h];
    for (int j = 0; j < S; j++) {
      float *sj = sm + (size_t)j * S;   /* sj[i]: key dimension i of value dimension j */
      float sk = 0;
      for (int i = 0; i < S; i++) { sj[i] *= eg; sk += sj[i] * kk[i]; }
      const float d = (vv[j] - sk) * bh;
      float o = 0;
      for (int i = 0; i < S; i++) { sj[i] += kk[i] * d; o += sj[i] * qq[i]; }
      out[h * S + j] = o * qscale;
    }
  }
  for (int h = 0; h < HV; h++) {   /* gated RMS-norm: norm(out) * silu(z), per value head */
    float *o = out + h * S;
    rmsnorm(o, o, y->ssm_norm_f, S, f->rms_eps);
    const float *zz = z + h * S;
    for (int i = 0; i < S; i++) o[i] *= siluf(zz[i]);
  }
}

/* The sequential part of a gated full-attention layer for ONE token at position pos: per-head QK-norm and rotary embeddings (qg holds query and gate per head, k / v the
 * projections), the key/value cache, causal attention, and the sigmoid output gate. */
static void attn_token(cl_fwd *f, const cl_fwd_layer *y, int pos, float *qg, float *k, const float *v, float *att) {
  const int H = f->n_head, KVH = f->n_head_kv, hd = f->head_dim, KW = f->n_kvw, GROUP = H / KVH;
  for (int h = 0; h < H; h++) { float *qh = qg + (size_t)h * 2 * hd; rmsnorm(qh, qh, y->qn_f, hd, f->rms_eps); rope_neox(qh, f->rope_dim, pos, f->rope_base, NULL); }
  for (int h = 0; h < KVH; h++) { float *kh = k + (size_t)h * hd; rmsnorm(kh, kh, y->kn_f, hd, f->rms_eps); rope_neox(kh, f->rope_dim, pos, f->rope_base, NULL); }
  float *kc = f->kc + ((size_t)y->kv_idx * f->n_ctx_max + pos) * KW, *vc = f->vc + ((size_t)y->kv_idx * f->n_ctx_max + pos) * KW;
  memcpy(kc, k, (size_t)KW * 4); memcpy(vc, v, (size_t)KW * 4);
  for (int h = 0; h < H; h++) {
    const float *qh = qg + (size_t)h * 2 * hd, *gate = qh + hd;
    const int kvo = (h / GROUP) * hd;
    float mx = -INFINITY;
    for (int t = 0; t <= pos; t++) {
      const float *kt = f->kc + ((size_t)y->kv_idx * f->n_ctx_max + t) * KW + kvo;
      float sc = 0;
      for (int i = 0; i < hd; i++) sc += qh[i] * kt[i];
      f->scores[t] = sc * f->attn_scale;
      if (f->scores[t] > mx) mx = f->scores[t];
    }
    float sum = 0;
    for (int t = 0; t <= pos; t++) { f->scores[t] = expf(f->scores[t] - mx); sum += f->scores[t]; }
    float *oh = att + h * hd;
    memset(oh, 0, (size_t)hd * 4);
    for (int t = 0; t <= pos; t++) {
      const float *vt = f->vc + ((size_t)y->kv_idx * f->n_ctx_max + t) * KW + kvo;
      const float p = f->scores[t] / sum;
      for (int i = 0; i < hd; i++) oh[i] += p * vt[i];
    }
    for (int i = 0; i < hd; i++) oh[i] *= sigmoidf(gate[i]);
  }
}

/* nb tokens (1 <= nb <= CL_PF_B) through a qwen35 model, layer by layer: every weight matrix is multiplied with all nb tokens in one pass over its rows, and only the
 * parts that depend on the previous token (the convolution and delta-rule state, the attention over earlier positions) run one token at a time, in order. With nb == 1
 * this is the ordinary one-token step. `logits` may be NULL when only the state is wanted. */
static int run_q35(cl_fwd *f, const int *tok, int nb, float *logits) {
  const int E = f->n_embd, KW = f->n_kvw, FF = f->n_ff_exp, pos0 = f->pos;
  const int S = f->ssm_state, HK = f->ssm_groups, HV = f->ssm_heads, VALD = S * HV, CC = 2 * S * HK + S * HV, NQ = f->n_q;
  const size_t AS = (size_t)(f->ssm_inner > NQ ? f->ssm_inner : NQ);
  float *X = f->pf_x, *XN = f->pf_xn, *HD = f->pf_hd, *HG = f->pf_hg, *HU = f->pf_hu, *ATT = f->pf_att;
  for (int b = 0; b < nb; b++) cl_dequant_row(f->tok_embd.type, f->tok_embd.p + (size_t)tok[b] * cl_row_bytes(f->tok_embd.type, (uint64_t)E), X + (size_t)b * E, E);
  for (int l = 0; l < f->n_layer; l++) {
    cl_fwd_layer *y = &f->L[l];
    for (int b = 0; b < nb; b++) { rmsnorm(XN + (size_t)b * E, X + (size_t)b * E, y->attn_norm_f, E, f->rms_eps); cl_xq_quantize(&f->pf_xqe[b], XN + (size_t)b * E); }
    if (y->recurrent) {
      { cl_mv j[4] = {mvb(&y->attn_qkv, XN, E, f->pf_xqe, f->pf_qkv, nb), mvb(&y->attn_gate, XN, E, f->pf_xqe, f->pf_z, nb), mvb(&y->ssm_beta, XN, E, f->pf_xqe, f->pf_ab, nb), mvb(&y->ssm_alpha, XN, E, f->pf_xqe, f->pf_ab + HV, nb)};
        j[2].out_stride = j[3].out_stride = (size_t)2 * HV;   /* beta and alpha of a token sit side by side */
        cl_matvec_batch(j, 4); }
      for (int b = 0; b < nb; b++) deltanet_token(f, y, f->pf_qkv + (size_t)b * CC, f->pf_z + (size_t)b * VALD, f->pf_ab + (size_t)b * 2 * HV, ATT + (size_t)b * AS);
      for (int b = 0; b < nb; b++) cl_xq_quantize(&f->pf_xqss[b], ATT + (size_t)b * AS);
      { cl_mv j = mvb(&y->ssm_out, ATT, AS, f->pf_xqss, HD, nb); cl_matvec_batch(&j, 1); }
    } else {
      { cl_mv j[3] = {mvb(&y->wq, XN, E, f->pf_xqe, f->pf_qg, nb), mvb(&y->wk, XN, E, f->pf_xqe, f->pf_k, nb), mvb(&y->wv, XN, E, f->pf_xqe, f->pf_v, nb)}; cl_matvec_batch(j, 3); }
      for (int b = 0; b < nb; b++) attn_token(f, y, pos0 + b, f->pf_qg + (size_t)b * 2 * NQ, f->pf_k + (size_t)b * KW, f->pf_v + (size_t)b * KW, ATT + (size_t)b * AS);
      for (int b = 0; b < nb; b++) cl_xq_quantize(&f->pf_xqq[b], ATT + (size_t)b * AS);
      { cl_mv j = mvb(&y->wo, ATT, AS, f->pf_xqq, HD, nb); cl_matvec_batch(&j, 1); }
    }
    for (size_t i = 0; i < (size_t)nb * E; i++) X[i] += HD[i];
    for (int b = 0; b < nb; b++) { rmsnorm(XN + (size_t)b * E, X + (size_t)b * E, y->ffn_norm_f, E, f->rms_eps); cl_xq_quantize(&f->pf_xqe[b], XN + (size_t)b * E); }   /* post_attention_norm */
    { cl_mv j[2] = {mvb(&y->gate, XN, E, f->pf_xqe, HG, nb), mvb(&y->up, XN, E, f->pf_xqe, HU, nb)}; cl_matvec_batch(j, 2); }
    for (int b = 0; b < nb; b++) { float *hg = HG + (size_t)b * FF, *hu = HU + (size_t)b * FF; for (int i = 0; i < FF; i++) hg[i] = siluf(hg[i]) * hu[i]; cl_xq_quantize(&f->pf_xqff[b], hg); }
    { cl_mv j = mvb(&y->down, HG, FF, f->pf_xqff, HD, nb); cl_matvec_batch(&j, 1); }
    for (size_t i = 0; i < (size_t)nb * E; i++) X[i] += HD[i];
  }
  f->pos += nb;
  if (logits) {
    rmsnorm(f->xn, X + (size_t)(nb - 1) * E, f->out_norm_f, E, f->rms_eps);
    cl_xq_quantize(&f->xq_e, f->xn);
    cl_mv j = mv(&f->output, f->xn, &f->xq_e, logits); cl_matvec_batch(&j, 1);
  }
  return 0;
}

int cl_fwd_step(cl_fwd *f, int token, float *logits, cl_fwd_route *route, char *err, size_t errlen) {
  if (token < 0 || token >= f->n_vocab) { snprintf(err, errlen, "token %d is outside the vocabulary", token); return 1; }
  if (f->pos >= f->n_ctx_max) { snprintf(err, errlen, "the context is full (%d tokens)", f->n_ctx_max); return 1; }
  const int E = f->n_embd, H = f->n_head, KVH = f->n_head_kv, hd = f->head_dim, pos = f->pos, KW = f->n_kvw;
  cl_dequant_row(f->tok_embd.type, f->tok_embd.p + (size_t)token * cl_row_bytes(f->tok_embd.type, (uint64_t)E), f->x, E);
  if (f->hybrid) { run_q35(f, &token, 1, logits); return 0; }
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
    if (f->qkv_bias) {   /* qwen2: the projections carry biases */
      for (int i = 0; i < f->n_q; i++) f->q[i] += y->bq_f[i];
      for (int i = 0; i < f->n_kvw; i++) { f->k[i] += y->bk_f[i]; f->v[i] += y->bv_f[i]; }
    }
    if (f->qk_mode == 0) { /* granitemoe: no QK-norm */ }
    else if (f->qk_per_head) {   /* qwen3moe: each head's vector is normalised on its own, with one weight shared by all heads */
      for (int h = 0; h < H; h++) rmsnorm(f->q + h * hd, f->q + h * hd, y->qn_f, hd, f->rms_eps);
      for (int h = 0; h < KVH; h++) rmsnorm(f->k + h * hd, f->k + h * hd, y->kn_f, hd, f->rms_eps);
    } else {   /* olmoe: over the whole projected vector, before it is split into heads */
      rmsnorm(f->q, f->q, y->qn_f, f->n_q, f->rms_eps);
      rmsnorm(f->k, f->k, y->kn_f, f->n_kvw, f->rms_eps);
    }
    for (int h = 0; h < H; h++) (f->rope_pairs ? rope_pairs : rope_neox)(f->q + h * hd, hd, pos, f->rope_base, f->rope_factors);
    for (int h = 0; h < KVH; h++) (f->rope_pairs ? rope_pairs : rope_neox)(f->k + h * hd, hd, pos, f->rope_base, f->rope_factors);
    float *kc = f->kc + ((size_t)l * f->n_ctx_max + pos) * KW, *vc = f->vc + ((size_t)l * f->n_ctx_max + pos) * KW;
    memcpy(kc, f->k, (size_t)KW * 4); memcpy(vc, f->v, (size_t)KW * 4);
    attend(f, l, f->q, f->att, pos, 1);
    cl_xq_quantize(&f->xq_q, f->att);
    { cl_mv j = mv(&y->wo, f->att, &f->xq_q, f->hd); cl_matvec_batch(&j, 1); }
    for (int i = 0; i < E; i++) f->x[i] += f->resid_scale * f->hd[i];

    if (f->dense) {   /* one plain SwiGLU feed-forward instead of a router and experts */
      const int FF = f->n_ff_exp;
      rmsnorm(f->xn, f->x, y->ffn_norm_f, E, f->rms_eps);
      cl_xq_quantize(&f->xq_e, f->xn);
      { cl_mv j[2] = {mv(&y->gate, f->xn, &f->xq_e, f->hg), mv(&y->up, f->xn, &f->xq_e, f->hu)}; cl_matvec_batch(j, 2); }
      for (int i = 0; i < FF; i++) f->hg[i] = f->hg[i] / (1.0f + expf(-f->hg[i])) * f->hu[i];
      cl_xq_quantize(&f->xq_ff, f->hg);
      { cl_mv j = mv(&y->down, f->hg, &f->xq_ff, f->hd); cl_matvec_batch(&j, 1); }
      for (int i = 0; i < E; i++) f->x[i] += f->resid_scale * f->hd[i];
      continue;
    }

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
      j[2 * k] = (cl_mv){y->gate.type, g, f->n_ff_exp, E, f->xn, &f->xq_e, f->hg + k * FF, 1, 0, 0};
      j[2 * k + 1] = (cl_mv){y->up.type, u, f->n_ff_exp, E, f->xn, &f->xq_e, f->hu + k * FF, 1, 0, 0};
    }
    cl_matvec_batch(j, 2 * f->n_used);
    for (int k = 0; k < f->n_used; k++) {
      float *hg = f->hg + k * FF, *hu = f->hu + k * FF;
      for (size_t i = 0; i < FF; i++) hg[i] = hg[i] / (1.0f + expf(-hg[i])) * hu[i];   /* silu(gate) * up */
      cl_xq_quantize(&f->xq_ffk[k], hg);
      const uint8_t *d = f->xs ? P[k] + y->gate_slab + y->up_slab : y->down.p + (size_t)sel[k] * y->down_slab;
      j[k] = (cl_mv){y->down.type, d, E, f->n_ff_exp, hg, &f->xq_ffk[k], f->hd + (size_t)k * E, 1, 0, 0};
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

/* ---- Prompt processing for the plain transformer architectures (llama, qwen2, olmoe, qwen3moe, granitemoe) ---------------------------------------------------------- */

/* CLUTOR_PROFILE=1: where the time of the batched prompt path goes, printed to stderr after each prompt (for finding the next thing to speed up). Costs two clock reads per section. */
static int look_default(void) { const char *e = getenv("CLUTOR_LOOKAHEAD"); return e && *e ? atoi(e) : 1; }
static double prof_t[10];
static int prof_on = -1;
static const char *const prof_name[10] = {"norm+quantize", "q/k/v products", "bias, rope, store K/V", "attention", "output product", "routing", "waiting for experts", "expert products", "expert other", "final norm + logits"};
static double prof_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }
#define P0() double _pt = prof_on > 0 ? prof_now() : 0
#define PA(i) do { if (prof_on > 0) { double _n = prof_now(); prof_t[i] += _n - _pt; _pt = _n; } } while (0)
_Static_assert(CL_PS_B <= CL_MV_MAX, "a prompt batch must fit in one batched product");
static int ps_alloc(cl_fwd *f) {
  if (f->ps_ready) return 0;
  const size_t B = f->xs ? CL_PS_B : 64, E = (size_t)f->n_embd, NQ = (size_t)f->n_q, KW = (size_t)f->n_kvw, FF = (size_t)f->n_ff_exp;
  f->ps_x = calloc(B * E, 4); f->ps_xn = calloc(B * E, 4); f->ps_xg = calloc(B * E, 4); f->ps_hd = calloc(B * E, 4); f->ps_moe = calloc(B * E, 4); f->ps_hde = calloc(B * E, 4);
  f->ps_q = calloc(B * NQ, 4); f->ps_att = calloc(B * NQ, 4); f->ps_k = calloc(B * KW, 4); f->ps_v = calloc(B * KW, 4);
  f->ps_hg = calloc(B * FF, 4); f->ps_hu = calloc(B * FF, 4); f->ps_rg = calloc(B * (size_t)f->n_expert, 4);
  f->ps_sel = calloc(B * (size_t)f->n_used, sizeof(int)); f->ps_wt = calloc(B * (size_t)f->n_used, 4);
  f->ps_xqe = calloc(B, sizeof(cl_xq)); f->ps_xqq = calloc(B, sizeof(cl_xq)); f->ps_xqff = calloc(B, sizeof(cl_xq));
  int ok = f->ps_x && f->ps_xn && f->ps_xg && f->ps_hd && f->ps_moe && f->ps_hde && f->ps_q && f->ps_att && f->ps_k && f->ps_v && f->ps_hg && f->ps_hu && f->ps_rg && f->ps_sel && f->ps_wt && f->ps_xqe && f->ps_xqq && f->ps_xqff;
  for (size_t i = 0; ok && i < B; i++) ok = cl_xq_init(&f->ps_xqe[i], f->n_embd) && cl_xq_init(&f->ps_xqq[i], f->n_q) && cl_xq_init(&f->ps_xqff[i], f->n_ff_exp);
  if (!ok) return 1;
  f->ps_cap = (int)B;
  f->ps_ready = 1;
  return 0;
}

/* The router's dot products for every token of a batch, one task per token: each is 128 or more rows of the layer's hidden width, which in a serial loop was a fifth of the time. */
typedef struct { cl_fwd *f; const cl_fwd_layer *y; const float *X; float *XN; } route_job;
static void route_task(void *vc, int b) {
  route_job *c = vc; cl_fwd *f = c->f; const int E = f->n_embd;
  float *xn = c->XN + (size_t)b * E;
  rmsnorm(xn, c->X + (size_t)b * E, c->y->ffn_norm_f, E, f->rms_eps);
  cl_xq_quantize(&f->ps_xqe[b], xn);
  float *rg = f->ps_rg + (size_t)b * f->n_expert;
  for (int e = 0; e < f->n_expert; e++) { const float *r = c->y->router_f + (size_t)e * E; float acc = 0; for (int i = 0; i < E; i++) acc += r[i] * xn[i]; rg[e] = acc; }
}

/* nb tokens (1 <= nb <= CL_PS_B) through the model together, layer by layer. Every dense weight matrix is multiplied with all nb tokens in one pass over its rows; in a
 * mixture-of-experts layer each token picks its experts as it would alone, then each chosen expert is read once and run over all the tokens that chose it. The attention
 * for each token still looks only at earlier positions, in order. Gives the same results as nb calls to cl_fwd_step (the integer dot products are exact; only the order in
 * which a token's expert outputs are added differs from a step's, which can change the last bits). `logits` may be NULL when only the state is wanted. */
static int run_std(cl_fwd *f, const int *tok, int nb, float *logits, char *err, size_t errlen) {
  const int E = f->n_embd, H = f->n_head, KVH = f->n_head_kv, hd = f->head_dim, KW = f->n_kvw, NQ = f->n_q, FF = f->n_ff_exp, pos0 = f->pos;
  if (prof_on < 0) { const char *e = getenv("CLUTOR_PROFILE"); prof_on = e && *e && *e != '0'; }
  P0();
  float *X = f->ps_x, *XN = f->ps_xn, *Q = f->ps_q, *K = f->ps_k, *V = f->ps_v, *ATT = f->ps_att, *HD = f->ps_hd;
  for (int b = 0; b < nb; b++) {
    float *x = X + (size_t)b * E;
    cl_dequant_row(f->tok_embd.type, f->tok_embd.p + (size_t)tok[b] * cl_row_bytes(f->tok_embd.type, (uint64_t)E), x, E);
    if (f->embed_scale != 1.0f) for (int i = 0; i < E; i++) x[i] *= f->embed_scale;
  }
  for (int l = 0; l < f->n_layer; l++) {
    cl_fwd_layer *y = &f->L[l];
    for (int b = 0; b < nb; b++) { rmsnorm(XN + (size_t)b * E, X + (size_t)b * E, y->attn_norm_f, E, f->rms_eps); cl_xq_quantize(&f->ps_xqe[b], XN + (size_t)b * E); }
    PA(0);
    { cl_mv j[3] = {mvb(&y->wq, XN, E, f->ps_xqe, Q, nb), mvb(&y->wk, XN, E, f->ps_xqe, K, nb), mvb(&y->wv, XN, E, f->ps_xqe, V, nb)}; cl_matvec_batch(j, 3); }
    PA(1);
    for (int b = 0; b < nb; b++) {
      float *q = Q + (size_t)b * NQ, *k = K + (size_t)b * KW, *v = V + (size_t)b * KW;
      const int pos = pos0 + b;
      if (f->qkv_bias) { for (int i = 0; i < NQ; i++) q[i] += y->bq_f[i]; for (int i = 0; i < KW; i++) { k[i] += y->bk_f[i]; v[i] += y->bv_f[i]; } }
      if (f->qk_mode == 0) { /* no QK-norm */ }
      else if (f->qk_per_head) { for (int h = 0; h < H; h++) rmsnorm(q + h * hd, q + h * hd, y->qn_f, hd, f->rms_eps); for (int h = 0; h < KVH; h++) rmsnorm(k + h * hd, k + h * hd, y->kn_f, hd, f->rms_eps); }
      else { rmsnorm(q, q, y->qn_f, NQ, f->rms_eps); rmsnorm(k, k, y->kn_f, KW, f->rms_eps); }
      for (int h = 0; h < H; h++) (f->rope_pairs ? rope_pairs : rope_neox)(q + h * hd, hd, pos, f->rope_base, f->rope_factors);
      for (int h = 0; h < KVH; h++) (f->rope_pairs ? rope_pairs : rope_neox)(k + h * hd, hd, pos, f->rope_base, f->rope_factors);
      float *kc = f->kc + ((size_t)l * f->n_ctx_max + pos) * KW, *vc = f->vc + ((size_t)l * f->n_ctx_max + pos) * KW;
      memcpy(kc, k, (size_t)KW * 4); memcpy(vc, v, (size_t)KW * 4);
    }
    PA(2);
    attend(f, l, Q, ATT, pos0, nb);
    PA(3);   /* every token's key and value are stored now; each token attends only to positions up to its own */
    for (int b = 0; b < nb; b++) cl_xq_quantize(&f->ps_xqq[b], ATT + (size_t)b * NQ);
    { cl_mv j = mvb(&y->wo, ATT, NQ, f->ps_xqq, HD, nb); cl_matvec_batch(&j, 1); }
    for (size_t i = 0; i < (size_t)nb * E; i++) X[i] += f->resid_scale * HD[i];
    PA(4);

    if (f->dense) {   /* one plain SwiGLU feed-forward */
      float *HG = f->ps_hg, *HU = f->ps_hu;
      for (int b = 0; b < nb; b++) { rmsnorm(XN + (size_t)b * E, X + (size_t)b * E, y->ffn_norm_f, E, f->rms_eps); cl_xq_quantize(&f->ps_xqe[b], XN + (size_t)b * E); }
      { cl_mv j[2] = {mvb(&y->gate, XN, E, f->ps_xqe, HG, nb), mvb(&y->up, XN, E, f->ps_xqe, HU, nb)}; cl_matvec_batch(j, 2); }
      for (int b = 0; b < nb; b++) { float *hg = HG + (size_t)b * FF, *hu = HU + (size_t)b * FF; for (int i = 0; i < FF; i++) hg[i] = hg[i] / (1.0f + expf(-hg[i])) * hu[i]; cl_xq_quantize(&f->ps_xqff[b], hg); }
      { cl_mv j = mvb(&y->down, HG, FF, f->ps_xqff, HD, nb); cl_matvec_batch(&j, 1); }
      for (size_t i = 0; i < (size_t)nb * E; i++) X[i] += f->resid_scale * HD[i];
      continue;
    }

    /* mixture of experts: route every token */
    const int NU = f->n_used;
    { route_job rj = {f, y, X, XN}; cl_parallel_for(nb, route_task, &rj); }
    for (int b = 0; b < nb; b++) {
      memcpy(f->rl, f->ps_rg + (size_t)b * f->n_expert, (size_t)f->n_expert * sizeof(float));
      memcpy(f->rlogit, f->rl, (size_t)f->n_expert * sizeof(float));
      float mx = f->rl[0];
      for (int e = 1; e < f->n_expert; e++) if (f->rl[e] > mx) mx = f->rl[e];
      double sum = 0;
      for (int e = 0; e < f->n_expert; e++) { f->rl[e] = expf(f->rl[e] - mx); sum += f->rl[e]; }
      for (int e = 0; e < f->n_expert; e++) f->rl[e] = (float)(f->rl[e] / sum);
      int *sel = f->ps_sel + (size_t)b * NU; float *wt = f->ps_wt + (size_t)b * NU;
      if (f->smart.enabled) {
        struct { cl_xs *xs; int layer; } rc = {f->xs, l};
        cl_route_result rr;
        if (cl_route(f->rlogit, f->n_expert, NU, &f->smart, resident_cb, &rc, &rr)) { snprintf(err, errlen, "bad SMART-ROUTE settings"); return 1; }
        for (int k = 0; k < NU; k++) { sel[k] = rr.selected[k]; wt[k] = f->rl[sel[k]]; }
        f->subs += (uint64_t)rr.substituted; f->dropped_mass += rr.dropped_mass;
        if (rr.max_delta > f->max_delta) f->max_delta = rr.max_delta;
      } else {
        char used[4096] = {0};
        for (int k = 0; k < NU; k++) { int best = -1; for (int e = 0; e < f->n_expert; e++) if (!used[e] && (best < 0 || f->rl[e] > f->rl[best])) best = e; used[best] = 1; sel[k] = best; wt[k] = f->rl[best]; }
      }
      if (f->norm_topk) { float z = 0; for (int k = 0; k < NU; k++) z += wt[k]; if (z > 0) for (int k = 0; k < NU; k++) wt[k] /= z; }
      f->expert_slots += (uint64_t)NU;
    }
    if (f->prev_sel) memcpy(f->prev_sel + (size_t)l * NU, f->ps_sel + (size_t)(nb - 1) * NU, (size_t)NU * sizeof(int));
    memset(f->ps_moe, 0, (size_t)nb * E * 4);

    PA(5);
    /* then each chosen expert once, over the tokens that chose it, a few at a time so the store never holds more than a token's worth of requests */
    int todo[4096], n_todo = 0;
    { char seen[4096] = {0}; for (int b = 0; b < nb; b++) for (int k = 0; k < NU; k++) { int e = f->ps_sel[(size_t)b * NU + k]; if (!seen[e]) { seen[e] = 1; todo[n_todo++] = e; } } }
    /* Reads run one group ahead of the compute when the cache has room for two groups (and a few spare slots) at once, so the disk works while the cores multiply. */
    int hnd[4096] = {0}, issued = 0;
    int depth = 0;   /* groups requested ahead of the one being computed; the cache must hold them all pinned with room to spare. CLUTOR_LOOKAHEAD=k sets it (0 waits for each group before asking for the next) */
    if (f->xs) { depth = look_default(); const int room = (int)(cl_xs_cache_slots(f->xs) / (uint32_t)NU) - 2; if (depth > room) depth = room < 0 ? 0 : room; }
    for (int g0 = 0; g0 < n_todo; g0 += NU) {
      const int g1 = g0 + NU < n_todo ? g0 + NU : n_todo;
      if (f->xs) {
        const int want = g1 + depth * NU < n_todo ? g1 + depth * NU : n_todo;
        while (issued < want) {   /* resident experts first, so they are pinned before a miss can evict them */
          const int h1 = issued + NU < n_todo ? issued + NU : n_todo;
          for (int pass = 0; pass < 2; pass++) for (int i = issued; i < h1; i++) { int here = cl_xs_resident(f->xs, l, todo[i]); if ((pass == 0) == (here != 0)) hnd[i] = cl_xs_request(f->xs, l, todo[i]); }
          issued = h1;
        }
      }
      for (int i = g0; i < g1; i++) {
        PA(8);
        const int e = todo[i];
        const uint8_t *Pi = NULL;
        if (f->xs) {
          Pi = cl_xs_wait(f->xs, hnd[i]);
          if (!Pi) { snprintf(err, errlen, "reading expert %d of layer %d from disk failed", e, l); for (int j = i; j < issued; j++) cl_xs_release(f->xs, hnd[j]); return 1; }
        }
        PA(6);
        /* the tokens that chose this expert, with their weights */
        int who[CL_PS_B]; float wgt[CL_PS_B]; int ni = 0;
        for (int b = 0; b < nb; b++) for (int k = 0; k < NU; k++) if (f->ps_sel[(size_t)b * NU + k] == e) { who[ni] = b; wgt[ni] = f->ps_wt[(size_t)b * NU + k]; ni++; break; }
        const uint8_t *gp = f->xs ? Pi : y->gate.p + (size_t)e * y->gate_slab, *up = f->xs ? Pi + y->gate_slab : y->up.p + (size_t)e * y->up_slab;
        const uint8_t *dp = f->xs ? Pi + y->gate_slab + y->up_slab : y->down.p + (size_t)e * y->down_slab;
        cl_xq sub[CL_PS_B];
        for (int t = 0; t < ni; t++) { sub[t] = f->ps_xqe[who[t]]; memcpy(f->ps_xg + (size_t)t * E, XN + (size_t)who[t] * E, (size_t)E * 4); }
        float *HG = f->ps_hg, *HU = f->ps_hu;
        PA(8);
        { cl_mv j[2] = {{y->gate.type, gp, FF, E, f->ps_xg, sub, HG, ni, (size_t)E, (size_t)FF}, {y->up.type, up, FF, E, f->ps_xg, sub, HU, ni, (size_t)E, (size_t)FF}}; cl_matvec_batch(j, 2); }
        PA(7);
        for (int t = 0; t < ni; t++) { float *hg = HG + (size_t)t * FF, *hu = HU + (size_t)t * FF; for (int q = 0; q < FF; q++) hg[q] = hg[q] / (1.0f + expf(-hg[q])) * hu[q]; cl_xq_quantize(&f->ps_xqff[t], hg); }
        PA(8);
        { cl_mv j = {y->down.type, dp, E, FF, HG, f->ps_xqff, f->ps_hde, ni, (size_t)FF, (size_t)E}; cl_matvec_batch(&j, 1); }
        PA(7);
        for (int t = 0; t < ni; t++) { float *m = f->ps_moe + (size_t)who[t] * E; const float *d = f->ps_hde + (size_t)t * E; for (int q = 0; q < E; q++) m[q] += wgt[t] * d[q]; }
        if (f->xs) cl_xs_release(f->xs, hnd[i]);
      }
    }
    PA(8);
    for (size_t i = 0; i < (size_t)nb * E; i++) X[i] += f->resid_scale * f->ps_moe[i];
    PA(8);
  }
  f->pos += nb;
  if (logits) {
    rmsnorm(f->xn, X + (size_t)(nb - 1) * E, f->out_norm_f, E, f->rms_eps);
    cl_xq_quantize(&f->xq_e, f->xn);
    cl_mv j = mv(&f->output, f->xn, &f->xq_e, logits); cl_matvec_batch(&j, 1);
    if (f->logit_scale != 1.0f) for (int i = 0; i < f->n_vocab; i++) logits[i] /= f->logit_scale;
  }
  PA(9);
  return 0;
}

int cl_fwd_prefill(cl_fwd *f, const int *tokens, int n, float *logits, char *err, size_t errlen) {
  if (n < 1) { snprintf(err, errlen, "nothing to feed"); return 1; }
  if (f->pos + n > f->n_ctx_max) { snprintf(err, errlen, "the context is full (%d tokens)", f->n_ctx_max); return 1; }
  for (int i = 0; i < n; i++) if (tokens[i] < 0 || tokens[i] >= f->n_vocab) { snprintf(err, errlen, "token %d is outside the vocabulary", tokens[i]); return 1; }
  if (!f->hybrid) {
    if (n == 1 || getenv("CLUTOR_STEP_PREFILL") || ps_alloc(f)) {   /* one token is just a step; CLUTOR_STEP_PREFILL=1 forces the old token-by-token path, the reference the batched one is tested against */
      if (!f->logits_buf) f->logits_buf = malloc((size_t)f->n_vocab * 4);
      if (!f->logits_buf) { snprintf(err, errlen, "out of memory"); return 1; }
      for (int i = 0; i < n; i++) if (cl_fwd_step(f, tokens[i], i == n - 1 && logits ? logits : f->logits_buf, NULL, err, errlen)) return 1;
      return 0;
    }
    /* CLUTOR_PS_B=k (1..CL_PS_B) shrinks the batch, so the tests can cross many batch boundaries with short prompts */
    int cap = f->ps_cap;
    { const char *e = getenv("CLUTOR_PS_B"); if (e && atoi(e) >= 1 && atoi(e) < f->ps_cap) cap = atoi(e); }
    for (int i = 0; i < n; i += cap) { int nb = n - i < cap ? n - i : cap; if (run_std(f, tokens + i, nb, i + nb == n ? logits : NULL, err, errlen)) return 1; }
    if (prof_on > 0) {
      double tot = 0; for (int k = 0; k < 10; k++) tot += prof_t[k];
      fprintf(stderr, "[profile: %d tokens, %.2f s in the batched path]", n, tot);
      for (int k = 0; k < 10; k++) { fprintf(stderr, " %s %.2fs (%.0f%%);", prof_name[k], prof_t[k], tot > 0 ? 100 * prof_t[k] / tot : 0); prof_t[k] = 0; }
      fprintf(stderr, "\n");
    }
    return 0;
  }
  for (int i = 0; i < n; i += CL_PF_B) { int nb = n - i < CL_PF_B ? n - i : CL_PF_B; run_q35(f, tokens + i, nb, i + nb == n ? logits : NULL); }
  return 0;
}
