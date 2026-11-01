/*
 * A real forward pass for two mixture-of-experts architectures (llama.cpp's "olmoe", "qwen3moe" and "granitemoe"): token embedding, RMSNorm, attention with QK-norm and
 * NeoX rotary embeddings (grouped-query for qwen3moe), a softmax router over every expert, the top-k experts' SwiGLU feed-forwards, and the output projection.
 * Weights are read straight from the memory-mapped GGUF and dequantized block by block; the arithmetic is float32.
 *
 * Every step is checked against llama.cpp (see `make verify-llama`). Routing here is the model's own: the gate sees the real hidden state.
 */
#ifndef CL_FORWARD_H
#define CL_FORWARD_H
#include "gguf.h"
#include "meta.h"
#include "quant.h"
#include "route.h"
#include "xs.h"
#include <stddef.h>
#include <stdint.h>

typedef struct { const uint8_t *p; uint32_t type; int rows, cols; } cl_w;   /* rows x cols weights of `type`, row-major (cols is the input width) */

typedef struct {
  cl_w attn_norm, wq, wk, wv, wo, qn, kn, ffn_norm, router, gate, up, down; /* gate/up/down are the fused expert tensors */
  float *attn_norm_f, *qn_f, *kn_f, *ffn_norm_f, *router_f;
  uint64_t gate_slab, up_slab, down_slab;   /* bytes of one expert in each fused tensor */
} cl_fwd_layer;

typedef struct {
  cl_model model;
  cl_meta meta;
  uint8_t *map; size_t map_size;
  char arch[16];           /* "olmoe", "qwen3moe" or "granitemoe" */
  int n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff_exp, n_expert, n_used, n_vocab, n_ctx_max;
  int n_q, n_kvw;          /* width of the query projection (n_head * head_dim) and of the key / value projections (n_head_kv * head_dim) */
  int qk_mode;             /* 0: no QK-norm (granitemoe); 1: RMSNorm over the whole projected vector (olmoe); 2: over each head_dim vector with one shared weight (qwen3moe) */
  int qk_per_head;         /* qk_mode == 2 */
  int rope_pairs;          /* 1: rotate adjacent pairs (2i, 2i+1) (llama.cpp's NORM rope, granitemoe); 0: rotate the two halves (NEOX: olmoe, qwen3moe) */
  float embed_scale, attn_scale, resid_scale, logit_scale; /* granite's multipliers; 1, 1/sqrt(head_dim), 1, 1 for the others */
  int norm_topk;           /* qwen3moe and granitemoe renormalise the chosen experts' weights to sum to 1; olmoe uses the raw softmax probabilities */
  float rms_eps, rope_base;
  cl_w tok_embd, output;
  float *out_norm_f;
  cl_fwd_layer *L;
  float *kc, *vc;          /* [layer][position][n_head_kv * head_dim] */
  float *x, *xn, *q, *k, *v, *att, *scores, *logits_buf, *moe, *hg, *hu, *hd, *rl;
  cl_xq xq_e, xq_q, xq_ff, *xq_ffk;       /* the current activation vector, quantized for the integer dot products */
  cl_xs *xs;               /* when set, experts come from disk through the cache instead of from the mapped file */
  cl_smart smart;
  uint64_t subs, expert_slots; /* experts used in place of the router's own choice, out of all expert slots filled */
  double dropped_mass, max_delta;
  float *rlogit;           /* the router's raw scores for the layer being processed */
  int prefetch_n;          /* experts per layer to prefetch */
  int prefetch;            /* 0 off; 1 "prev": guess the next layer's experts from the previous token's choices; 2 "ahead": apply the next layer's router to the current hidden state; either way read them early */
  int *pred_sel;           /* [layer][prefetch_n]: what was predicted for this token's layer, -1 for none */
  uint64_t pred_total, pred_hit; /* predicted experts, and how many of them the router then chose */
  int *prev_sel;           /* [layer][n_used]: the experts each layer chose for the previous token, -1 before the first */
  int pos;                 /* tokens already in the KV cache */
} cl_fwd;

/* What the router chose for the token just processed: n_layer rows of n_used experts, best first, with the weight each was given. `logits` (if not NULL)
 * receives the router's raw scores for all experts, which is what the streaming engine replays. */
typedef struct { int *experts; float *weights; float *logits; /* optional: n_layer * n_expert raw router logits */ int n_layer, n_used; } cl_fwd_route;

/* Streaming options. cache_bytes == 0 keeps the whole file mapped (fast, but resident memory grows with the model); otherwise experts are read
 * on demand into a cache of that many bytes and the rest of the model stays on disk. */
typedef struct { uint64_t cache_bytes; int stage_slots, io_threads, os_bypass, prefetch;
  cl_smart smart;   /* SMART-ROUTE: needs streaming (cache_bytes > 0), since it asks the cache what is in memory */
  int prefetch_n;   /* how many of the previous token's experts per layer to read early (best-ranked first); 0 = all that were used */ cl_policy policy; } cl_fwd_opts;
int cl_fwd_open(cl_fwd *f, const char *path, int n_ctx, const cl_fwd_opts *opts, char *err, size_t errlen);
void cl_fwd_close(cl_fwd *f);
void cl_fwd_reset(cl_fwd *f);
/* Runs one token at position f->pos (then advances it). Writes n_vocab logits. `route` may be NULL, else it must hold n_layer*n_used entries. */
int cl_fwd_step(cl_fwd *f, int token, float *logits, cl_fwd_route *route, char *err, size_t errlen);
#endif
