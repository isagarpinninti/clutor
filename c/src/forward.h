/*
 * A real forward pass for OLMoE-style models (llama.cpp's "olmoe" architecture): token embedding, RMSNorm, attention with QK-norm and NeoX
 * rotary embeddings, a softmax router over every expert, the top-k experts' SwiGLU feed-forwards, and the output projection.
 * Weights are read straight from the memory-mapped GGUF and dequantized block by block; the arithmetic is float32.
 *
 * Every step is checked against llama.cpp (see `make verify-llama`). Routing here is the model's own: the gate sees the real hidden state.
 */
#ifndef CL_FORWARD_H
#define CL_FORWARD_H
#include "gguf.h"
#include "meta.h"
#include "quant.h"
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
  int n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff_exp, n_expert, n_used, n_vocab, n_ctx_max;
  float rms_eps, rope_base;
  cl_w tok_embd, output;
  float *out_norm_f;
  cl_fwd_layer *L;
  float *kc, *vc;          /* [layer][position][n_head_kv * head_dim] */
  float *x, *xn, *q, *k, *v, *att, *scores, *logits_buf, *moe, *hg, *hu, *hd, *rl;
  cl_xq xq_e, xq_ff, *xq_ffk;       /* the current activation vector, quantized for the integer dot products */
  cl_xs *xs;               /* when set, experts come from disk through the cache instead of from the mapped file */
  int prefetch_n;          /* experts per layer to prefetch */
  int prefetch;            /* guess the next layer's experts from the previous token's choices and read them early */
  int *prev_sel;           /* [layer][n_used]: the experts each layer chose for the previous token, -1 before the first */
  int pos;                 /* tokens already in the KV cache */
} cl_fwd;

/* What the router chose for the token just processed: n_layer rows of n_used experts, best first, with the weight each was given. `logits` (if not NULL)
 * receives the router's raw scores for all experts, which is what the streaming engine replays. */
typedef struct { int *experts; float *weights; float *logits; /* optional: n_layer * n_expert raw router logits */ int n_layer, n_used; } cl_fwd_route;

/* Streaming options. cache_bytes == 0 keeps the whole file mapped (fast, but resident memory grows with the model); otherwise experts are read
 * on demand into a cache of that many bytes and the rest of the model stays on disk. */
typedef struct { uint64_t cache_bytes; int stage_slots, io_threads, os_bypass, prefetch;
  int prefetch_n;   /* how many of the previous token's experts per layer to read early (best-ranked first); 0 = all that were used */ cl_policy policy; } cl_fwd_opts;
int cl_fwd_open(cl_fwd *f, const char *path, int n_ctx, const cl_fwd_opts *opts, char *err, size_t errlen);
void cl_fwd_close(cl_fwd *f);
void cl_fwd_reset(cl_fwd *f);
/* Runs one token at position f->pos (then advances it). Writes n_vocab logits. `route` may be NULL, else it must hold n_layer*n_used entries. */
int cl_fwd_step(cl_fwd *f, int token, float *logits, cl_fwd_route *route, char *err, size_t errlen);
#endif
