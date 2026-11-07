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
  cl_w attn_norm, wq, wk, wv, wo, qn, kn, ffn_norm, router, gate, up, down; /* gate/up/down are the fused expert tensors, or (dense models) the plain feed-forward matrices */
  cl_w bq, bk, bv;                          /* qwen2: biases of the query, key and value projections */
  cl_w attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out;   /* qwen35: the Gated DeltaNet layers (recurrent); the other layers use wq/wk/wv/wo with a gated query */
  float *ssm_a_f, *ssm_dt_f, *ssm_norm_f, *conv_f;
  int recurrent, rec_idx, kv_idx;           /* qwen35: Gated DeltaNet layer or gated full attention; index among the recurrent layers / among the attention layers */
  float *bq_f, *bk_f, *bv_f;
  float *attn_norm_f, *qn_f, *kn_f, *ffn_norm_f, *router_f;
  uint64_t gate_slab, up_slab, down_slab;   /* bytes of one expert in each fused tensor */
} cl_fwd_layer;

typedef struct {
  cl_model model;
  cl_meta meta;
  uint8_t *map; size_t map_size;
  char arch[16];           /* "olmoe", "qwen3moe" or "granitemoe" */
  int n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff_exp, n_expert, n_used, n_vocab, n_ctx_max, n_ctx_train;
  int n_q, n_kvw;          /* width of the query projection (n_head * head_dim) and of the key / value projections (n_head_kv * head_dim) */
  int qk_mode;             /* 0: no QK-norm (granitemoe); 1: RMSNorm over the whole projected vector (olmoe); 2: over each head_dim vector with one shared weight (qwen3moe) */
  int qk_per_head;         /* qk_mode == 2 */
  int dense;               /* qwen2: no experts, every layer has one plain SwiGLU feed-forward of width n_ff_exp */
  int qkv_bias;            /* the query, key and value projections have bias vectors */
  int rope_pairs;          /* 1: rotate adjacent pairs (2i, 2i+1) (llama.cpp's NORM rope, granitemoe); 0: rotate the two halves (NEOX: olmoe, qwen3moe) */
  float *rope_factors;     /* llama 3: rope_freqs.weight, one divisor per rotated pair (head_dim / 2 values), or NULL */
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
  int hybrid;              /* qwen35: three Gated DeltaNet (linear attention) layers, then one gated full-attention layer, repeating; a plain SwiGLU feed-forward in every layer */
  int rope_dim;            /* dimensions of each head that are rotated (the first rope_dim); head_dim for every architecture but qwen35 */
  int ssm_conv, ssm_state, ssm_groups, ssm_heads, ssm_inner, attn_interval, n_kv_layers, n_rec;
  float *conv_state, *ssm_st;  /* per recurrent layer: the last (conv - 1) inputs of the causal convolution, and one state matrix per head */
  float *qg, *qkv, *zbuf, *ab, *cvo;   /* qwen35 scratch */
  cl_xq xq_ss;
  /* qwen35 prompt processing: scratch for up to CL_PF_B tokens at once (the weights are streamed once per batch, not once per token) */
  float *pf_x, *pf_xn, *pf_qkv, *pf_z, *pf_ab, *pf_qg, *pf_k, *pf_v, *pf_att, *pf_hd, *pf_hg, *pf_hu;
  cl_xq *pf_xqe, *pf_xqq, *pf_xqss, *pf_xqff;
  /* prompt processing for every architecture but qwen35: scratch for up to CL_PS_B tokens, allocated on first use */
  int ps_ready;
  float *ps_x, *ps_xn, *ps_xg, *ps_q, *ps_k, *ps_v, *ps_att, *ps_hd, *ps_moe, *ps_hg, *ps_hu, *ps_wt, *ps_hde;
  int *ps_sel;
  cl_xq *ps_xqe, *ps_xqq, *ps_xqff;
  int locked;              /* the mapped weights are pinned in memory */
  int pos;                 /* tokens already in the KV cache */
} cl_fwd;

/* What the router chose for the token just processed: n_layer rows of n_used experts, best first, with the weight each was given. `logits` (if not NULL)
 * receives the router's raw scores for all experts, which is what the streaming engine replays. */
typedef struct { int *experts; float *weights; float *logits; /* optional: n_layer * n_expert raw router logits */ int n_layer, n_used; } cl_fwd_route;

/* Streaming options. cache_bytes == 0 keeps the whole file mapped (fast, but resident memory grows with the model); otherwise experts are read
 * on demand into a cache of that many bytes and the rest of the model stays on disk. */
typedef struct { uint64_t cache_bytes; int stage_slots, io_threads, os_bypass, prefetch;
  int lock;         /* pin the mapped weights in memory (mlock), so other programs' memory is compressed or swapped out instead of the model being evicted: what GPU runtimes get by wiring their buffers. Mapped (non-streaming) models only */
  cl_smart smart;   /* SMART-ROUTE: needs streaming (cache_bytes > 0), since it asks the cache what is in memory */
  int prefetch_n;   /* how many of the previous token's experts per layer to read early (best-ranked first); 0 = all that were used */ cl_policy policy; } cl_fwd_opts;
int cl_fwd_open(cl_fwd *f, const char *path, int n_ctx, const cl_fwd_opts *opts, char *err, size_t errlen);
void cl_fwd_close(cl_fwd *f);
void cl_fwd_reset(cl_fwd *f);
/* Runs one token at position f->pos (then advances it). Writes n_vocab logits. `route` may be NULL, else it must hold n_layer*n_used entries. */
int cl_fwd_step(cl_fwd *f, int token, float *logits, cl_fwd_route *route, char *err, size_t errlen);
/* Feeds n tokens. `logits` (n_vocab floats, may be NULL) receives the distribution after the LAST one. Tokens are processed in batches (CL_PF_B for qwen35, CL_PS_B for
 * every other architecture): each weight matrix is read once per batch, and in a mixture-of-experts layer each expert is read once per batch for all the tokens that chose
 * it, so a long prompt costs far fewer passes over the weights (and over the disk) than stepping token by token. Only the last token's output distribution is computed. */
#define CL_PF_B 32
#define CL_PS_B 64
int cl_fwd_prefill(cl_fwd *f, const int *tokens, int n, float *logits, char *err, size_t errlen);
#endif
