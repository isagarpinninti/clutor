#ifndef CL_QUANT_H
#define CL_QUANT_H
#include <stddef.h>
#include <stdint.h>

/* ggml tensor type ids this engine can compute with (see gguf.c for the full table). */
enum { CL_T_F32 = 0, CL_T_F16 = 1, CL_T_Q4_0 = 2, CL_T_Q5_0 = 6, CL_T_Q8_0 = 8, CL_T_Q4_K = 12, CL_T_Q5_K = 13, CL_T_Q6_K = 14, CL_T_BF16 = 30 };

/* 1 when cl_dequant_row / cl_matvec understand this type. */
int cl_type_supported(uint32_t type);
/* Elements per block and bytes per block (F32: 1 and 4). 0 for unsupported types. */
uint32_t cl_type_block(uint32_t type);
uint32_t cl_type_bytes(uint32_t type);
/* Bytes of one row of n elements (n must be a multiple of the block). */
uint64_t cl_row_bytes(uint32_t type, uint64_t n);

/* Dequantizes n elements (a multiple of the block) from one row into floats. */
void cl_dequant_row(uint32_t type, const void *src, float *dst, int n);

/* An activation vector quantized to 8-bit blocks of 256 (ggml's Q8_K layout: a scale, 256 int8, and the sum of each 16), the partner of
 * the K-quant weight types in the integer dot products. n must be a multiple of 256. */
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } cl_q8k;
typedef struct { cl_q8k *b; int n; } cl_xq;
int cl_xq_init(cl_xq *q, int n);
void cl_xq_free(cl_xq *q);
void cl_xq_quantize(cl_xq *q, const float *x);

/* One matrix-vector product: out[r] = dot(row r of W, x). K-quant weights use `xq` and integer arithmetic; other types use the float `xf`. */
/* The most tokens one batched product can hold (cl_mv.nb). */
#define CL_MV_MAX 256
typedef struct { uint32_t type; const void *w; int rows, cols; const float *xf; const cl_xq *xq; float *out;
  /* Several inputs against the same weights (prompt processing): nb > 1 makes `xq` an array of nb vectors, `xf` nb vectors spaced xf_stride floats apart, and
   * `out` nb results spaced out_stride floats apart. Each weight row is read once and multiplied with all nb inputs, so the weights are streamed once per batch. */
  int nb; size_t xf_stride, out_stride; } cl_mv;
/* Runs several products at once, their rows split into chunks over the thread pool. */
/* Runs fn(ctx, i) for i in [0, n) on the same threads as the matrix products, and returns when all are done. The calls may run in any order and at the same time. */
void cl_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx);
void cl_matvec_batch(const cl_mv *jobs, int n);
/* 1 when the weights of this type are multiplied with integer arithmetic (and so need `xq`). */
int cl_type_integer_dot(uint32_t type);
/* Name of the integer dot path compiled in: "neon+dotprod" or "scalar". */
const char *cl_quant_path(void);

/* out[r] = dot(row r of W, x) for rows x cols weights of `type`, using a pool of threads. Dequantizes each block to float on the fly. */
void cl_matvec(uint32_t type, const void *w, int rows, int cols, const float *x, float *out);

/* Threads used by cl_matvec (default: the number of CPUs, at most 16). */
void cl_set_threads(int n);
int cl_get_threads(void);
#endif
