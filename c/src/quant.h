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

/* out[r] = dot(row r of W, x) for rows x cols weights of `type`, using a pool of threads. Dequantizes each block to float on the fly. */
void cl_matvec(uint32_t type, const void *w, int rows, int cols, const float *x, float *out);

/* Threads used by cl_matvec (default: the number of CPUs, at most 16). */
void cl_set_threads(int n);
int cl_get_threads(void);
#endif
