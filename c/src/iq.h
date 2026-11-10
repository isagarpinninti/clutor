#ifndef CL_IQ_H
#define CL_IQ_H
#include <stdint.h>

/* The "IQ" quantization types of ggml: weights are picked from small tables of vectors (grids), with signs and per-group scales. They give the smallest files
 * (about 1.5 to 4.5 bits per weight), which is what lets a big model fit a small memory. */
enum { CL_T_IQ2_XXS = 16, CL_T_IQ2_XS = 17, CL_T_IQ3_XXS = 18, CL_T_IQ1_S = 19, CL_T_IQ4_NL = 20, CL_T_IQ3_S = 21, CL_T_IQ2_S = 22, CL_T_IQ4_XS = 23, CL_T_IQ1_M = 29 };

/* 1 for a type of this family. */
int cl_iq_is(uint32_t type);
/* Elements per block and bytes per block (0 when not an IQ type). */
uint32_t cl_iq_block(uint32_t type);
uint32_t cl_iq_bytes(uint32_t type);
/* The blocks of 256 as small integers: weight i is d * sc[i / 16] * q[i]. Returns 0 for types that are not 256 wide (IQ4_NL). */
int cl_iq_unpack(uint32_t type, const void *blk, int8_t *q, int16_t *sc, float *d);
/* One block to floats (block-size elements). */
void cl_iq_dequant(uint32_t type, const void *blk, float *y);
#endif
