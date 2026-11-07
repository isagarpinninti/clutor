#ifndef CL_KERNEL_H
#define CL_KERNEL_H
#include <stdint.h>

/*
 * Q4_0 weights times Q8_0 activations: the inner loop of a quantized matrix-vector product.
 *
 * A Q4_0 block holds 32 weights as 4-bit numbers plus one half-precision scale (18 bytes). Weight j (0..15) is the low nibble of byte j and
 * weight j+16 is the high nibble of the same byte; the value is (nibble - 8) * scale. This is ggml's layout, so a real Q4_0 tensor can be read
 * as it is. The activations are quantized to int8 blocks of 32 with a float scale (our own format, 36 bytes), and the dot product is exact
 * integer arithmetic inside a block, so the vector paths agree with the scalar path up to the order in which block sums are added.
 *
 * Compile-time paths: ARM NEON (with the dot-product instruction when the CPU target has it), AVX2, and portable C11. -DCL_FORCE_SCALAR picks
 * the portable one on any machine; the tests use that to compare it with the vector path.
 */
#define CL_QK 32

typedef struct { uint16_t d; uint8_t qs[16]; } cl_block_q4_0;   /* 18 bytes */
typedef struct { float d; int8_t qs[CL_QK]; } cl_block_q8_0;    /* 36 bytes */
_Static_assert(sizeof(cl_block_q4_0) == 18, "a Q4_0 block is 18 bytes");
_Static_assert(sizeof(cl_block_q8_0) == 36, "a Q8_0 activation block is 36 bytes");

/* The path this build uses: "neon+dotprod", "neon", "avx2" or "scalar". */
const char *cl_kernel_name(void);

/* Half to single precision. The software version handles every value (subnormals, infinities, NaN) and is the reference; where the processor converts in one instruction
 * (Arm with half-precision support, x86 with F16C) that instruction is used, because this runs twice per 256 weights in every matrix product and in software it cost as much
 * as the dot product itself. The tests compare the two on all 65,536 values. */
float cl_f16_to_f32_soft(uint16_t h);
#if defined(__aarch64__) && !defined(CL_FORCE_SCALAR) && (defined(__ARM_FP16_FORMAT_IEEE) || defined(__clang__))
#include <string.h>
static inline float cl_f16_to_f32(uint16_t h) { __fp16 x; memcpy(&x, &h, sizeof x); return (float)x; }
#define CL_F16_HARDWARE 1
#elif defined(__F16C__) && !defined(CL_FORCE_SCALAR)
#include <immintrin.h>
static inline float cl_f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
#define CL_F16_HARDWARE 1
#else
static inline float cl_f16_to_f32(uint16_t h) { return cl_f16_to_f32_soft(h); }
#endif

/* Quantizes n floats (a multiple of 32) to n/32 blocks. */
void cl_quantize_q8_0(const float *x, cl_block_q8_0 *y, int n);

/* The dot product of nblocks weight blocks with nblocks activation blocks. */
float cl_dot_q4_0_q8_0(int nblocks, const cl_block_q4_0 *w, const cl_block_q8_0 *a);
float cl_dot_q4_0_q8_0_scalar(int nblocks, const cl_block_q4_0 *w, const cl_block_q8_0 *a);

/* rows x blocks_per_row weights times one activation vector of blocks_per_row blocks. */
void cl_gemv_q4_0(const cl_block_q4_0 *w, int rows, int blocks_per_row, const cl_block_q8_0 *a, float *out);

/*
 * Runs the product over a buffer of streamed bytes as if it were Q4_0 rows of blocks_per_row blocks (a trailing partial row is ignored) and
 * folds the finite results into a digest. Used by the streaming engine: the bytes of a Q4_K or IQ2 model are not Q4_0, so what is
 * computed is not that model's maths; the point is that the loaded bytes go through a real quantized kernel.
 */
uint64_t cl_q4_0_pass(const void *bytes, uint64_t n, const cl_block_q8_0 *act, int blocks_per_row);

#endif
