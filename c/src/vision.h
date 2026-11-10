/*
 * The vision side of Qwen3-VL: the image encoder that turns a picture into the vectors the language model reads in place of text tokens.
 *
 * Its weights come from a separate "mmproj" GGUF file (architecture "clip", projector "qwen3vl_merger"). The picture is cut into 16 x 16 pixel patches, each patch becomes a vector,
 * 27 transformer blocks (bidirectional attention with a two-dimensional rotary embedding) mix them, and groups of 2 x 2 neighbouring patches are merged into one language-model-sized
 * vector. Three intermediate states (after blocks 8, 16 and 24) are merged the same way and returned too: the language model adds them to its first three layers ("DeepStack").
 *
 * Follows the reference implementation (Hugging Face's Qwen3VLVisionModel); the layout of the file is the one llama.cpp writes.
 */
#ifndef CL_VISION_H
#define CL_VISION_H
#include <stddef.h>
#include <stdint.h>

typedef struct cl_vision cl_vision;

typedef struct {
  int n;            /* language-model tokens for the picture: (width / 32) * (height / 32) */
  int gh, gw;       /* the grid of those tokens: rows and columns */
  int n_ds;         /* DeepStack feature sets (3) */
  float *emb;       /* n x embd */
  float *ds[3];     /* n x embd each */
} cl_vision_out;

/* Opens an mmproj file for a language model whose embedding width is `llm_embd`. NULL (with err filled) when it is not a Qwen3-VL projector or does not match. */
cl_vision *cl_vision_open(const char *path, int llm_embd, char *err, size_t errlen);
void cl_vision_close(cl_vision *v);

/* The picture is 8-bit RGB, row by row, `w` x `h` pixels; both must be multiples of 32 (see cl_vision_fit) and at most 4096 patches in all. */
int cl_vision_encode(cl_vision *v, const uint8_t *rgb, int w, int h, cl_vision_out *out, char *err, size_t errlen);
void cl_vision_out_free(cl_vision_out *o);

/* The size a picture of w x h is resized to before encoding: both sides rounded to multiples of 32, keeping the shape, with at most `max_tokens` language-model tokens
 * (one per 32 x 32 pixels) and at least 4. */
void cl_vision_fit(int w, int h, int max_tokens, int *ow, int *oh);
/* Resizes 8-bit RGB with bilinear filtering (area-averaging when shrinking by more than 2x). Returns a malloc'd buffer or NULL. */
uint8_t *cl_image_resize(const uint8_t *rgb, int w, int h, int ow, int oh);
/* Reads a binary PPM (P6, 8-bit) file. Returns a malloc'd buffer or NULL. */
uint8_t *cl_image_load_ppm(const char *path, int *w, int *h);
#endif
