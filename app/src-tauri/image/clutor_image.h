/*
 * Image generation inside Clutor's own process, through the stable-diffusion.cpp library (MIT). One call loads the model, makes one picture, writes a PNG and frees everything, so nothing
 * stays in memory between pictures: the chat model and the image model take turns instead of sitting side by side.
 */
#ifndef CLUTOR_IMAGE_H
#define CLUTOR_IMAGE_H
#include <stddef.h>
#include <stdint.h>

typedef void (*clutor_image_progress)(int step, int steps, float seconds, void *user);

typedef struct {
  const char *diffusion, *text_encoder, *vae;   /* the three files of an image generator */
  const char *prompt, *negative;
  int width, height, steps, turbo;              /* turbo: the model's own 8-step schedule at CFG 1 */
  float cfg;
  int64_t seed;
  int low_memory;                               /* keep weights in plain memory and move them to the GPU as needed (slower, much smaller) */
  const char *out_png;
} clutor_image_job;

/* Returns 0 and writes the PNG, or nonzero with a message in err. `cancel` is polled between steps; a nonzero value stops the picture. `stage` (optional) is told what is going on. */
int clutor_image_generate(const clutor_image_job *job, clutor_image_progress progress, void *user, volatile int *cancel, char *err, size_t errlen);

/* 1 when this build includes the image engine. */
int clutor_image_available(void);
#endif
