#include "clutor_image.h"
#include "stable-diffusion.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int clutor_image_available(void) { return 1; }

/* ---- a PNG writer with no dependencies: 8-bit RGB, stored (uncompressed) deflate blocks ---- */
static uint32_t crc_table[256];
static void crc_init(void) { for (uint32_t n = 0; n < 256; n++) { uint32_t c = n; for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1; crc_table[n] = c; } }
static uint32_t crc_update(uint32_t crc, const uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) crc = crc_table[(crc ^ p[i]) & 255] ^ (crc >> 8); return crc; }
static void put32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static int chunk(FILE *f, const char *type, const uint8_t *data, uint32_t len) {
  uint8_t h[8]; put32(h, len); memcpy(h + 4, type, 4);
  uint32_t crc = crc_update(0xffffffffu, (const uint8_t *)type, 4);
  if (len) crc = crc_update(crc, data, len);
  uint8_t t[4]; put32(t, crc ^ 0xffffffffu);
  return fwrite(h, 1, 8, f) == 8 && (!len || fwrite(data, 1, len, f) == len) && fwrite(t, 1, 4, f) == 4;
}
static int write_png(const char *path, const uint8_t *rgb, int w, int h, int channels) {
  crc_init();
  FILE *f = fopen(path, "wb");
  if (!f) return 0;
  static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  uint8_t ihdr[13]; put32(ihdr, (uint32_t)w); put32(ihdr + 4, (uint32_t)h); ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = ihdr[11] = ihdr[12] = 0;
  const size_t row = (size_t)w * 3 + 1, raw = row * (size_t)h;
  uint8_t *img = malloc(raw);
  if (!img) { fclose(f); return 0; }
  for (int y = 0; y < h; y++) { img[(size_t)y * row] = 0; for (int x = 0; x < w; x++) for (int c = 0; c < 3; c++) img[(size_t)y * row + 1 + (size_t)x * 3 + (size_t)c] = rgb[((size_t)y * w + x) * (size_t)channels + (size_t)c]; }
  size_t blocks = (raw + 65534) / 65535, zlen = 2 + raw + blocks * 5 + 4;
  uint8_t *z = malloc(zlen);
  if (!z) { free(img); fclose(f); return 0; }
  size_t o = 0; z[o++] = 0x78; z[o++] = 0x01;
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < raw; i++) { a = (a + img[i]) % 65521; b = (b + a) % 65521; }
  for (size_t pos = 0; pos < raw; pos += 65535) {
    size_t n = raw - pos < 65535 ? raw - pos : 65535;
    z[o++] = pos + n >= raw ? 1 : 0; z[o++] = (uint8_t)(n & 255); z[o++] = (uint8_t)(n >> 8); z[o++] = (uint8_t)(~n & 255); z[o++] = (uint8_t)((~n >> 8) & 255);
    memcpy(z + o, img + pos, n); o += n;
  }
  put32(z + o, (b << 16) | a); o += 4;
  int ok = fwrite(sig, 1, 8, f) == 8 && chunk(f, "IHDR", ihdr, 13) && chunk(f, "IDAT", z, (uint32_t)o) && chunk(f, "IEND", NULL, 0);
  free(img); free(z);
  return fclose(f) == 0 && ok;
}

/* ---- generation ---- */
static struct { clutor_image_progress cb; void *user; volatile int *cancel; sd_ctx_t *ctx; } g;
static void on_progress(int step, int steps, float secs, void *data) {
  (void)data;
  if (g.cb) g.cb(step, steps, secs, g.user);
  if (g.cancel && *g.cancel && g.ctx) sd_cancel_generation(g.ctx, SD_CANCEL_ALL);
}

int clutor_image_generate(const clutor_image_job *job, clutor_image_progress progress, void *user, volatile int *cancel, char *err, size_t errlen) {
  if (!job || !job->diffusion || !job->text_encoder || !job->vae || !job->prompt || !job->out_png) { snprintf(err, errlen, "an image needs the model, its text encoder, its VAE and a prompt"); return 1; }
  if (job->width < 256 || job->height < 256 || job->width > 2048 || job->height > 2048 || job->width % 16 || job->height % 16) { snprintf(err, errlen, "the picture size must be 256 to 2048 pixels on each side, in multiples of 16"); return 1; }
  sd_ctx_params_t cp;
  sd_ctx_params_init(&cp);
  cp.diffusion_model_path = job->diffusion; cp.llm_path = job->text_encoder; cp.vae_path = job->vae;
  cp.n_threads = sd_get_num_physical_cores();
  cp.enable_mmap = true;                          /* the weights are read from the files on demand and can be dropped: measured 7.1 GB instead of 15.9 GB */
  if (job->low_memory) cp.params_backend = "*=cpu";
  sd_ctx_t *ctx = new_sd_ctx(&cp);
  if (!ctx) { snprintf(err, errlen, "the image model could not be loaded (not enough memory, or the files do not belong together)"); return 1; }
  g.cb = progress; g.user = user; g.cancel = cancel; g.ctx = ctx;
  sd_set_progress_callback(on_progress, NULL);
  sd_img_gen_params_t gp;
  sd_img_gen_params_init(&gp);
  gp.prompt = job->prompt; gp.negative_prompt = job->negative ? job->negative : "";
  gp.width = job->width; gp.height = job->height; gp.seed = job->seed; gp.batch_count = 1;
  gp.sample_params.sample_method = EULER_SAMPLE_METHOD;
  gp.sample_params.sample_steps = job->steps > 0 ? job->steps : 8;
  gp.sample_params.guidance.txt_cfg = job->cfg > 0 ? job->cfg : 1.0f;
  static float turbo[9] = {1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f, 0.0f};   /* the Turbo model's published eight-step schedule */
  if (job->turbo) { gp.sample_params.custom_sigmas = turbo; gp.sample_params.custom_sigmas_count = 9; gp.sample_params.sample_steps = 8; gp.sample_params.guidance.txt_cfg = 1.0f; }
  sd_image_generation_result_t *res = NULL;
  bool ok = generate_image(ctx, &gp, &res);
  int rc = 1;
  if (ok && res && res->count > 0 && res->items[0].image.data) {
    const sd_image_t *im = &res->items[0].image;
    if (write_png(job->out_png, im->data, (int)im->width, (int)im->height, (int)im->channel)) rc = 0; else snprintf(err, errlen, "the picture could not be saved to %s", job->out_png);
  } else snprintf(err, errlen, cancel && *cancel ? "stopped" : "the picture could not be made");
  if (res) free_sd_image_generation_result(res);
  g.ctx = NULL;
  free_sd_ctx(ctx);   /* everything the model held is given back now */
  return rc;
}
