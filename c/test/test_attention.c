/* One token's attention over a long context, two ways: the grouped, chunked version used for long contexts (each key and value read once per group of query heads, chunks merged
 * with the usual rescaling) and the plain one-head-per-task version. Both are compared with a double-precision reference, on random keys and values of several softmax sharpnesses
 * and chunk lengths, for grouped heads and for plain ones. It includes forward.c to reach the static functions. */
#include "../src/forward.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#define SETENV(k, v) _putenv_s(k, v)
#define UNSETENV(k) _putenv_s(k, "")
#else
#define SETENV(k, v) setenv(k, v, 1)
#define UNSETENV(k) unsetenv(k)
#endif

static float rnd(float s) { return ((float)rand() / (float)RAND_MAX * 2 - 1) * s; }

int main(void) {
  int failed = 0;
  const int hd = 128, CTX = 700;
  const int shapes[][2] = {{8, 4}, {8, 2}, {4, 4}, {16, 2}};   /* heads, key/value heads: groups of 2, 4, 8 and none */
  for (size_t si = 0; si < sizeof shapes / sizeof shapes[0]; si++) {
    const int H = shapes[si][0], KVH = shapes[si][1], KW = KVH * hd;
    for (int sharp = 0; sharp < 3; sharp++) {
      const float qs = sharp == 0 ? 1.0f : sharp == 1 ? 4.0f : 12.0f;
      cl_fwd f; memset(&f, 0, sizeof f);
      f.n_head = H; f.n_head_kv = KVH; f.head_dim = hd; f.n_kvw = KW; f.n_ctx_max = CTX; f.n_layer = 1; f.attn_scale = 1.0f / sqrtf((float)hd);
      f.kc = malloc((size_t)CTX * KW * 2); f.vc = malloc((size_t)CTX * KW * 2);   /* the cache holds half-precision floats */
      float *Q = malloc((size_t)H * hd * 4), *A = malloc((size_t)H * hd * 4), *B = malloc((size_t)H * hd * 4);
      for (size_t i = 0; i < (size_t)CTX * KW; i++) { f.kc[i] = cl_f32_to_f16(rnd(qs)); f.vc[i] = cl_f32_to_f16(rnd(1)); }
      for (int i = 0; i < H * hd; i++) Q[i] = rnd(qs);
      static const int chunks[3] = {7, 16, 256};
      for (int ci = 0; ci < 3; ci++) {
        const int chunk = chunks[ci];
        char buf[16]; snprintf(buf, sizeof buf, "%d", chunk);
        double worst = 0, worst_plain = 0;
        for (int pos = 40; pos < CTX; pos += 53) {
          SETENV("CLUTOR_ATT_CHUNK", buf); UNSETENV("CLUTOR_PLAIN_ATTENTION");
          attend(&f, 0, Q, A, pos, 1);
          SETENV("CLUTOR_PLAIN_ATTENTION", "1");
          attend(&f, 0, Q, B, pos, 1);
          for (int h = 0; h < H; h++) {
            const int g = h / (H / KVH);
            double sc[1024], mx = -1e300, sum = 0, ref[128] = {0};
            for (int t = 0; t <= pos; t++) { double s = 0; for (int d = 0; d < hd; d++) s += (double)Q[h * hd + d] * cl_f16_to_f32(f.kc[(size_t)t * KW + g * hd + d]); sc[t] = s * f.attn_scale; if (sc[t] > mx) mx = sc[t]; }
            for (int t = 0; t <= pos; t++) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
            for (int t = 0; t <= pos; t++) for (int d = 0; d < hd; d++) ref[d] += sc[t] / sum * cl_f16_to_f32(f.vc[(size_t)t * KW + g * hd + d]);
            for (int d = 0; d < hd; d++) { const double e1 = fabs(A[h * hd + d] - ref[d]), e2 = fabs(B[h * hd + d] - ref[d]); if (e1 > worst) worst = e1; if (e2 > worst_plain) worst_plain = e2; }
          }
        }
        /* the chunked version must be as accurate as the plain one (allowing for float rounding), and close to the double-precision answer */
        const int ok = worst <= 2 * worst_plain + 2e-6 && worst < 2e-5;
        if (!ok) { failed++; printf("FAIL heads %d/%d sharpness %.0f chunk %d: grouped error %.3g, plain %.3g\n", H, KVH, qs, chunk, worst, worst_plain); }
      }
      free(f.kc); free(f.vc); free(Q); free(A); free(B); free(f.attn_part);
    }
  }
  printf("attention: %s\n", failed ? "FAILED" : "grouped attention matches the reference (4 head layouts, 3 sharpnesses, 3 chunk lengths)");
  return failed ? 1 : 0;
}
