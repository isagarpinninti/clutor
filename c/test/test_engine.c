#define _GNU_SOURCE
#include "cache.h"
#include "gguf.h"
#include "kernel.h"
#include "route.h"
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0, checks = 0;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int touch_key(cl_cache *c, uint32_t key) { int slot; int r = cl_cache_access(c, key, &slot); if (slot >= 0) cl_cache_unpin(c, slot); return r; }

static void test_lru(void) {
  cl_cache *c = cl_cache_new(3, 0, 100, CL_LRU, 0);
  CHECK(touch_key(c, 1) == 0 && touch_key(c, 2) == 0 && touch_key(c, 3) == 0, "first touches miss");
  CHECK(touch_key(c, 1) == 1, "1 is still there");
  touch_key(c, 4); /* evicts 2, the least recently used */
  CHECK(cl_cache_find(c, 2) < 0, "2 was evicted");
  CHECK(cl_cache_find(c, 1) >= 0 && cl_cache_find(c, 3) >= 0 && cl_cache_find(c, 4) >= 0, "the others stayed");
  CHECK(cl_cache_count(c) == 3, "count is capacity");
  cl_cache_free(c);
}

static void test_pins(void) {
  cl_cache *c = cl_cache_new(2, 0, 100, CL_LRU, 0);
  int a, b, d;
  cl_cache_access(c, 1, &a); cl_cache_access(c, 2, &b); /* both pinned */
  int r = cl_cache_access(c, 3, &d);
  CHECK(r == -1 && d == -1, "nothing evictable: the access bypasses the cache (got %d)", r);
  CHECK(cl_cache_find(c, 1) >= 0 && cl_cache_find(c, 2) >= 0, "pinned entries survived");
  CHECK(cl_cache_find(c, 3) < 0, "the bypassed key was not kept");
  cl_cache_unpin(c, a);
  r = cl_cache_access(c, 3, &d);
  CHECK(r == 0 && cl_cache_find(c, 1) < 0, "once 1 is unpinned it can be evicted");
  cl_cache_free(c);
}

static void test_stage(void) {
  cl_cache *c = cl_cache_new(2, 2, 100, CL_LRU, 0);
  touch_key(c, 1); touch_key(c, 2);
  int s = cl_cache_stage(c, 10);
  CHECK(s >= 0 && cl_cache_list(c, s) == CL_STAGE, "a prefetch goes to staging");
  CHECK(cl_cache_find(c, 1) >= 0 && cl_cache_find(c, 2) >= 0, "staging did not evict cache entries");
  CHECK(cl_cache_stage(c, 10) < 0, "already staged");
  cl_cache_unpin(c, s);
  int slot, r = cl_cache_access(c, 10, &slot);
  CHECK(r == 2 && slot == s, "a staged expert that is wanted is adopted without a new slot (r=%d)", r);
  CHECK(cl_cache_get_stats(c).stage_hits == 1, "counted as a staging hit");
  cl_cache_unpin(c, slot);
  CHECK(cl_cache_count(c) == 2, "adoption respected capacity");
  /* staging overflow drops the oldest unused prefetch and counts it as wasted */
  int a = cl_cache_stage(c, 20); cl_cache_unpin(c, a);
  int b = cl_cache_stage(c, 21); cl_cache_unpin(c, b);
  int d = cl_cache_stage(c, 22); cl_cache_unpin(c, d);
  CHECK(cl_cache_find(c, 20) < 0 && cl_cache_get_stats(c).stage_wasted == 1, "oldest prefetch dropped and counted");
  cl_cache_free(c);
}

/* A hot set that is read again and again, then a one-time scan much larger than the cache. */
static int hot_survivors(cl_policy p, uint32_t window_pct) {
  cl_cache *c = cl_cache_new(32, 0, 5000, p, window_pct);
  for (int round = 0; round < 20; round++) for (uint32_t k = 0; k < 16; k++) touch_key(c, k);
  for (uint32_t k = 1000; k < 1600; k++) touch_key(c, k);
  int alive = 0;
  for (uint32_t k = 0; k < 16; k++) if (cl_cache_find(c, k) >= 0) alive++;
  cl_cache_free(c);
  return alive;
}
static void test_scan_resistance(void) {
  int lru = hot_survivors(CL_LRU, 0), tiny = hot_survivors(CL_WTINYLFU, 0);
  CHECK(lru == 0, "LRU loses the whole hot set to a scan (kept %d)", lru);
  CHECK(tiny >= 12, "W-TinyLFU keeps most of the hot set through a scan (kept %d of 16)", tiny);
  cl_cache *c = cl_cache_new(32, 0, 5000, CL_WTINYLFU, 0);
  for (int round = 0; round < 20; round++) for (uint32_t k = 0; k < 16; k++) touch_key(c, k);
  for (uint32_t k = 1000; k < 1600; k++) touch_key(c, k);
  CHECK(cl_cache_get_stats(c).admit_rejects > 400, "the scan was turned away at the door (%llu rejects)", (unsigned long long)cl_cache_get_stats(c).admit_rejects);
  CHECK(cl_cache_freq(c, 3) > cl_cache_freq(c, 1500), "the sketch knows which is popular");
  cl_cache_free(c);
}

static void test_invariants(void) {
  /* random traffic with random pins: the cache never holds more than its capacity or the same key twice */
  cl_cache *c = cl_cache_new(20, 6, 300, CL_ADAPTIVE, 0);
  unsigned x = 12345;
  int pinned[64], np = 0;
  for (int i = 0; i < 50000; i++) {
    x = x * 1664525u + 1013904223u;
    uint32_t key = (x >> 8) % (i % 1000 < 500 ? 30 : 300);
    int slot, r;
    if ((x >> 4) % 5 == 0) { slot = cl_cache_stage(c, key); if (slot >= 0) { if (np < 64) pinned[np++] = slot; else cl_cache_unpin(c, slot); } }
    else { r = cl_cache_access(c, key, &slot); if (slot >= 0) { CHECK(cl_cache_slot_key(c, slot) == key, "slot holds the key asked for (r=%d)", r); if (np < 64) pinned[np++] = slot; else cl_cache_unpin(c, slot); } }
    if (np > 8) { for (int k = 0; k < np; k++) cl_cache_unpin(c, pinned[k]); np = 0; }
    if (cl_cache_count(c) > 20) { CHECK(0, "cache over capacity: %u", cl_cache_count(c)); break; }
  }
  for (uint32_t k = 0; k < 300; k++) { int s = cl_cache_find(c, k); if (s >= 0) CHECK(cl_cache_slot_key(c, s) == k, "map and slots agree for key %u", k); }
  cl_cache_free(c);
}

static int resident_set[CL_MAX_E];
static int is_resident(void *ctx, int e) { (void)ctx; return resident_set[e]; }

static void test_route(void) {
  float logits[64];
  for (int i = 0; i < 64; i++) logits[i] = (float)(-0.12 * i); /* expert 0 is the most likely, then 1, 2... */
  cl_route_result r;
  CHECK(cl_route(logits, 64, 8, NULL, NULL, NULL, &r) == 0, "plain routing runs");
  for (int i = 0; i < 8; i++) CHECK(r.selected[i] == i, "plain routing takes the top-K in order (slot %d got %d)", i, r.selected[i]);
  CHECK(r.substituted == 0, "no substitutions without smart-route");
  CHECK(cl_route(logits, 8, 9, NULL, NULL, NULL, &r) < 0 && cl_route(logits, 64, 0, NULL, NULL, NULL, &r) < 0, "bad arguments are refused");

  memset(resident_set, 0, sizeof resident_set);
  for (int e = 8; e < 14; e++) resident_set[e] = 1; /* only experts just outside the top-8 are in memory */
  cl_smart s = {1, 2, 12, 0.015};
  cl_route(logits, 64, 8, &s, is_resident, NULL, &r);
  CHECK(r.selected[0] == 0 && r.selected[1] == 1, "the true top-J are kept");
  CHECK(r.max_delta <= 0.015 + 1e-12, "no substitution moves more than tau (%.6f)", r.max_delta);
  int seen[64] = {0}, dup = 0;
  for (int i = 0; i < 8; i++) { if (seen[r.selected[i]]++) dup++; }
  CHECK(dup == 0, "selected experts are distinct");
  cl_smart loose = {1, 2, 12, 1e9};
  cl_route_result r2;
  cl_route(logits, 64, 8, &loose, is_resident, NULL, &r2);
  CHECK(r2.substituted >= r.substituted && r2.substituted == 4, "with no bound, every cold slot is filled from the 4 resident experts inside the top-M (got %d)", r2.substituted);
  CHECK(r2.dropped_mass >= r.dropped_mass, "a looser bound moves more probability");
  cl_smart zero = {1, 2, 12, 0.0};
  cl_route_result r3;
  cl_route(logits, 64, 8, &zero, is_resident, NULL, &r3);
  CHECK(r3.substituted == 0, "tau = 0 never substitutes");
  cl_smart narrowM = {1, 2, 8, 1e9};
  cl_route_result r4;
  cl_route(logits, 64, 8, &narrowM, is_resident, NULL, &r4);
  CHECK(r4.substituted == 0, "M = K leaves no candidates outside the top-K");
  for (int e = 0; e < 8; e++) resident_set[e] = 1;
  cl_route_result r5;
  cl_route(logits, 64, 8, &s, is_resident, NULL, &r5);
  CHECK(r5.substituted == 0, "nothing is replaced when the router's own choices are all in memory");
}

static void test_gguf(const char *fixture) {
  cl_model m; char err[256] = {0};
  CHECK(cl_model_open(&m, "/definitely/not/here.gguf", err, sizeof err) != 0 && err[0], "a missing file is an error with a message");
  char bad[] = "/tmp/clutor-bad-XXXXXX";
  int fd = mkstemp(bad);
  if (fd >= 0) { char junk[64]; memset(junk, 'x', sizeof junk); if (write(fd, junk, sizeof junk) < 0) {} close(fd); }
  CHECK(cl_model_open(&m, bad, err, sizeof err) != 0, "a file that is not GGUF is refused: %s", err);
  unlink(bad);
  if (!fixture) { printf("(skipping GGUF fixture checks: no file given)\n"); return; }
  CHECK(cl_model_open(&m, fixture, err, sizeof err) == 0, "the fixture opens: %s", err);
  if (m.n_layers) {
    CHECK(strcmp(m.arch, "llama") == 0, "architecture read (%s)", m.arch);
    CHECK(m.n_layers > 0 && m.moe_layers == 0 && m.n_experts == 0, "a dense model has layers but no experts");
    CHECK(m.total_bytes > 0 && m.total_bytes <= m.file_size, "tensor bytes fit in the file");
    CHECK(m.unsized_tensors == 0, "every tensor type understood");
    cl_model_close(&m);
  }
}


/* ---- the Q4_0 x Q8_0 kernel ---- */
static uint64_t kseed = 12345;
static uint32_t krand(void) { kseed = kseed * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(kseed >> 33); }

/* Reference by definition: dequantize both sides to doubles and multiply. Shares no code with the kernel. */
static double ref_dot(int nb, const cl_block_q4_0 *w, const cl_block_q8_0 *a) {
  double sum = 0;
  for (int b = 0; b < nb; b++) {
    double dw = cl_f16_to_f32(w[b].d);
    for (int j = 0; j < 16; j++) {
      double lo = ((int)(w[b].qs[j] & 15) - 8) * dw, hi = ((int)(w[b].qs[j] >> 4) - 8) * dw;
      sum += lo * ((double)a[b].qs[j] * a[b].d) + hi * ((double)a[b].qs[j + 16] * a[b].d);
    }
  }
  return sum;
}

static void random_blocks(int nb, cl_block_q4_0 *w, cl_block_q8_0 *a, int extreme) {
  for (int b = 0; b < nb; b++) {
    /* scales from 2^-8 to 2^3 as half floats: exponent bits 7..18 */
    w[b].d = (uint16_t)(((7 + krand() % 12) << 10) | (krand() & 0x3ff));
    for (int j = 0; j < 16; j++) w[b].qs[j] = extreme ? (krand() & 1 ? 0x00 : 0xff) : (uint8_t)krand();
    a[b].d = 0.001f + (float)(krand() % 1000) / 1000.0f;
    for (int j = 0; j < 32; j++) a[b].qs[j] = extreme ? (krand() & 1 ? 127 : -127) : (int8_t)((int)(krand() % 255) - 127);
  }
}

static void test_kernel(void) {
  printf("kernel path: %s\n", cl_kernel_name());
  CHECK(cl_f16_to_f32(0x3c00) == 1.0f && cl_f16_to_f32(0xc000) == -2.0f && cl_f16_to_f32(0x3555) > 0.333f && cl_f16_to_f32(0x3555) < 0.3334f, "half floats: 1, -2, one third");
  CHECK(cl_f16_to_f32(0x0001) == ldexpf(1.0f, -24) && cl_f16_to_f32(0x0200) == ldexpf(1.0f, -15) && isinf(cl_f16_to_f32(0x7c00)) && isnan(cl_f16_to_f32(0x7e00)), "half floats: subnormal, infinity, NaN");
  CHECK(sizeof(cl_block_q4_0) == 18 && sizeof(cl_block_q8_0) == 36, "block sizes");

  /* hand-computed: low nibbles 8 (weight 0), high nibbles 9 (weight 1), scale 1.0; activations -16..15 at scale 0.5.
     Only weights 16..31 count, so the sum is (0+1+...+15) * 1.0 * 0.5 = 60 */
  cl_block_q4_0 w = { 0x3c00, {0} };
  cl_block_q8_0 a = { 0.5f, {0} };
  for (int j = 0; j < 16; j++) w.qs[j] = 0x98;
  for (int j = 0; j < 32; j++) a.qs[j] = (int8_t)(j - 16);
  CHECK(cl_dot_q4_0_q8_0_scalar(1, &w, &a) == 60.0f, "scalar: hand-computed block is 60 (got %g)", cl_dot_q4_0_q8_0_scalar(1, &w, &a));
  CHECK(cl_dot_q4_0_q8_0(1, &w, &a) == 60.0f, "%s: hand-computed block is 60 (got %g)", cl_kernel_name(), cl_dot_q4_0_q8_0(1, &w, &a));
  /* the nibble order: low nibbles pair with activations 0..15, high nibbles with 16..31 */
  for (int j = 0; j < 16; j++) w.qs[j] = 0x89;
  CHECK(cl_dot_q4_0_q8_0(1, &w, &a) == 0.5f * (-16 + -15 + -14 + -13 + -12 + -11 + -10 + -9 + -8 + -7 + -6 + -5 + -4 + -3 + -2 + -1), "low nibbles meet activations 0..15");
  for (int j = 0; j < 16; j++) w.qs[j] = 0x88;
  CHECK(cl_dot_q4_0_q8_0(1, &w, &a) == 0.0f, "all weights zero gives zero");

  /* quantizing keeps every value within half a step, and a zero vector stays zero */
  float x[64];
  for (int i = 0; i < 64; i++) x[i] = ((float)(krand() % 2001) - 1000.0f) / 100.0f;
  cl_block_q8_0 q[2];
  cl_quantize_q8_0(x, q, 64);
  for (int i = 0; i < 64; i++) CHECK(fabsf(q[i / 32].qs[i % 32] * q[i / 32].d - x[i]) <= q[i / 32].d * 0.5f + 1e-6f, "q8 round trip of %g", x[i]);
  float z[32] = {0};
  cl_quantize_q8_0(z, q, 32);
  CHECK(q[0].d == 0.0f && q[0].qs[7] == 0, "zero in, zero out");

  /* the vector path against the definition and against the portable path, over many random rows */
  enum { NB = 40 };
  cl_block_q4_0 ws[NB];
  cl_block_q8_0 as[NB];
  for (int trial = 0; trial < 500; trial++) {
    int extreme = trial % 5 == 0, nb = 1 + (int)(krand() % NB);
    random_blocks(nb, ws, as, extreme);
    double ref = ref_dot(nb, ws, as);
    float fast = cl_dot_q4_0_q8_0(nb, ws, as), slow = cl_dot_q4_0_q8_0_scalar(nb, ws, as);
    double tol = 1e-4 * fabs(ref) + 1e-3;
    CHECK(fabs(fast - ref) <= tol, "%s vs definition: %g vs %g (trial %d)", cl_kernel_name(), fast, ref, trial);
    CHECK(fabs(slow - ref) <= tol, "scalar vs definition: %g vs %g (trial %d)", slow, ref, trial);
    CHECK(fabs(fast - slow) <= 1e-5 * fabs(slow) + 1e-4, "%s vs scalar: %g vs %g (trial %d)", cl_kernel_name(), fast, slow, trial);
  }

  /* gemv rows are independent dot products; the byte pass ignores a trailing partial row and is deterministic */
  enum { ROWS = 7 };
  cl_block_q4_0 mat[ROWS * 8];
  cl_block_q8_0 act[8];
  random_blocks(8, mat, act, 0);
  for (int i = 8; i < ROWS * 8; i++) mat[i] = mat[i % 8], mat[i].qs[0] ^= (uint8_t)i;
  float out[ROWS];
  cl_gemv_q4_0(mat, ROWS, 8, act, out);
  for (int r = 0; r < ROWS; r++) CHECK(out[r] == cl_dot_q4_0_q8_0(8, mat + r * 8, act), "gemv row %d", r);
  uint64_t d1 = cl_q4_0_pass(mat, sizeof mat, act, 8), d2 = cl_q4_0_pass(mat, sizeof mat, act, 8);
  CHECK(d1 == d2 && d1 != 0, "the byte pass is deterministic");
  CHECK(cl_q4_0_pass(mat, 8 * sizeof(cl_block_q4_0) - 1, act, 8) == 0, "less than one row is nothing to compute");
  uint8_t flipped[sizeof mat];
  memcpy(flipped, mat, sizeof mat);
  flipped[5] ^= 0x10; /* a different weight in row 0 */
  CHECK(cl_q4_0_pass(flipped, sizeof mat, act, 8) != d1, "changing a byte changes the digest");
  /* an Inf or NaN scale must not poison the digest */
  cl_block_q4_0 bad[8];
  memcpy(bad, mat, sizeof bad);
  bad[0].d = 0x7c00; bad[1].d = 0x7e00;
  CHECK(cl_q4_0_pass(bad, sizeof bad, act, 8) == (0ull * 1099511628211ull + 0), "a non-finite row counts as zero");
}

int main(int argc, char **argv) {
  test_lru(); test_pins(); test_stage(); test_scan_resistance(); test_invariants(); test_route(); test_kernel();
  test_gguf(argc > 1 ? argv[1] : getenv("CLUTOR_FIXTURE"));
  printf("%d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
