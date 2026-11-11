#define _GNU_SOURCE
#include "cache.h"
#include "gguf.h"
#include "kernel.h"
#include "forward.h"
#include "quant.h"
#include "backend.h"
#include "jsongate.h"
#include "iq.h"
#include "vision.h"
#include "meta.h"
#include "xs.h"
#include <math.h>
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
  { /* the fast conversion agrees with the reference on every one of the 65,536 half-precision bit patterns (NaNs only need to be NaNs) */
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
      float a = cl_f16_to_f32((uint16_t)h), b = cl_f16_to_f32_soft((uint16_t)h);
      if (isnan(b) ? !isnan(a) : memcmp(&a, &b, sizeof a) != 0) bad++;
    }
    CHECK(bad == 0, "half to single conversion: the fast path equals the reference for all 65,536 values");
  }
  { /* the cache a model that does not fit gets: 45 percent of memory, at most 60 percent of what is free, 256 MiB at least, never more than all the experts */
    const uint64_t G = (uint64_t)1 << 30, M = (uint64_t)1 << 20;
    CHECK(cl_should_lock(4 * G, 16 * G, 12 * G) == 1, "lock: a model well inside memory is pinned");
    CHECK(cl_should_lock(12 * G, 16 * G, 14 * G) == 0, "lock: a model over 60 percent of memory is not pinned");
    CHECK(cl_should_lock(6 * G, 16 * G, 5 * G) == 0, "lock: a model bigger than the free memory is not pinned");
    CHECK(cl_should_lock(4 * G, 0, 0) == 1, "lock: unknown memory does not stop pinning a model");
    CHECK(cl_auto_cache_bytes(16 * G, 0, 40 * G) == 16 * G / 100 * 45, "auto cache: 45 percent of memory when free memory is unknown");
    CHECK(cl_auto_cache_bytes(16 * G, 14 * G, 40 * G) == 16 * G / 100 * 45, "auto cache: 45 percent of memory when plenty is free");
    CHECK(cl_auto_cache_bytes(16 * G, 4 * G, 40 * G) == 4 * G / 100 * 60, "auto cache: at most 60 percent of what is free when other programs hold the rest");
    CHECK(cl_auto_cache_bytes(16 * G, 100 * M, 40 * G) == 256 * M, "auto cache: never below 256 MiB");
    CHECK(cl_auto_cache_bytes(64 * G, 60 * G, 3 * G) == 3 * G, "auto cache: never more than all the experts");
  }
  { /* single to half (the key/value cache is stored this way): the fast path equals the software reference, every half value survives the round trip, and the rule is round to nearest even */
    int bad = 0, trips = 0, ties = 0;
    for (uint32_t h = 0; h < 65536; h++) { const float a = cl_f16_to_f32_soft((uint16_t)h); if (!isnan(a) && cl_f32_to_f16_soft(a) != (uint16_t)h) trips++; }
    uint32_t seed = 12345u;
    for (int i = 0; i < 4000000; i++) {
      seed = seed * 1664525u + 1013904223u;
      uint32_t bits = seed;
      if (i % 3 == 0) bits = (seed & 0x80000000u) | ((((seed >> 23) % 60u) + 90u) << 23) | (seed & 0x7fffffu);   /* around the half range, where the rounding happens */
      float f; memcpy(&f, &bits, sizeof f);
      const uint16_t a = cl_f32_to_f16(f), b = cl_f32_to_f16_soft(f);
      if (isnan(f) ? ((a & 0x7c00u) != 0x7c00u || !(a & 0x3ffu)) : a != b) bad++;
    }
    /* ties: halfway between two halves goes to the even one (1 + 2^-11 is halfway between 1 and 1 + 2^-10) */
    if (cl_f32_to_f16_soft(1.0f + ldexpf(1.0f, -11)) != 0x3c00u) ties++;
    if (cl_f32_to_f16_soft(1.0f + 3.0f * ldexpf(1.0f, -11)) != 0x3c02u) ties++;
    if (cl_f32_to_f16_soft(65520.0f) != 0x7c00u || cl_f32_to_f16_soft(65519.0f) != 0x7bffu) ties++;   /* overflow to infinity starts halfway above the largest half */
    if (cl_f32_to_f16_soft(ldexpf(1.0f, -25)) != 0x0000u || cl_f32_to_f16_soft(ldexpf(1.5f, -25)) != 0x0001u) ties++;   /* the smallest subnormal, and halfway below it */
    CHECK(bad == 0 && trips == 0 && ties == 0, "single to half conversion: fast path equals the reference on 4 million values, every half round-trips, ties go to even");
  }
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

static void test_quant(void) {
  /* Q8_0: scale 1.0, values -16..15 */
  uint8_t b8[34] = {0x00, 0x3c};
  for (int i = 0; i < 32; i++) b8[2 + i] = (uint8_t)(int8_t)(i - 16);
  float y[256];
  cl_dequant_row(CL_T_Q8_0, b8, y, 32);
  CHECK(y[0] == -16.0f && y[31] == 15.0f, "Q8_0 dequantizes to scale * value");
  /* Q4_K: d = 1, dmin = 0, every sub-block scale 1, every byte 0x21: low nibbles read 1, high nibbles read 2, in groups of 32 */
  uint8_t q4k[144] = {0};
  q4k[0] = 0x00; q4k[1] = 0x3c;
  for (int i = 0; i < 4; i++) { q4k[4 + i] = 1; q4k[12 + i] = 1; }   /* scales 0..3 */
  for (int i = 8; i < 12; i++) q4k[4 + i] = 1;                       /* scales 4..7 (low nibble of bytes 8..11, mins stay 0) */
  memset(q4k + 16, 0x21, 128);
  cl_dequant_row(CL_T_Q4_K, q4k, y, 256);
  CHECK(y[0] == 1.0f && y[31] == 1.0f && y[32] == 2.0f && y[63] == 2.0f && y[64] == 1.0f && y[255] == 2.0f, "Q4_K: low nibbles then high nibbles per 64 values");
  /* Q6_K: all-zero quants decode to -32 * scale * d */
  uint8_t q6k[210] = {0};
  for (int i = 0; i < 16; i++) q6k[192 + i] = 1;
  q6k[208] = 0x00; q6k[209] = 0x3c;
  cl_dequant_row(CL_T_Q6_K, q6k, y, 256);
  CHECK(y[0] == -32.0f && y[255] == -32.0f, "Q6_K: zero quants are -32");
  /* the threaded matvec equals a plain dot product, row by row */
  enum { R = 301, C = 512 };
  static float w[R * C], x[C], out[R];
  uint32_t s = 12345;
  for (int i = 0; i < R * C; i++) { s = s * 1664525u + 1013904223u; w[i] = (float)(s >> 8) / (1 << 24) - 0.5f; }
  for (int i = 0; i < C; i++) { s = s * 1664525u + 1013904223u; x[i] = (float)(s >> 8) / (1 << 24) - 0.5f; }
  cl_set_threads(4);
  cl_matvec(CL_T_F32, w, R, C, x, out);
  int bad = 0;
  for (int r = 0; r < R; r++) { double a = 0; for (int c = 0; c < C; c++) a += (double)w[r * C + c] * x[c]; if (fabs(a - out[r]) > 1e-3) bad++; }
  CHECK(bad == 0, "threaded matvec matches a plain dot product (%d rows differ)", bad);
}

static void test_meta(const char *fixture) {
  if (!fixture) return;
  FILE *fp = fopen(fixture, "rb");
  if (!fp) return;
  static uint8_t buf[1 << 20];
  size_t n = fread(buf, 1, sizeof buf, fp);
  fclose(fp);
  cl_meta m; char err[128];
  CHECK(cl_meta_parse(&m, buf, n, err, sizeof err) == 0, "metadata parses: %s", err);
  cl_str arch;
  CHECK(cl_meta_str(&m, "general.architecture", &arch) && arch.len == 5 && !memcmp(arch.p, "llama", 5), "architecture is read");
  uint64_t layers = 0, nt = 0;
  CHECK(cl_meta_u64(&m, "llama.block_count", &layers) && layers > 0, "a numeric key is read");
  cl_str *toks = cl_meta_str_array(&m, "tokenizer.ggml.tokens", &nt);
  CHECK(toks && nt > 0, "the token table is read");
  free(toks);
  CHECK(cl_meta_get(&m, "no.such.key") == NULL, "a missing key is NULL");
  cl_meta_free(&m);
}

/* The integer dot products must equal "dequantize the weights, dequantize the 8-bit activations, multiply in double". */
static void test_integer_dot(void) {
  enum { ROWS = 40, COLS = 768 };   /* 3 blocks of 256 per row */
  uint32_t s = 777;
  static float x[COLS];
  static uint8_t w4[ROWS * 3 * 144], w6[ROWS * 3 * 210];
  for (int i = 0; i < COLS; i++) { s = s * 1664525u + 1013904223u; x[i] = ((float)(s >> 8) / (1 << 24) - 0.5f) * (i % 97 == 0 ? 8.0f : 1.0f); }
  for (size_t i = 0; i < sizeof w4; i++) { s = s * 1664525u + 1013904223u; w4[i] = (uint8_t)(s >> 16); }
  for (size_t i = 0; i < sizeof w6; i++) { s = s * 1664525u + 1013904223u; w6[i] = (uint8_t)(s >> 16); }
  for (int b = 0; b < ROWS * 3; b++) {   /* sane scales: half-precision 0.01 .. 0.04, so no Inf or NaN */
    uint16_t d = (uint16_t)(0x2400 + (b % 16) * 8), dm = (uint16_t)(0x2000 + (b % 7) * 8);
    memcpy(w4 + b * 144, &d, 2); memcpy(w4 + b * 144 + 2, &dm, 2);
    memcpy(w6 + b * 210 + 208, &d, 2);
  }
  cl_xq q;
  CHECK(cl_xq_init(&q, COLS), "activation buffer");
  cl_xq_quantize(&q, x);
  float xd[COLS];
  for (int b = 0; b < COLS / 256; b++) for (int j = 0; j < 256; j++) xd[b * 256 + j] = q.b[b].d * q.b[b].qs[j];
  for (int pass = 0; pass < 2; pass++) {
    uint32_t type = pass ? CL_T_Q6_K : CL_T_Q4_K;
    const uint8_t *w = pass ? w6 : w4;
    static float out[ROWS];
    cl_mv job = {type, w, ROWS, COLS, x, &q, out, 1, 0, 0};
    cl_matvec_batch(&job, 1);
    int bad = 0;
    float row[COLS];
    for (int r = 0; r < ROWS; r++) {
      cl_dequant_row(type, w + (size_t)r * cl_row_bytes(type, COLS), row, COLS);
      double ref = 0, mag = 0;
      for (int c = 0; c < COLS; c++) { ref += (double)row[c] * xd[c]; mag += fabs((double)row[c] * xd[c]); }
      if (fabs(ref - out[r]) > 1e-4 * mag + 1e-6) bad++;
    }
    CHECK(bad == 0, "%s integer dot matches dequantize-and-multiply on %d of %d rows (%s)", pass ? "Q6_K" : "Q4_K", ROWS - bad, ROWS, cl_quant_path());
  }
  cl_xq_free(&q);
}

/* The JSON checker used for constrained decoding: complete objects are accepted and finish, broken text is refused at the first bad character, and the answer does not depend on how the text is cut into pieces. */

/* ---- backends: a backend gets the products it supports, the CPU keeps the rest, and the result is the same ---- */
static int be_calls, be_jobs, be_types_seen;
static int be_supports(uint32_t type) { return type == CL_T_F32; }       /* handles F32 only */
static void be_run(const cl_mv *jobs, int n) {
  be_calls++; be_jobs += n;
  for (int j = 0; j < n; j++) { if (jobs[j].type != CL_T_F32) be_types_seen++; cl_backend_cpu()->matvec_batch(&jobs[j], 1); }
}
static const cl_backend counting = {"counting", be_supports, be_run};
static void test_backend(void) {
  CHECK(strcmp(cl_backend_active()->name, "cpu") == 0, "the default backend is cpu");
  CHECK(!cl_backend_select("nope"), "an unknown backend is refused");
  CHECK(strcmp(cl_backend_active()->name, "cpu") == 0, "and the active one is unchanged");
  CHECK(cl_backend_register(&counting) == 1 && cl_backend_register(&counting) == 0, "a backend registers once");
  enum { R = 8, C = 256 };
  float w[R * C], x[C], ref[R], got[R], ref2[R], got2[R];
  for (int i = 0; i < R * C; i++) w[i] = (float)((i * 37) % 19 - 9) / 7.0f;
  for (int i = 0; i < C; i++) x[i] = (float)((i * 11) % 13 - 6) / 5.0f;
  cl_matvec(CL_T_F32, w, R, C, x, ref);
  cl_backend_select("counting");
  be_calls = be_jobs = be_types_seen = 0;
  cl_matvec(CL_T_F32, w, R, C, x, got);
  int same = 1; for (int i = 0; i < R; i++) if (got[i] != ref[i]) same = 0;
  CHECK(be_calls == 1 && be_jobs == 1 && same, "an F32 product goes to the backend and gives the same numbers");
  /* a type the backend does not support (F16) stays on the CPU */
  uint16_t h[R * C]; for (int i = 0; i < R * C; i++) { float v = w[i]; uint32_t u; memcpy(&u, &v, 4); h[i] = (uint16_t)(u >> 16); }   /* not real F16: compared with itself below */
  cl_backend_select("cpu"); cl_matvec(CL_T_BF16, h, R, C, x, ref2);
  cl_backend_select("counting"); be_calls = be_jobs = 0;
  cl_matvec(CL_T_BF16, h, R, C, x, got2);
  int same2 = 1; for (int i = 0; i < R; i++) if (got2[i] != ref2[i]) same2 = 0;
  CHECK(be_calls == 0 && same2, "a type the backend does not support runs on the CPU, unchanged");
  /* one batch with both kinds is split */
  cl_xq none = {0}; (void)none;
  cl_mv jobs[2] = {{CL_T_F32, w, R, C, x, NULL, got, 1, 0, 0}, {CL_T_BF16, h, R, C, x, NULL, got2, 1, 0, 0}};
  be_calls = be_jobs = 0;
  cl_matvec_batch(jobs, 2);
  CHECK(be_calls == 1 && be_jobs == 1, "a mixed batch sends only the supported product to the backend");
  cl_backend_select("cpu");
}

static void test_json_gate(void) {
  static const char *const good[] = {"{}", " {\"a\": 1}", "{\"tool\": \"write_file\", \"args\": {\"path\": \"a.txt\", \"content\": \"line1\\nline2 \\\"q\\\" \\u00e9 \xc3\xa9\"}}",
    "{\"a\": [1, -2.5, 3e10, 0, 1E-2, true, false, null, [], {}, \"x\"], \"b\": {\"c\": {}}}\n", "{\"n\":-0.5e+3}"};
  static const char *const bad[] = {"[1]", "1", "\"s\"", "{a:1}", "{\"a\" 1}", "{\"a\": 01}", "{\"a\": 1,}", "{\"a\": tru}", "{\"a\": \"\n\"}", "{\"a\": \"\\x\"}", "{\"a\": 1.}", "{\"a\": -}", "{\"a\": [1}", "{}}", "{} x", "{\"a\": \"\\u12g4\"}", "{\"a\": 1e}"};
  for (size_t i = 0; i < sizeof good / sizeof *good; i++) {
    cl_json j; cl_json_init(&j, 1);
    CHECK(cl_json_feed(&j, good[i], strlen(good[i])) && cl_json_done(&j), "JSON gate accepts and completes: %s", good[i]);
    cl_json k; cl_json_init(&k, 1);   /* one byte at a time must agree */
    int ok = 1; for (size_t b = 0; good[i][b]; b++) ok &= cl_json_feed(&k, good[i] + b, 1);
    CHECK(ok && cl_json_done(&k), "JSON gate gives the same answer one byte at a time: %s", good[i]);
    cl_json h; cl_json_init(&h, 1);   /* every proper prefix is still a valid beginning and is not yet complete, unless the text ends in whitespace after the object */
    size_t n = strlen(good[i]); int pre = 1;
    for (size_t cut = 1; cut < n; cut++) { cl_json_init(&h, 1); pre &= cl_json_feed(&h, good[i], cut); }
    CHECK(pre, "JSON gate accepts every prefix of: %s", good[i]);
  }
  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
    cl_json j; cl_json_init(&j, 1);
    int ok = cl_json_feed(&j, bad[i], strlen(bad[i]));
    CHECK(!ok || !cl_json_done(&j), "JSON gate refuses or leaves open: %s", bad[i]);
  }
  cl_json a; cl_json_init(&a, 0);
  CHECK(cl_json_feed(&a, "[1, 2]", 6) && cl_json_done(&a), "without object_only, an array is a value too");
  cl_json d; cl_json_init(&d, 1);
  CHECK(cl_json_feed(&d, "{\"k\": 1", 7) && !cl_json_done(&d), "an open object is not done");
  CHECK(cl_json_feed(&d, "}", 1) && cl_json_done(&d) && !cl_json_feed(&d, "{", 1), "after the object, only whitespace may follow");
}

/* The IQ types: every block of a golden file (random blocks and what gguf-py, ggml's reference in Python, dequantizes them to; test/make_iq_golden.py writes it) must dequantize to the
 * same numbers, and the integer dot products must equal "dequantize and multiply". */
static void test_iq(void) {
  FILE *fp = fopen("test/iq_golden.bin", "rb");
  if (!fp) fp = fopen("c/test/iq_golden.bin", "rb");
  if (!fp) { printf("  (no test/iq_golden.bin: IQ types not checked)\n"); return; }
  uint32_t hdr[2];
  while (fread(hdr, 4, 2, fp) == 2) {
    const uint32_t type = hdr[0], n = hdr[1], bl = cl_type_block(type), bb = cl_type_bytes(type);
    CHECK(bl > 0 && bb > 0, "IQ type %u is known", type);
    if (!bl) break;
    uint8_t *blocks = malloc((size_t)n * bb); float *ref = malloc((size_t)n * bl * 4), *got = malloc((size_t)n * bl * 4);
    if (fread(blocks, 1, (size_t)n * bb, fp) != (size_t)n * bb || fread(ref, 4, (size_t)n * bl, fp) != (size_t)n * bl) { CHECK(0, "golden file is complete"); break; }
    cl_dequant_row(type, blocks, got, (int)(n * bl));
    double worst = 0;
    for (uint32_t i = 0; i < n * bl; i++) { double e = fabs((double)got[i] - ref[i]) / (fabs((double)ref[i]) + 1e-3); if (e > worst) worst = e; }
    CHECK(worst < 1e-5, "IQ type %u dequantizes like gguf-py (worst relative difference %.2e)", type, worst);
    if (cl_type_integer_dot(type)) {   /* a row of n blocks against a random activation vector */
      const int cols = (int)(n * bl);
      float *x = malloc((size_t)cols * 4), *xd = malloc((size_t)cols * 4); uint32_t s = 99;
      for (int i = 0; i < cols; i++) { s = s * 1664525u + 1013904223u; x[i] = ((float)(s >> 8) / (1 << 24) - 0.5f) * 3.0f; }
      cl_xq q; cl_xq_init(&q, cols); cl_xq_quantize(&q, x);
      for (int b = 0; b < cols / 256; b++) for (int j = 0; j < 256; j++) xd[b * 256 + j] = q.b[b].d * q.b[b].qs[j];
      float out = 0; cl_mv job = {type, blocks, 1, cols, x, &q, &out, 1, 0, 0};
      cl_matvec_batch(&job, 1);
      double r = 0, mag = 0; for (int i = 0; i < cols; i++) { r += (double)ref[i] * xd[i]; mag += fabs((double)ref[i] * xd[i]); }
      CHECK(fabs(r - out) <= 1e-4 * mag + 1e-5, "IQ type %u integer dot matches dequantize-and-multiply (%.5f vs %.5f)", type, out, r);
      cl_xq_free(&q); free(x); free(xd);
    }
    free(blocks); free(ref); free(got);
  }
  fclose(fp);
}

/* Every instruction set must give the same bits. Random blocks of all five K-quant types are multiplied with a batch of activations (and with a single one) under each kernel
 * set this processor can run; the integer sums are exact in any order, so the floats must be identical, not just close. A batch must also equal the same inputs one at a time. */
static void test_isa_paths(void) {
  enum { ROWS = 37, COLS = 768, NB = 11 };
  static const struct { uint32_t type; int bytes, d_off, dm_off; const char *name; } T[] = {
    {CL_T_Q2_K, 84, 80, 82, "Q2_K"}, {CL_T_Q3_K, 110, 108, -1, "Q3_K"}, {CL_T_Q4_K, 144, 0, 2, "Q4_K"}, {CL_T_Q5_K, 176, 0, 2, "Q5_K"}, {CL_T_Q6_K, 210, 208, -1, "Q6_K"},
    {CL_T_IQ2_XXS, 66, 0, -1, "IQ2_XXS"}, {CL_T_IQ2_XS, 74, 0, -1, "IQ2_XS"}, {CL_T_IQ2_S, 82, 0, -1, "IQ2_S"}, {CL_T_IQ3_XXS, 98, 0, -1, "IQ3_XXS"}, {CL_T_IQ3_S, 110, 0, -1, "IQ3_S"}, {CL_T_IQ1_S, 50, 0, -1, "IQ1_S"}, {CL_T_IQ4_XS, 136, 0, -1, "IQ4_XS"} };
  static float x[NB][COLS], out[3][NB][ROWS], one[ROWS];
  static uint8_t w[ROWS * 3 * 210];
  uint32_t s = 4242;
  for (int b = 0; b < NB; b++) for (int i = 0; i < COLS; i++) { s = s * 1664525u + 1013904223u; x[b][i] = ((float)(s >> 8) / (1 << 24) - 0.5f) * (i % 61 == 0 ? 6.0f : 1.0f); }
  cl_xq q[NB];
  for (int b = 0; b < NB; b++) { CHECK(cl_xq_init(&q[b], COLS), "activation buffer"); cl_xq_quantize(&q[b], x[b]); }
  const char *isas[] = {"scalar", "avx2", "avx512vnni"};
  int ran = 0;
  for (size_t ti = 0; ti < sizeof T / sizeof T[0]; ti++) {
    for (int r = 0; r < ROWS * 3; r++) {
      uint8_t *blk = w + (size_t)r * T[ti].bytes;
      for (int i = 0; i < T[ti].bytes; i++) { s = s * 1664525u + 1013904223u; blk[i] = (uint8_t)(s >> 16); }
      uint16_t d = (uint16_t)(0x2400 + (r % 16) * 8), dm = (uint16_t)(0x2000 + (r % 7) * 8);
      memcpy(blk + T[ti].d_off, &d, 2);
      if (T[ti].dm_off >= 0) memcpy(blk + T[ti].dm_off, &dm, 2);
    }
    int have = 0;
    for (int k = 0; k < 3; k++) {
      if (!cl_quant_set_isa(isas[k]) && k > 0) continue;
      cl_mv job = {T[ti].type, w, ROWS, COLS, x[0], q, out[have ? 1 : 0][0], NB, COLS, ROWS};
      cl_matvec_batch(&job, 1);
      if (k == 0) {   /* the batch against the same inputs one at a time (a different kernel path on most builds) */
        int bad = 0;
        for (int b = 0; b < NB; b++) {
          cl_mv j1 = {T[ti].type, w, ROWS, COLS, x[b], &q[b], one, 1, 0, 0};
          cl_matvec_batch(&j1, 1);
          for (int r = 0; r < ROWS; r++) if (fabsf(one[r] - out[0][b][r]) > 1e-4f * (fabsf(one[r]) + 1.0f)) bad++;
        }
        CHECK(bad == 0, "%s: a batch of %d equals the same inputs one at a time (%d values differ)", T[ti].name, NB, bad);
      } else {
        CHECK(memcmp(out[0], out[1], sizeof out[0]) == 0, "%s: %s gives the same bits as the portable kernel", T[ti].name, isas[k]);
        ran++;
      }
      have = 1;
    }
    cl_quant_set_isa("auto");
  }
  for (int b = 0; b < NB; b++) cl_xq_free(&q[b]);
  printf("  instruction sets compared: %d (this processor: %s)\n", ran, cl_quant_path());
}

/* Q2_K and Q3_K: the dequantizers are checked against a literal transcription of ggml's loops (a different way of indexing the same layout), and the integer dot products against
 * "dequantize and multiply", on random blocks. */
static void ref_q2_K(const uint8_t *b, float *y) {
  uint16_t dh, mh; memcpy(&dh, b + 80, 2); memcpy(&mh, b + 82, 2);
  const float d = cl_f16_to_f32(dh), mn = cl_f16_to_f32(mh);
  const uint8_t *sc = b, *q = b + 16;
  int is = 0;
  for (int n = 0; n < 256; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; j++) {
      uint8_t c = sc[is++];
      float dl = d * (float)(c & 0xF), ml = mn * (float)(c >> 4);
      for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l] >> shift) & 3) - ml;
      c = sc[is++]; dl = d * (float)(c & 0xF); ml = mn * (float)(c >> 4);
      for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l + 16] >> shift) & 3) - ml;
      shift += 2;
    }
    q += 32;
  }
}
static void ref_q3_K(const uint8_t *b, float *y) {
  uint16_t dh; memcpy(&dh, b + 108, 2);
  const float dall = cl_f16_to_f32(dh);
  const uint8_t *hm = b, *q = b + 32;
  uint32_t aux[4]; memcpy(aux, b + 96, 12);
  const uint32_t k1 = 0x03030303u, k2 = 0x0f0f0f0fu, tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4); aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
  aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4); aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
  const int8_t *scales = (const int8_t *)(const void *)aux;
  uint8_t m = 1;
  int is = 0;
  for (int n = 0; n < 256; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; j++) {
      float dl = dall * (float)(scales[is++] - 32);
      for (int l = 0; l < 16; l++) *y++ = dl * (float)((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
      dl = dall * (float)(scales[is++] - 32);
      for (int l = 0; l < 16; l++) *y++ = dl * (float)((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
      shift += 2; m = (uint8_t)(m << 1);
    }
    q += 32;
  }
}
static void test_q2_q3(void) {
  enum { ROWS = 24, NBLK = 3, COLS = NBLK * 256 };
  uint32_t s = 4242;
  static float x[COLS];
  for (int i = 0; i < COLS; i++) { s = s * 1664525u + 1013904223u; x[i] = ((float)(s >> 8) / (1 << 24) - 0.5f) * (i % 53 == 0 ? 6.0f : 1.0f); }
  cl_xq q;
  CHECK(cl_xq_init(&q, COLS), "Q2_K/Q3_K: activation buffer");
  cl_xq_quantize(&q, x);
  float xd[COLS];
  for (int b = 0; b < NBLK; b++) for (int j = 0; j < 256; j++) xd[b * 256 + j] = q.b[b].d * q.b[b].qs[j];
  for (int pass = 0; pass < 2; pass++) {
    const uint32_t type = pass ? CL_T_Q3_K : CL_T_Q2_K;
    const int bs = pass ? 110 : 84;
    CHECK(cl_type_supported(type) && cl_type_block(type) == 256 && cl_type_bytes(type) == (uint32_t)bs && cl_type_integer_dot(type), "%s: the engine knows the type", pass ? "Q3_K" : "Q2_K");
    static uint8_t w[ROWS * NBLK * 110];
    for (int i = 0; i < ROWS * NBLK * bs; i++) { s = s * 1664525u + 1013904223u; w[i] = (uint8_t)(s >> 16); }
    for (int b = 0; b < ROWS * NBLK; b++) {   /* sane half-precision scales */
      uint16_t d = (uint16_t)(0x2400 + (b % 16) * 8), dm = (uint16_t)(0x2000 + (b % 7) * 8);
      if (pass) memcpy(w + (size_t)b * bs + 108, &d, 2); else { memcpy(w + (size_t)b * bs + 80, &d, 2); memcpy(w + (size_t)b * bs + 82, &dm, 2); }
    }
    int badq = 0;
    float a[256], r[256];
    for (int b = 0; b < ROWS * NBLK; b++) {
      cl_dequant_row(type, w + (size_t)b * bs, a, 256);
      if (pass) ref_q3_K(w + (size_t)b * bs, r); else ref_q2_K(w + (size_t)b * bs, r);
      for (int i = 0; i < 256; i++) if (fabsf(a[i] - r[i]) > 1e-6f * (fabsf(r[i]) + 1.0f)) { badq++; break; }
    }
    CHECK(badq == 0, "%s dequantization equals ggml's reference loops on %d of %d blocks", pass ? "Q3_K" : "Q2_K", ROWS * NBLK - badq, ROWS * NBLK);
    static float out[ROWS];
    cl_mv job = {type, w, ROWS, COLS, x, &q, out, 1, 0, 0};
    cl_matvec_batch(&job, 1);
    int bad = 0;
    float row[COLS];
    for (int rw = 0; rw < ROWS; rw++) {
      cl_dequant_row(type, w + (size_t)rw * cl_row_bytes(type, COLS), row, COLS);
      double ref = 0, mag = 0;
      for (int c = 0; c < COLS; c++) { ref += (double)row[c] * xd[c]; mag += fabs((double)row[c] * xd[c]); }
      if (fabs(ref - out[rw]) > 1e-4 * mag + 1e-6) bad++;
    }
    CHECK(bad == 0, "%s integer dot matches dequantize-and-multiply on %d of %d rows", pass ? "Q3_K" : "Q2_K", ROWS - bad, ROWS);
    {   /* a batch of tokens (prompt processing) takes the batched kernel: every token's result must equal its own single-token product */
      enum { NB = 11 };
      static float xb[NB * COLS], ob[NB * ROWS];
      static cl_xq qb[NB];
      for (int b = 0; b < NB; b++) { for (int i = 0; i < COLS; i++) { s = s * 1664525u + 1013904223u; xb[b * COLS + i] = ((float)(s >> 8) / (1 << 24) - 0.5f) * (i % 41 == 0 ? 5.0f : 1.0f); } cl_xq_init(&qb[b], COLS); cl_xq_quantize(&qb[b], xb + b * COLS); }
      cl_mv bj = {type, w, ROWS, COLS, xb, qb, ob, NB, (size_t)COLS, (size_t)ROWS};
      cl_matvec_batch(&bj, 1);
      int badb = 0;
      for (int b = 0; b < NB; b++) {
        static float one[ROWS];
        cl_mv sj = {type, w, ROWS, COLS, xb + b * COLS, &qb[b], one, 1, 0, 0};
        cl_matvec_batch(&sj, 1);
        for (int rw = 0; rw < ROWS; rw++) if (fabsf(one[rw] - ob[b * ROWS + rw]) > 1e-4f * (fabsf(one[rw]) + 1.0f)) badb++;
      }
      CHECK(badb == 0, "%s batched product equals the single-token product on every token and row (%d mismatches)", pass ? "Q3_K" : "Q2_K", badb);
      for (int b = 0; b < NB; b++) cl_xq_free(&qb[b]);
    }
  }
  cl_xq_free(&q);
}

/* Pictures: how big one is made before the encoder sees it, and the resize that does it. */
static void test_pictures(void) {
  int w, h;
  cl_vision_fit(448, 448, 256, &w, &h);
  CHECK(w == 448 && h == 448, "a 448 x 448 picture is left as it is (%d x %d)", w, h);
  cl_vision_fit(4000, 3000, 256, &w, &h);
  CHECK(w % 32 == 0 && h % 32 == 0 && w * h <= 256 * 1024 && w * h > 150 * 1024 && w > h, "a large picture shrinks to at most 256 tokens, keeping its shape (%d x %d)", w, h);
  cl_vision_fit(10, 10, 256, &w, &h);
  CHECK(w == 64 && h == 64, "a tiny picture grows to the minimum (%d x %d)", w, h);
  cl_vision_fit(1000, 100, 256, &w, &h);
  CHECK(w % 32 == 0 && h % 32 == 0 && h >= 64 && w > 8 * h / 2, "a wide strip keeps its shape (%d x %d)", w, h);
  uint8_t flat[8 * 8 * 3];
  memset(flat, 77, sizeof flat);
  uint8_t *r = cl_image_resize(flat, 8, 8, 64, 64);
  int same = r != NULL;
  for (int i = 0; r && i < 64 * 64 * 3; i++) if (r[i] != 77) same = 0;
  CHECK(same, "growing a flat picture keeps it flat");
  free(r);
  static uint8_t big[256 * 256 * 3];
  for (int y = 0; y < 256; y++) for (int x = 0; x < 256; x++) for (int c = 0; c < 3; c++) big[((size_t)y * 256 + x) * 3 + c] = (uint8_t)((x / 128 + y / 128) % 2 ? 200 : 100);   /* four 128 x 128 squares */
  r = cl_image_resize(big, 256, 256, 64, 64);
  CHECK(r && r[(10 * 64 + 10) * 3] == 100 && r[(10 * 64 + 50) * 3] == 200 && r[(50 * 64 + 10) * 3] == 200 && r[(50 * 64 + 50) * 3] == 100, "shrinking keeps the squares' colours where they were");
  free(r);
}

static void test_xs(const char *fixture) {
  if (!fixture) return;
  FILE *fp = fopen(fixture, "rb");
  if (!fp) return;
  static uint8_t file[1 << 20];
  size_t n = fread(file, 1, sizeof file, fp);
  fclose(fp);
  if (n < 200000) return;
  /* two fake layers of 4 "experts"; each expert's gate/up/down are 1000/2000/3000-byte slabs at known places in the file */
  cl_xs_layer lay[2];
  for (int l = 0; l < 2; l++) for (int i = 0; i < 3; i++) { lay[l].slab[i] = 1000u * (uint64_t)(i + 1); lay[l].off[i] = (uint64_t)(l * 60000 + i * 20000); }
  char err[128];
  cl_xs *x = cl_xs_open(fixture, lay, 2, 4, 2, 3 * 6000, 2, CL_LRU, 2, 0, err, sizeof err);
  CHECK(x != NULL, "store opens: %s", err);
  if (!x) return;
  int bad = 0, pass;
  for (pass = 0; pass < 3; pass++) {
    for (int l = 0; l < 2; l++) for (int e = 0; e < 4; e++) {
      int h[2], k;
      h[0] = cl_xs_request(x, l, e); h[1] = cl_xs_request(x, l, (e + 1) % 4);
      for (k = 0; k < 2; k++) {
        int ex = k == 0 ? e : (e + 1) % 4;
        const uint8_t *p = cl_xs_wait(x, h[k]);
        if (!p) { bad++; continue; }
        uint64_t at = 0;
        for (int i = 0; i < 3; i++) { if (memcmp(p + at, file + lay[l].off[i] + (uint64_t)ex * lay[l].slab[i], lay[l].slab[i])) bad++; at += lay[l].slab[i]; }
      }
      cl_xs_release(x, h[0]); cl_xs_release(x, h[1]);
      cl_xs_prefetch(x, l, (e + 2) % 4);
    }
  }
  CHECK(bad == 0, "every expert served from the store equals the bytes in the file (%d wrong)", bad);
  cl_xs_stats st = cl_xs_get_stats(x);
  CHECK(st.accesses == 3 * 2 * 4 * 2 && st.hits > 0 && st.misses > 0 && st.hits + st.misses == st.accesses, "hits and misses add up (%llu + %llu of %llu)", (unsigned long long)st.hits, (unsigned long long)st.misses, (unsigned long long)st.accesses);
  CHECK(cl_xs_request(x, 2, 0) < 0 && cl_xs_request(x, 0, 4) < 0, "out-of-range requests are refused");
  cl_xs_close(x);
}

int main(int argc, char **argv) {
  test_lru(); test_pins(); test_stage(); test_scan_resistance(); test_invariants(); test_route(); test_kernel(); test_quant(); test_integer_dot(); test_q2_q3(); test_isa_paths(); test_backend(); test_json_gate(); test_iq(); test_pictures(); test_xs(argc > 1 ? argv[1] : getenv("CLUTOR_FIXTURE")); test_meta(argc > 1 ? argv[1] : getenv("CLUTOR_FIXTURE"));
  test_gguf(argc > 1 ? argv[1] : getenv("CLUTOR_FIXTURE"));
  printf("%d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
