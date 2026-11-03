#define _GNU_SOURCE
#include "stream.h"
#include "kernel.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "plat.h"

static uint64_t phys_ram_bytes(void) { return cl_phys_mem(); }

static uint64_t now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }
static uint64_t mix64(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
static double unit(uint64_t x) { return (double)(mix64(x) >> 11) * (1.0 / 9007199254740992.0); }
static double gauss(uint64_t x) { return (unit(x) + unit(x + 1) + unit(x + 2) + unit(x + 3) - 2.0) * 1.7320508; }

enum { S_EMPTY = 0, S_LOADING = 1, S_READY = 2 };
enum { J_EXPERT = 1, J_DENSE = 2 };
typedef struct job { int kind, slot, layer, expert, ring, prefetch; struct job *next; } job;

typedef struct {
  const cl_model *m;
  int fd;
  int bypass_os_cache;
  pthread_mutex_t mu;
  pthread_cond_t cv_work, cv_done;
  job *dq_head, *dq_tail, *pq_head, *pq_tail; /* demand and prefetch queues */
  int shutdown, failed;
  char error[256];
  cl_cache *cache;
  uint8_t **buf;       /* one buffer per slot, plus bypass buffers after them; all inside `arena` */
  uint8_t *arena;
  int *state;
  int slot_bytes;
  uint8_t *ring[2];
  uint8_t **dense_res; /* per layer, when the dense weights are resident */
  int64_t ring_step[2];
  cl_stream_totals t;
} eng;

static void push_job(eng *e, job *j, int demand) {
  job **h = demand ? &e->dq_head : &e->pq_head, **t = demand ? &e->dq_tail : &e->pq_tail;
  j->next = NULL;
  if (*t) (*t)->next = j; else *h = j;
  *t = j;
  pthread_cond_signal(&e->cv_work);
}
static job *pop_job(eng *e) {
  job **h = e->dq_head ? &e->dq_head : &e->pq_head, **t = e->dq_head ? &e->dq_tail : &e->pq_tail;
  job *j = *h;
  if (!j) return NULL;
  *h = j->next;
  if (!*h) *t = NULL;
  return j;
}

static int read_full(eng *e, uint8_t *dst, uint64_t len, uint64_t off) {
  uint64_t done = 0;
  while (done < len) {
    long long n = cl_pread(e->fd, dst + done, (size_t)(len - done), off + done);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return -1;
    done += (uint64_t)n;
  }
  if (e->bypass_os_cache) cl_drop_cache(e->fd, off, len);
  return 0;
}

static void *worker(void *arg) {
  eng *e = arg;
  pthread_mutex_lock(&e->mu);
  for (;;) {
    job *j = pop_job(e);
    if (!j) { if (e->shutdown) break; pthread_cond_wait(&e->cv_work, &e->mu); continue; }
    pthread_mutex_unlock(&e->mu);
    uint64_t t0 = now_ns();
    int rc = 0;
    uint64_t bytes = 0;
    if (j->kind == J_EXPERT) {
      const cl_layer *L = &e->m->layers[j->layer];
      uint8_t *dst = e->buf[j->slot];
      uint64_t at = 0;
      for (int i = 0; i < L->n_exp && !rc; i++) {
        rc = read_full(e, dst + at, L->exp_slab[i], L->exp_off[i] + (uint64_t)j->expert * L->exp_slab[i]);
        at += L->exp_slab[i];
      }
      bytes = L->expert_bytes;
    } else {
      const cl_layer *L = &e->m->layers[j->layer];
      uint8_t *dst = e->ring[j->ring];
      uint64_t at = 0;
      for (int i = 0; i < L->n_dense && !rc; i++) { rc = read_full(e, dst + at, L->dense[i].length, L->dense[i].offset); at += L->dense[i].length; }
      bytes = L->dense_bytes;
    }
    uint64_t dt = now_ns() - t0;
    pthread_mutex_lock(&e->mu);
    e->t.read_ns_total += dt;
    if (rc) { e->failed = 1; snprintf(e->error, sizeof e->error, "read failed (%s) for layer %d", strerror(errno), j->layer); }
    if (j->kind == J_EXPERT) {
      e->state[j->slot] = rc ? S_EMPTY : S_READY;
      if (j->prefetch) { e->t.prefetch_bytes += bytes; cl_cache_unpin(e->cache, j->slot); } else e->t.demand_bytes += bytes;
    } else { e->t.dense_bytes += bytes; e->ring_step[j->ring] = j->prefetch; /* prefetch field carries the step for dense jobs */ }
    pthread_cond_broadcast(&e->cv_done);
    free(j);
  }
  pthread_mutex_unlock(&e->mu);
  return NULL;
}

/* ---- synthetic router: deterministic, skewed, with topics that drift; labelled SYNTHETIC wherever it is reported ---- */
typedef struct { int L, E, topics; float *base, *bonus; double sigma; uint64_t seed; double rho; } router;
#define TOPIC_LEN 16
static void router_init(router *r, int L, int E, double locality, uint64_t seed, double rho) {
  r->L = L; r->E = E; r->topics = 8; r->sigma = 1.0; r->seed = seed; r->rho = rho;
  r->base = malloc((size_t)L * E * sizeof(float));
  r->bonus = calloc((size_t)r->topics * L * E, sizeof(float));
  int *perm = malloc((size_t)E * sizeof(int));
  for (int l = 0; l < L; l++) {
    for (int i = 0; i < E; i++) perm[i] = i;
    for (int i = E - 1; i > 0; i--) { int j = (int)(mix64(seed ^ ((uint64_t)l << 20) ^ (uint64_t)i) % (uint64_t)(i + 1)); int t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (int rank = 0; rank < E; rank++) r->base[(size_t)l * E + perm[rank]] = (float)(-3.0 * locality * log(rank + 1.0) / log(E));
    for (int tp = 0; tp < r->topics; tp++)
      for (int k = 0; k < 12 && k < E; k++) r->bonus[((size_t)tp * L + l) * E + (int)(mix64(seed ^ 0xabcdef ^ ((uint64_t)tp << 32) ^ ((uint64_t)l << 16) ^ (uint64_t)k) % (uint64_t)E)] += 1.5f;
  }
  free(perm);
}
/* mode 0: topics drift every TOPIC_LEN tokens; 1: one fixed topic (a tight cluster); 2: no structure at all (uniform noise) */
static void router_logits(const router *r, long t, int l, int predicted, int mode, float *out) {
  int tp = mode == 1 ? 0 : (int)((t / TOPIC_LEN) % r->topics);
  const float *b = r->base + (size_t)l * r->E, *bo = r->bonus + ((size_t)tp * r->L + l) * r->E;
  for (int e = 0; e < r->E; e++) {
    uint64_t k = r->seed ^ ((uint64_t)t << 24) ^ ((uint64_t)l << 12) ^ (uint64_t)e * 0x9e3779b1ull;
    double n1 = gauss(k * 4 + 1);
    double n = n1;
    if (predicted) n = mode == 2 ? gauss(k * 4 + 7777) /* scan noise has no structure to predict: the predictor gets its own, independent draw */ : r->rho * n1 + sqrt(fmax(0.0, 1.0 - r->rho * r->rho)) * gauss(k * 4 + 7777);
    out[e] = mode == 2 ? (float)(r->sigma * n) : (float)(b[e] + bo[e] + r->sigma * n);
  }
}

#define KERNEL_COLS 1024 /* activation length: 32 Q8_0 blocks; each 576 bytes of streamed data is one row */
typedef struct { int kernel; cl_block_q8_0 act[KERNEL_COLS / CL_QK]; } compute_ctx;

static void compute_init(compute_ctx *cc, int mode, uint64_t seed) {
  float x[KERNEL_COLS];
  uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
  for (int i = 0; i < KERNEL_COLS; i++) { s = s * 6364136223846793005ull + 1442695040888963407ull; x[i] = (float)((int64_t)(s >> 40) - (1 << 23)) / (float)(1 << 23); }
  cl_quantize_q8_0(x, cc->act, KERNEL_COLS);
  cc->kernel = mode == CL_COMPUTE_KERNEL;
}

static uint64_t touch_bytes(const uint8_t *p, uint64_t n) {
  uint64_t s = 0;
  const uint64_t *w = (const uint64_t *)(const void *)p;
  uint64_t nw = n / 8;
  for (uint64_t i = 0; i < nw; i++) s += w[i] * 0x9e3779b97f4a7c15ull;
  for (uint64_t i = nw * 8; i < n; i++) s += p[i];
  return s;
}

static uint64_t compute_bytes(const compute_ctx *cc, const uint8_t *p, uint64_t n) {
  return cc->kernel ? cl_q4_0_pass(p, n, cc->act, KERNEL_COLS / CL_QK) : touch_bytes(p, n);
}

typedef struct { eng *e; int layer, E; } resctx;
static int resident_cb(void *ctx, int expert) {
  resctx *c = ctx;
  return cl_cache_find(c->e->cache, (uint32_t)(c->layer * c->E + expert)) >= 0;
}

static void wait_slot(eng *e, int slot, uint64_t *wait_ns) {
  uint64_t t0 = now_ns();
  int waited = 0;
  while (e->state[slot] != S_READY && !e->failed) { waited = 1; pthread_cond_wait(&e->cv_done, &e->mu); }
  if (waited) *wait_ns += now_ns() - t0;
}

/* Peak resident set size of this process so far, in bytes (macOS ru_maxrss is bytes; Linux uses VmHWM, falling back to ru_maxrss in kilobytes). */
static uint64_t peak_rss(void) {
#if defined(__linux__)
  /* ru_maxrss survives fork+exec on Linux, so it can still show the parent's peak; VmHWM belongs to this process image only. */
  FILE *f = fopen("/proc/self/status", "r");
  if (f) {
    char line[256]; unsigned long long kb = 0; int found = 0;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "VmHWM: %llu kB", &kb) == 1) { found = 1; break; }
    fclose(f);
    if (found) return (uint64_t)kb * 1024ull;
  }
#endif
  return cl_peak_rss();
}

void cl_stream_result_free(cl_stream_result *r) { free(r->layers); r->layers = NULL; free(r->windows); r->windows = NULL; }

int cl_stream_run(const cl_model *m, const char *path, const cl_stream_opts *o, cl_stream_result *res) {
  memset(res, 0, sizeof *res);
  res->rss_start = peak_rss();
  res->layers = calloc(m->n_layers, sizeof(cl_layer_stats));
  if (m->moe_layers == 0 || m->n_experts == 0 || m->n_used == 0) { snprintf(res->error, sizeof res->error, "this model has no routed experts to stream"); return 1; }
  if (m->n_used > CL_MAX_K || m->n_experts > CL_MAX_E) { snprintf(res->error, sizeof res->error, "unsupported expert counts (%u of %u)", m->n_used, m->n_experts); return 1; }
  uint64_t slot_bytes = m->max_expert_bytes;
  /* Slots sit back to back in one arena at a 4 KiB stride (aligned reads), so the budget is divided by the stride, not by the raw expert size. */
  uint64_t stride = (slot_bytes + 4095) / 4096 * 4096;
  uint32_t cache_slots = (uint32_t)(o->cache_bytes / stride);
  if (cache_slots < 1) cache_slots = 1;
  uint32_t stage_slots = o->stage_slots > 0 ? (uint32_t)o->stage_slots : (o->prefetch ? 2 * m->n_used : 0);
  uint32_t universe = m->n_layers * m->n_experts;
  eng *e = calloc(1, sizeof *e);
  e->m = m; e->bypass_os_cache = o->os_cache_bypass;
  e->fd = cl_open_ro(path);
  if (e->fd < 0) { snprintf(res->error, sizeof res->error, "cannot open %s", path); free(e); return 1; }
  if (o->os_cache_bypass) cl_nocache(e->fd);
  e->cache = cl_cache_new(cache_slots, stage_slots, universe, o->policy, (uint32_t)o->window_pct);
  uint32_t total = cl_cache_total_slots(e->cache), nbuf = total + m->n_used;
  e->buf = calloc(nbuf, sizeof(uint8_t *));
  e->state = calloc(nbuf, sizeof(int));
  { /* the arena is page aligned (16 KiB on Apple silicon) and a multiple of the page size; individual buffers are only 4 KiB aligned */
    long pg = (long)cl_page_size(); if (pg < 4096) pg = 4096;
    uint64_t bytes = ((uint64_t)nbuf * stride + (uint64_t)pg - 1) / (uint64_t)pg * (uint64_t)pg;
    e->arena = cl_aligned_alloc((size_t)pg, bytes);
    if (!e->arena) { snprintf(res->error, sizeof res->error, "out of memory for the expert cache"); return 1; }
    for (uint32_t i = 0; i < nbuf; i++) e->buf[i] = e->arena + (uint64_t)i * stride;
  }
  for (int i = 0; i < 2; i++) { e->ring[i] = cl_aligned_alloc(4096, (m->max_dense_bytes + 4095) / 4096 * 4096 + 4096); e->ring_step[i] = -1; }
  uint64_t dense_total = 0;
  for (uint32_t l = 0; l < m->n_layers; l++) dense_total += m->layers[l].dense_bytes;
  int resident = o->dense_mode == CL_DENSE_RESIDENT;
  if (o->dense_mode == CL_DENSE_AUTO) { uint64_t ram = phys_ram_bytes(); resident = ram > 0 && dense_total <= ram / 4; }
  res->dense_resident = resident;
  if (resident) {
    e->dense_res = calloc(m->n_layers, sizeof(uint8_t *));
    for (uint32_t l = 0; l < m->n_layers; l++) {
      const cl_layer *ly = &m->layers[l];
      if (!ly->n_dense) continue;
      e->dense_res[l] = cl_aligned_alloc(4096, (ly->dense_bytes + 4095) / 4096 * 4096);
      if (!e->dense_res[l]) { snprintf(res->error, sizeof res->error, "out of memory for resident dense weights (%.1f MiB)", dense_total / 1048576.0); return 1; }
      uint64_t at = 0;
      for (int i = 0; i < ly->n_dense; i++) { if (read_full(e, e->dense_res[l] + at, ly->dense[i].length, ly->dense[i].offset)) { snprintf(res->error, sizeof res->error, "read failed for layer %u", l); return 1; } at += ly->dense[i].length; }
    }
    res->dense_preload_bytes = dense_total;
  }
  pthread_mutex_init(&e->mu, NULL); pthread_cond_init(&e->cv_work, NULL); pthread_cond_init(&e->cv_done, NULL);
  int nth = o->io_threads < 1 ? 1 : o->io_threads > 16 ? 16 : o->io_threads;
  pthread_t th[16];
  for (int i = 0; i < nth; i++) pthread_create(&th[i], NULL, worker, e);

  router rt;
  router_init(&rt, (int)m->n_layers, (int)m->n_experts, o->locality, o->seed, o->predict_corr);
  float logits[CL_MAX_E];
  const int K = (int)m->n_used, E = (int)m->n_experts, L = (int)m->n_layers;
  long total_tokens = o->warmup + o->tokens;
  int64_t steps = (int64_t)total_tokens * L, issued = 0;
  cl_cache_stats base_cs = {0};
  cl_stream_totals zero = {0};
  uint64_t t_start = now_ns();
  int counting = o->warmup == 0;
  if (!counting) { /* stats restart after warm-up */ }
  uint64_t sink = 0;
  int win = o->window_tokens > 0 ? o->window_tokens : (o->shift_len > 0 ? 10 : 0);
  int max_win = win > 0 ? (int)(o->tokens / win) + 1 : 0;
  uint64_t win_acc = 0, win_hit = 0;
  res->windows = max_win ? calloc((size_t)max_win, sizeof(cl_window)) : NULL; res->n_windows = 0;
  compute_ctx cc;
  compute_init(&cc, o->compute_mode, o->seed);

  pthread_mutex_lock(&e->mu);
  for (int64_t s = 0; s < steps && !e->failed; s++) {
    long tok = (long)(s / L);
    int l = (int)(s % L);
    if (!counting && tok >= o->warmup) { counting = 1; e->t = zero; base_cs = cl_cache_get_stats(e->cache); t_start = now_ns(); for (int i = 0; i < L; i++) memset(&res->layers[i], 0, sizeof(cl_layer_stats)); sink = 0; }
    const cl_layer *ly = &m->layers[l];
    long mt = tok - o->warmup;
    int rmode = o->shift_len > 0 ? ((mt >= o->shift_at && mt < o->shift_at + o->shift_len) ? 2 : 1) : 0;
    /* dense weights: keep one layer ahead in the 2-slot ring */
    while (!resident && issued <= s + 1 && issued < steps) {
      const cl_layer *dl = &m->layers[issued % L];
      if (dl->n_dense) { job *j = calloc(1, sizeof *j); j->kind = J_DENSE; j->layer = (int)(issued % L); j->ring = (int)(issued & 1); j->prefetch = (int)issued; push_job(e, j, issued == s); }
      else e->ring_step[issued & 1] = issued;
      issued++;
    }
    uint64_t w0 = e->t.io_wait_ns, lw = 0;
    const uint8_t *dbuf = resident ? e->dense_res[l] : e->ring[s & 1];
    if (!resident) { uint64_t t0 = now_ns(); int waited = 0; while (e->ring_step[s & 1] != s && !e->failed) { waited = 1; pthread_cond_wait(&e->cv_done, &e->mu); } if (waited) { lw += now_ns() - t0; } }
    if (ly->n_exp) {
      if (o->trace) memcpy(logits, o->trace + ((size_t)(tok % o->trace_tokens) * L + (size_t)l) * E, (size_t)E * sizeof(float));
      else router_logits(&rt, tok, l, 0, rmode, logits);
      resctx rc = {e, l, E};
      cl_route_result rr;
      cl_route(logits, E, K, &o->smart, resident_cb, &rc, &rr);
      int slots[CL_MAX_K], pinned[CL_MAX_K];
      for (int i = 0; i < K; i++) {
        uint32_t key = (uint32_t)(l * E + rr.selected[i]);
        int slot, r = cl_cache_access(e->cache, key, &slot);
        pinned[i] = r >= 0;
        e->t.accesses++; res->layers[l].accesses++;
        if (r == 1) { e->t.hits++; res->layers[l].hits++; }
        else if (r == 2) { e->t.stage_hits++; e->t.hits++; res->layers[l].hits++; if (e->state[slot] != S_READY) e->t.prefetch_late++; }
        else {
          e->t.misses++; res->layers[l].misses++;
          if (r == -1) { slot = (int)(total + (uint32_t)i); e->t.bypass++; }
          e->state[slot] = S_LOADING;
          job *j = calloc(1, sizeof *j); j->kind = J_EXPERT; j->slot = slot; j->layer = l; j->expert = rr.selected[i]; push_job(e, j, 1);
        }
        slots[i] = slot;
      }
      e->t.substituted += (uint64_t)rr.substituted; res->layers[l].substituted += (uint64_t)rr.substituted;
      e->t.dropped_mass += rr.dropped_mass;
      if (rr.max_delta > e->t.max_delta) e->t.max_delta = rr.max_delta;
      /* while this layer's bytes arrive and are used, start reading the next layer's predicted experts */
      /* With a recorded trace the predictor is causal: it guesses layer l+1 from what that layer chose for the PREVIOUS token (nothing is known for the first). */
      if (o->prefetch && l + 1 < L && m->layers[l + 1].n_exp && stage_slots && (!o->trace || tok > 0)) {
        if (o->trace) memcpy(logits, o->trace + ((size_t)((tok - 1) % o->trace_tokens) * L + (size_t)(l + 1)) * E, (size_t)E * sizeof(float));
        else router_logits(&rt, tok, l + 1, 1, rmode, logits);
        cl_route_result pr;
        cl_route(logits, E, K, NULL, NULL, NULL, &pr);
        for (int i = 0; i < K; i++) {
          uint32_t key = (uint32_t)((l + 1) * E + pr.selected[i]);
          int slot = cl_cache_stage(e->cache, key);
          if (slot < 0) continue;
          e->state[slot] = S_LOADING;
          e->t.prefetch_issued++;
          job *j = calloc(1, sizeof *j); j->kind = J_EXPERT; j->slot = slot; j->layer = l + 1; j->expert = pr.selected[i]; j->prefetch = 1; push_job(e, j, 0);
        }
      }
      for (int i = 0; i < K; i++) wait_slot(e, slots[i], &lw);
      pthread_mutex_unlock(&e->mu);
      uint64_t c0 = now_ns();
      if (dbuf) sink += compute_bytes(&cc, dbuf, ly->dense_bytes);
      for (int i = 0; i < K; i++) sink += compute_bytes(&cc, e->buf[slots[i]], ly->expert_bytes);
      if (o->compute_us > 0) { uint64_t until = now_ns() + (uint64_t)o->compute_us * 1000; while (now_ns() < until) { } }
      uint64_t cdt = now_ns() - c0;
      pthread_mutex_lock(&e->mu);
      e->t.compute_ns += cdt;
      for (int i = 0; i < K; i++) { if (pinned[i]) cl_cache_unpin(e->cache, slots[i]); }
    } else {
      pthread_mutex_unlock(&e->mu);
      uint64_t c0 = now_ns();
      if (dbuf) sink += compute_bytes(&cc, dbuf, ly->dense_bytes);
      uint64_t cdt = now_ns() - c0;
      pthread_mutex_lock(&e->mu);
      e->t.compute_ns += cdt;
    }
    e->t.io_wait_ns += lw;
    res->layers[l].io_wait_ns += lw;
    (void)w0;
    if (win > 0 && counting && l == L - 1 && (tok - o->warmup + 1) % win == 0 && res->n_windows < max_win) {
      cl_window *w = &res->windows[res->n_windows++];
      w->tok_end = (int)(tok - o->warmup + 1); w->accesses = e->t.accesses - win_acc; w->hits = e->t.hits - win_hit; w->peak_rss = peak_rss();
      win_acc = e->t.accesses; win_hit = e->t.hits;
    }
  }
  uint64_t t_end = now_ns();
  e->shutdown = 1;
  pthread_cond_broadcast(&e->cv_work);
  pthread_mutex_unlock(&e->mu);
  for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);

  cl_cache_stats cs = cl_cache_get_stats(e->cache);
  res->t = e->t;
  res->rss_peak = peak_rss();
  {
    uint64_t pg = (uint64_t)cl_page_size(); if (pg < 4096) pg = 4096;
    uint64_t p = res->rss_start + (((uint64_t)nbuf * stride + pg - 1) / pg * pg) + cl_cache_meta_bytes(e->cache);
    p += 2 * (((m->max_dense_bytes + 4095) / 4096 * 4096) + 4096);
    if (resident) for (uint32_t l = 0; l < m->n_layers; l++) p += (m->layers[l].dense_bytes + 4095) / 4096 * 4096;
    p += (uint64_t)m->n_layers * m->n_experts * 4 * (1 + rt.topics) + (uint64_t)nbuf * (sizeof(uint8_t *) + sizeof(int)) + (uint64_t)m->n_layers * sizeof(cl_layer_stats) + (uint64_t)max_win * sizeof(cl_window);
    res->rss_predicted = p;
  }
  res->t.wall_ns = t_end - t_start;
  res->t.admit_rejects = cs.admit_rejects - base_cs.admit_rejects;
  res->t.evictions = cs.evictions - base_cs.evictions;
  res->t.prefetch_wasted = cs.stage_wasted - base_cs.stage_wasted;
  res->t.checksum = sink;
  res->cache_slots = cache_slots; res->stage_slots = stage_slots; res->slot_bytes = slot_bytes;
  res->ok = !e->failed;
  if (e->failed) snprintf(res->error, sizeof res->error, "%s", e->error);
  cl_aligned_free(e->arena);
  if (e->dense_res) { for (uint32_t l = 0; l < m->n_layers; l++) cl_aligned_free(e->dense_res[l]); free(e->dense_res); }
  free(e->buf); free(e->state); cl_aligned_free(e->ring[0]); cl_aligned_free(e->ring[1]); free(rt.base); free(rt.bonus);
  cl_cache_free(e->cache); cl_close(e->fd); free(e);
  return res->ok ? 0 : 1;
}
