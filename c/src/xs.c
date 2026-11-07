#define _GNU_SOURCE
#include "xs.h"
#include <errno.h>
#include "cl_thread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "plat.h"

enum { S_EMPTY = 0, S_LOADING = 1, S_READY = 2 };
typedef struct job { int slot, layer, expert, prefetch; struct job *next; } job;

struct cl_xs {
  int fd, n_layer, n_expert, n_used, bypass_os;
  const cl_xs_layer *lay;
  cl_xs_layer *lay_copy;
  pthread_mutex_t mu;
  pthread_cond_t cv_work, cv_done;
  job *dq_head, *dq_tail, *pq_head, *pq_tail;
  int shutdown, failed, nth;
  pthread_t th[16];
  cl_cache *cache;
  uint32_t total;          /* cache + staging slots */
  int nbypass, bypass_next;
  uint64_t slot_bytes;
  uint8_t *arena;
  int *state;
  cl_xs_stats st;
};

static uint64_t now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }

static void push_job(cl_xs *x, job *j, int demand) {
  job **h = demand ? &x->dq_head : &x->pq_head, **t = demand ? &x->dq_tail : &x->pq_tail;
  j->next = NULL;
  if (*t) (*t)->next = j; else *h = j;
  *t = j;
  pthread_cond_signal(&x->cv_work);
}
static job *pop_job(cl_xs *x) {
  job **h = x->dq_head ? &x->dq_head : &x->pq_head, **t = x->dq_head ? &x->dq_tail : &x->pq_tail;   /* demand before prefetch */
  job *j = *h;
  if (!j) return NULL;
  *h = j->next;
  if (!*h) *t = NULL;
  return j;
}

static int read_full(cl_xs *x, uint8_t *dst, uint64_t len, uint64_t off) {
  uint64_t done = 0;
  while (done < len) {
    long long n = cl_pread(x->fd, dst + done, (size_t)(len - done), off + done);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return -1;
    done += (uint64_t)n;
  }
  if (x->bypass_os) cl_drop_cache(x->fd, off, len);
  return 0;
}

static void *worker(void *arg) {
  cl_xs *x = arg;
  pthread_mutex_lock(&x->mu);
  for (;;) {
    job *j = pop_job(x);
    if (!j) { if (x->shutdown) break; pthread_cond_wait(&x->cv_work, &x->mu); continue; }
    pthread_mutex_unlock(&x->mu);
    uint64_t t0 = now_ns();
    const cl_xs_layer *L = &x->lay_copy[j->layer];
    uint8_t *dst = x->arena + (size_t)j->slot * x->slot_bytes;
    uint64_t at = 0, bytes = 0;
    int rc = 0;
    for (int i = 0; i < 3 && !rc; i++) { rc = read_full(x, dst + at, L->slab[i], L->off[i] + (uint64_t)j->expert * L->slab[i]); at += L->slab[i]; }
    bytes = at;
    uint64_t dt = now_ns() - t0;
    pthread_mutex_lock(&x->mu);
    x->st.read_ns += dt;
    if (rc) x->failed = 1;
    x->state[j->slot] = rc ? S_EMPTY : S_READY;
    if (j->prefetch) { x->st.prefetch_bytes += bytes; cl_cache_unpin(x->cache, j->slot); } else x->st.demand_bytes += bytes;
    pthread_cond_broadcast(&x->cv_done);
    free(j);
  }
  pthread_mutex_unlock(&x->mu);
  return NULL;
}

cl_xs *cl_xs_open(const char *path, const cl_xs_layer *layers, int n_layer, int n_expert, int n_used, uint64_t cache_bytes, int stage_slots,
                  cl_policy policy, int io_threads, int os_cache_bypass, char *err, size_t errlen) {
  cl_xs *x = calloc(1, sizeof *x);
  if (!x) { snprintf(err, errlen, "out of memory"); return NULL; }
  x->n_layer = n_layer; x->n_expert = n_expert; x->n_used = n_used; x->bypass_os = os_cache_bypass;
  x->lay_copy = malloc((size_t)n_layer * sizeof *layers);
  if (!x->lay_copy) { snprintf(err, errlen, "out of memory"); free(x); return NULL; }
  memcpy(x->lay_copy, layers, (size_t)n_layer * sizeof *layers);
  for (int l = 0; l < n_layer; l++) { uint64_t s = layers[l].slab[0] + layers[l].slab[1] + layers[l].slab[2]; if (s > x->slot_bytes) x->slot_bytes = s; }
  x->fd = cl_open_ro(path);
  if (x->fd < 0) { snprintf(err, errlen, "cannot open %s: %s", path, strerror(errno)); free(x->lay_copy); free(x); return NULL; }
  if (os_cache_bypass) cl_nocache(x->fd);
  uint32_t cslots = (uint32_t)(cache_bytes / x->slot_bytes);
  if (cslots < (uint32_t)n_used) cslots = (uint32_t)n_used;   /* a layer's experts must fit at once */
  uint32_t sslots = stage_slots < 0 ? 0 : (uint32_t)stage_slots;
  x->cache = cl_cache_new(cslots, sslots, (uint32_t)n_layer * (uint32_t)n_expert, policy, 0);
  x->total = x->cache ? cl_cache_total_slots(x->cache) : 0;   /* what the cache really manages (it may round) */
  x->nbypass = n_used + 2;
  size_t nbuf = (size_t)x->total + (size_t)x->nbypass;
  x->arena = cl_aligned_alloc(4096, (nbuf * x->slot_bytes + 4095) / 4096 * 4096);
  x->state = calloc(nbuf, sizeof(int));
  if (!x->cache || !x->arena || !x->state) { snprintf(err, errlen, "out of memory"); cl_xs_close(x); return NULL; }
  pthread_mutex_init(&x->mu, NULL); pthread_cond_init(&x->cv_work, NULL); pthread_cond_init(&x->cv_done, NULL);
  x->nth = io_threads < 1 ? 1 : io_threads > 16 ? 16 : io_threads;
  for (int i = 0; i < x->nth; i++) pthread_create(&x->th[i], NULL, worker, x);
  return x;
}

void cl_xs_close(cl_xs *x) {
  if (!x) return;
  if (x->nth) {
    pthread_mutex_lock(&x->mu);
    x->shutdown = 1;
    pthread_cond_broadcast(&x->cv_work);
    pthread_mutex_unlock(&x->mu);
    for (int i = 0; i < x->nth; i++) pthread_join(x->th[i], NULL);
  }
  if (x->cache) cl_cache_free(x->cache);
  if (x->fd > 0) cl_close(x->fd);
  cl_aligned_free(x->arena); free(x->state); free(x->lay_copy); free(x);
}

uint64_t cl_xs_slot_bytes(const cl_xs *x) { return x->slot_bytes; }
uint32_t cl_xs_cache_slots(const cl_xs *x) { return x->total ? cl_cache_total_slots(x->cache) : 0; }
uint64_t cl_xs_buffer_bytes(const cl_xs *x) { return ((uint64_t)x->total + (uint64_t)x->nbypass) * x->slot_bytes + cl_cache_meta_bytes(x->cache); }

int cl_xs_request(cl_xs *x, int layer, int expert) {
  if (layer < 0 || layer >= x->n_layer || expert < 0 || expert >= x->n_expert) return -1;
  pthread_mutex_lock(&x->mu);
  int slot, r = cl_cache_access(x->cache, (uint32_t)(layer * x->n_expert + expert), &slot);
  x->st.accesses++;
  if (r == 1) x->st.hits++;
  else if (r == 2) { x->st.stage_hits++; x->st.hits++; if (x->state[slot] != S_READY) x->st.prefetch_late++; }
  else {
    x->st.misses++;
    if (r == -1) {   /* nothing could be evicted: read into a spare buffer that the cache does not manage */
      slot = (int)x->total + (x->bypass_next++ % x->nbypass);
      x->st.bypass++;
    }
    x->state[slot] = S_LOADING;
    job *j = calloc(1, sizeof *j);
    j->slot = slot; j->layer = layer; j->expert = expert; j->prefetch = 0;
    push_job(x, j, 1);
  }
  pthread_mutex_unlock(&x->mu);
  return slot;
}

const uint8_t *cl_xs_wait(cl_xs *x, int h) {
  pthread_mutex_lock(&x->mu);
  uint64_t t0 = 0;
  int waited = 0;
  while (x->state[h] != S_READY && !x->failed) {
    if (!waited) { waited = 1; t0 = now_ns(); }
    pthread_cond_wait(&x->cv_done, &x->mu);
  }
  if (waited) x->st.wait_ns += now_ns() - t0;
  int ok = x->state[h] == S_READY;
  pthread_mutex_unlock(&x->mu);
  return ok ? x->arena + (size_t)h * x->slot_bytes : NULL;
}

void cl_xs_release(cl_xs *x, int h) {
  if (h < 0) return;
  pthread_mutex_lock(&x->mu);
  if ((uint32_t)h < x->total) cl_cache_unpin(x->cache, h);
  else x->state[h] = S_EMPTY;
  pthread_mutex_unlock(&x->mu);
}

int cl_xs_resident(cl_xs *x, int layer, int expert) {
  if (layer < 0 || layer >= x->n_layer || expert < 0 || expert >= x->n_expert) return 0;
  pthread_mutex_lock(&x->mu);
  int slot = cl_cache_find(x->cache, (uint32_t)(layer * x->n_expert + expert));
  int r = slot >= 0 && x->state[slot] == S_READY;
  pthread_mutex_unlock(&x->mu);
  return r;
}

void cl_xs_prefetch(cl_xs *x, int layer, int expert) {
  if (layer < 0 || layer >= x->n_layer || expert < 0 || expert >= x->n_expert) return;
  pthread_mutex_lock(&x->mu);
  int slot = cl_cache_stage(x->cache, (uint32_t)(layer * x->n_expert + expert));
  if (slot >= 0) {
    x->state[slot] = S_LOADING;
    x->st.prefetch_issued++;
    job *j = calloc(1, sizeof *j);
    j->slot = slot; j->layer = layer; j->expert = expert; j->prefetch = 1;
    push_job(x, j, 0);
  }
  pthread_mutex_unlock(&x->mu);
}

cl_xs_stats cl_xs_get_stats(cl_xs *x) {
  pthread_mutex_lock(&x->mu);
  cl_xs_stats s = x->st;
  cl_cache_stats cs = cl_cache_get_stats(x->cache);
  s.admit_rejects = cs.admit_rejects; s.evictions = cs.evictions; s.prefetch_wasted = cs.stage_wasted;
  pthread_mutex_unlock(&x->mu);
  return s;
}
void cl_xs_reset_stats(cl_xs *x) { pthread_mutex_lock(&x->mu); memset(&x->st, 0, sizeof x->st); pthread_mutex_unlock(&x->mu); }
