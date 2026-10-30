#include "cache.h"
#include <stdlib.h>
#include <string.h>

#ifndef CL_DEFAULT_WINDOW_PCT
#define CL_DEFAULT_WINDOW_PCT 1
#endif
typedef struct { int prev, next; uint32_t key; int list; int pin; } slot_t;
typedef struct { int head, tail; uint32_t n; } dlist_t;

struct cl_cache {
  cl_policy policy;
  uint32_t cache_cap, stage_cap, total, universe, win_cap, main_cap, prot_cap;
  slot_t *s;
  int *where;
  dlist_t lists[5];
  int free_head;
  uint32_t free_n;
  /* 4-bit count-min sketch: 4 rows, packed two counters per byte. */
  uint8_t *cms;
  uint32_t cms_mask;
  uint64_t additions, sample;
  /* adaptive window: compare hit rate across periods and keep moving the window the way that helped */
  uint64_t hc_period, hc_acc, hc_hit;
  double hc_prev;
  int hc_dir;
  cl_cache_stats st;
};

static uint64_t mix(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
static uint32_t cms_idx(const cl_cache *c, uint32_t key, int row) { return (uint32_t)(mix(((uint64_t)key << 3) ^ (uint64_t)(row + 1) * 0x2545f4914f6cdd1dull)) & c->cms_mask; }
static uint32_t cms_get(const cl_cache *c, uint32_t i, int row) { uint8_t b = c->cms[(size_t)row * (c->cms_mask + 1) / 2 + i / 2]; return (i & 1) ? b >> 4 : b & 15; }
static void cms_set(cl_cache *c, uint32_t i, int row, uint32_t v) {
  uint8_t *p = &c->cms[(size_t)row * (c->cms_mask + 1) / 2 + i / 2];
  *p = (i & 1) ? (uint8_t)((*p & 0x0f) | (v << 4)) : (uint8_t)((*p & 0xf0) | v);
}
uint32_t cl_cache_freq(const cl_cache *c, uint32_t key) {
  uint32_t m = 15;
  for (int r = 0; r < 4; r++) { uint32_t v = cms_get(c, cms_idx(c, key, r), r); if (v < m) m = v; }
  return m;
}
static void cms_add(cl_cache *c, uint32_t key) {
  uint32_t idx[4], cur[4], m = 15;
  for (int r = 0; r < 4; r++) { idx[r] = cms_idx(c, key, r); cur[r] = cms_get(c, idx[r], r); if (cur[r] < m) m = cur[r]; }
  if (m < 15) for (int r = 0; r < 4; r++) if (cur[r] == m) cms_set(c, idx[r], r, m + 1); /* conservative update */
  if (++c->additions >= c->sample) { /* aging: halve every counter so old popularity fades */
    size_t bytes = (size_t)4 * (c->cms_mask + 1) / 2;
    for (size_t i = 0; i < bytes; i++) c->cms[i] = (uint8_t)((c->cms[i] >> 1) & 0x77);
    c->additions = 0;
    c->st.aging_resets++;
  }
}

static void unlink_slot(cl_cache *c, int i) {
  slot_t *s = &c->s[i];
  dlist_t *l = &c->lists[s->list];
  if (s->prev >= 0) c->s[s->prev].next = s->next; else l->head = s->next;
  if (s->next >= 0) c->s[s->next].prev = s->prev; else l->tail = s->prev;
  l->n--;
  s->prev = s->next = -1;
}
static void push_front(cl_cache *c, int i, int list) {
  slot_t *s = &c->s[i];
  dlist_t *l = &c->lists[list];
  s->list = list; s->prev = -1; s->next = l->head;
  if (l->head >= 0) c->s[l->head].prev = i; else l->tail = i;
  l->head = i;
  l->n++;
}
static int tail_unpinned(const cl_cache *c, int list, int skip) {
  for (int i = c->lists[list].tail; i >= 0; i = c->s[i].prev) if (!c->s[i].pin && i != skip) return i;
  return -1;
}
static void release(cl_cache *c, int i) {
  slot_t *s = &c->s[i];
  if (s->list != CL_FREE) unlink_slot(c, i);
  if (s->key != UINT32_MAX) c->where[s->key] = -1;
  s->key = UINT32_MAX; s->pin = 0;
  push_front(c, i, CL_FREE);
}

cl_cache *cl_cache_new(uint32_t cache_slots, uint32_t stage_slots, uint32_t universe, cl_policy policy, uint32_t window_pct) {
  if (cache_slots == 0 || universe == 0) return NULL;
  cl_cache *c = calloc(1, sizeof *c);
  if (!c) return NULL;
  c->policy = policy; c->cache_cap = cache_slots; c->stage_cap = stage_slots; c->universe = universe;
  c->total = cache_slots + stage_slots + 1; /* one spare so an insert can precede the eviction it causes */
  if (policy == CL_ADAPTIVE) { c->hc_period = (uint64_t)cache_slots * 8 > 500 ? (uint64_t)cache_slots * 8 : 500; c->hc_dir = 1; c->hc_prev = -1; if (!window_pct) window_pct = 10; }
  if (policy == CL_LRU) { c->win_cap = cache_slots; c->main_cap = 0; }
  else { if (!window_pct || window_pct > 90) window_pct = CL_DEFAULT_WINDOW_PCT;
    c->win_cap = cache_slots * window_pct / 100 ? cache_slots * window_pct / 100 : 1; if (c->win_cap >= cache_slots) c->win_cap = cache_slots > 1 ? cache_slots - 1 : 1; c->main_cap = cache_slots - c->win_cap; }
  c->prot_cap = c->main_cap * 4 / 5;
  c->s = calloc(c->total, sizeof(slot_t));
  c->where = malloc((size_t)universe * sizeof(int));
  uint32_t w = 256;
  while (w < universe * 4) w <<= 1;
  c->cms_mask = w - 1;
  c->cms = calloc((size_t)4 * w / 2, 1);
  c->sample = (uint64_t)cache_slots * 10 > 1000 ? (uint64_t)cache_slots * 10 : 1000;
  if (!c->s || !c->where || !c->cms) { cl_cache_free(c); return NULL; }
  for (uint32_t k = 0; k < universe; k++) c->where[k] = -1;
  for (int l = 0; l < 5; l++) c->lists[l].head = c->lists[l].tail = -1;
  for (uint32_t i = 0; i < c->total; i++) { c->s[i].key = UINT32_MAX; c->s[i].prev = c->s[i].next = -1; push_front(c, (int)i, CL_FREE); }
  return c;
}
void cl_cache_free(cl_cache *c) { if (!c) return; free(c->s); free(c->where); free(c->cms); free(c); }
uint32_t cl_cache_total_slots(const cl_cache *c) { return c->total; }
uint64_t cl_cache_meta_bytes(const cl_cache *c) { return sizeof *c + (uint64_t)c->total * sizeof(slot_t) + (uint64_t)c->universe * sizeof(int) + (uint64_t)4 * (c->cms_mask + 1) / 2; }
int cl_cache_find(const cl_cache *c, uint32_t key) { return key < c->universe ? c->where[key] : -1; }
cl_list cl_cache_list(const cl_cache *c, int slot) { return (cl_list)c->s[slot].list; }
uint32_t cl_cache_slot_key(const cl_cache *c, int slot) { return c->s[slot].key; }
cl_cache_stats cl_cache_get_stats(const cl_cache *c) { cl_cache_stats s = c->st; s.window_slots = c->win_cap; return s; }
uint32_t cl_cache_count(const cl_cache *c) { return c->lists[CL_WINDOW].n + c->lists[CL_PROBATION].n + c->lists[CL_PROTECTED].n; }
void cl_cache_pin(cl_cache *c, int slot) { c->s[slot].pin++; }
void cl_cache_unpin(cl_cache *c, int slot) { if (c->s[slot].pin > 0) c->s[slot].pin--; }
void cl_cache_abandon(cl_cache *c, int slot) { release(c, slot); }

/* Brings the cache back to capacity after an insert. Returns 0 if everything evictable is pinned. */
static int make_room(cl_cache *c, int keep) {
  while (cl_cache_count(c) > c->cache_cap) {
    if (c->policy == CL_LRU) {
      int v = tail_unpinned(c, CL_WINDOW, keep);
      if (v < 0) return 0;
      c->st.evictions++; release(c, v);
      continue;
    }
    if (c->lists[CL_WINDOW].n > c->win_cap) {
      int cand = tail_unpinned(c, CL_WINDOW, keep);
      if (cand >= 0) {
        unlink_slot(c, cand); push_front(c, cand, CL_PROBATION);
        if (c->lists[CL_PROBATION].n + c->lists[CL_PROTECTED].n > c->main_cap) {
          int victim = tail_unpinned(c, CL_PROBATION, -1);
          if (victim < 0) victim = cand;
          int loser = (victim != cand && cl_cache_freq(c, c->s[cand].key) >= cl_cache_freq(c, c->s[victim].key)) ? victim : cand;
          if (loser == cand) c->st.admit_rejects++;
          c->st.evictions++; release(c, loser);
        }
        continue;
      }
    }
    int v = tail_unpinned(c, CL_PROBATION, keep);
    if (v < 0) v = tail_unpinned(c, CL_PROTECTED, keep);
    if (v < 0) v = tail_unpinned(c, CL_WINDOW, keep);
    if (v < 0) return 0;
    c->st.evictions++; release(c, v);
  }
  return 1;
}

static void touch(cl_cache *c, int i) {
  switch (c->s[i].list) {
    case CL_WINDOW: unlink_slot(c, i); push_front(c, i, CL_WINDOW); break;
    case CL_PROTECTED: unlink_slot(c, i); push_front(c, i, CL_PROTECTED); break;
    case CL_PROBATION:
      if (c->policy == CL_LRU) break;
      unlink_slot(c, i); push_front(c, i, CL_PROTECTED);
      while (c->lists[CL_PROTECTED].n > c->prot_cap) {
        int d = tail_unpinned(c, CL_PROTECTED, i);
        if (d < 0) break;
        unlink_slot(c, d); push_front(c, d, CL_PROBATION);
      }
      break;
    default: break;
  }
}

static void set_window(cl_cache *c, uint32_t w) {
  uint32_t lo = 1, hi = c->cache_cap > 1 ? c->cache_cap - 1 : 1;
  if (w < lo) w = lo;
  if (w > hi) w = hi;
  c->win_cap = w; c->main_cap = c->cache_cap - w; c->prot_cap = c->main_cap * 4 / 5;
}
static void hill_climb(cl_cache *c, int hit) {
  c->hc_acc++; c->hc_hit += (uint64_t)hit;
  if (c->hc_acc < c->hc_period) return;
  double rate = (double)c->hc_hit / (double)c->hc_acc;
  if (c->hc_prev >= 0) {
    if (rate < c->hc_prev - 0.002) c->hc_dir = -c->hc_dir; /* it got worse: turn around */
    uint32_t step = c->cache_cap / 20 ? c->cache_cap / 20 : 1;
    set_window(c, c->hc_dir > 0 ? c->win_cap + step : (c->win_cap > step ? c->win_cap - step : 1));
  }
  c->hc_prev = rate; c->hc_acc = c->hc_hit = 0;
}
static int take_free(cl_cache *c) {
  int i = c->lists[CL_FREE].head;
  if (i >= 0) unlink_slot(c, i);
  return i;
}

int cl_cache_access(cl_cache *c, uint32_t key, int *slot) {
  *slot = -1;
  if (key >= c->universe) return -1;
  cms_add(c, key);
  int i = c->where[key];
  if (i >= 0 && c->s[i].list == CL_STAGE) {
    /* A prefetched expert was wanted: it joins the cache. */
    c->st.stage_hits++;
    if (c->policy == CL_ADAPTIVE) hill_climb(c, 1);
    unlink_slot(c, i); push_front(c, i, CL_WINDOW);
    c->s[i].pin++;
    make_room(c, i);
    *slot = i;
    return 2;
  }
  if (i >= 0) { c->st.hits++; if (c->policy == CL_ADAPTIVE) hill_climb(c, 1); touch(c, i); c->s[i].pin++; *slot = i; return 1; }
  c->st.misses++;
  if (c->policy == CL_ADAPTIVE) hill_climb(c, 0);
  int n = take_free(c);
  if (n < 0) return -1;
  c->s[n].key = key; c->s[n].pin = 1; c->where[key] = n;
  push_front(c, n, CL_WINDOW);
  c->st.inserts++;
  if (!make_room(c, n)) { /* everything else is pinned: do not keep the over-capacity entry */ release(c, n); return -1; }
  *slot = n;
  return 0;
}

int cl_cache_stage(cl_cache *c, uint32_t key) {
  if (key >= c->universe || c->where[key] >= 0 || c->stage_cap == 0) return -1;
  if (c->lists[CL_STAGE].n >= c->stage_cap) {
    int old = tail_unpinned(c, CL_STAGE, -1);
    if (old < 0) return -1;
    c->st.stage_wasted++;
    release(c, old);
  }
  int n = take_free(c);
  if (n < 0) return -1;
  c->s[n].key = key; c->s[n].pin = 1; c->where[key] = n;
  push_front(c, n, CL_STAGE);
  c->st.stage_inserts++;
  return n;
}
