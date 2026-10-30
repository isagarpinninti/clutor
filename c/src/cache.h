/*
 * Expert cache policy: decides WHICH expert occupies WHICH slot. It holds no data and takes no locks; the caller owns the buffers and
 * serialises access. Two policies: plain LRU, and W-TinyLFU (a small admission window, then a segmented LRU whose admission is gated
 * by a 4-bit count-min sketch of how often each expert has been wanted). A separate staging area (the "Buffer B" of the design) holds
 * prefetched experts that are not yet wanted, so a prefetch can never evict something the policy chose to keep.
 */
#ifndef CL_CACHE_H
#define CL_CACHE_H
#include <stdint.h>

typedef enum { CL_LRU = 0, CL_WTINYLFU = 1, CL_ADAPTIVE = 2 } cl_policy;
typedef enum { CL_FREE = 0, CL_WINDOW, CL_PROBATION, CL_PROTECTED, CL_STAGE } cl_list;

typedef struct {
  uint64_t hits, misses, stage_hits, inserts, evictions, admit_rejects, stage_inserts, stage_wasted, aging_resets;
  uint32_t window_slots; /* current admission window size */
} cl_cache_stats;

typedef struct cl_cache cl_cache;

/* `universe` is the number of distinct keys (layers x experts); keys are 0..universe-1. */
/* `window_pct` is the share of the cache used as the admission window (0 = default). */
cl_cache *cl_cache_new(uint32_t cache_slots, uint32_t stage_slots, uint32_t universe, cl_policy policy, uint32_t window_pct);
void cl_cache_free(cl_cache *c);
uint32_t cl_cache_total_slots(const cl_cache *c);
/* Bytes this cache allocates for itself: the struct, one slot record per slot, the key-to-slot map and the Count-Min Sketch. */
uint64_t cl_cache_meta_bytes(const cl_cache *c); /* slots the caller must provide buffers for */

int cl_cache_find(const cl_cache *c, uint32_t key); /* slot holding the key (cached or staged), or -1 */
cl_list cl_cache_list(const cl_cache *c, int slot);
uint32_t cl_cache_slot_key(const cl_cache *c, int slot);

/* A demand access. Records the access in the frequency sketch. Result:
 *  1 hit in the cache (slot returned, recency updated)   2 hit in staging (slot adopted into the cache)
 *  0 miss: a slot was assigned and pinned; the caller must fill it   -1 miss and nothing could be evicted: bypass the cache */
int cl_cache_access(cl_cache *c, uint32_t key, int *slot);
/* A prefetch: assign a staging slot for `key` (pinned until `cl_cache_unpin`). Returns the slot, or -1 if already present / no room. */
int cl_cache_stage(cl_cache *c, uint32_t key);

void cl_cache_pin(cl_cache *c, int slot);
void cl_cache_unpin(cl_cache *c, int slot);
/* Drops a slot that was assigned but whose fill failed. */
void cl_cache_abandon(cl_cache *c, int slot);

uint32_t cl_cache_freq(const cl_cache *c, uint32_t key); /* estimated access count (0..15) */
cl_cache_stats cl_cache_get_stats(const cl_cache *c);
uint32_t cl_cache_count(const cl_cache *c); /* experts in the cache proper (not staging) */
#endif
