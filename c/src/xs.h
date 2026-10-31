/*
 * Expert store: serves the experts of a real forward pass from disk through Clutor's cache, so resident memory is bounded by the cache
 * budget and not by the model's size. A slot holds one expert's gate, up and down matrices back to back. Misses are read by a pool of I/O
 * threads (pread, with the OS cache bypassed when asked); a prefetch fills a staging slot that the cache policy cannot be evicted by.
 */
#ifndef CL_XS_H
#define CL_XS_H
#include "cache.h"
#include <stddef.h>
#include <stdint.h>

typedef struct { uint64_t off[3], slab[3]; } cl_xs_layer;   /* file offset of expert 0 and bytes per expert, for gate, up, down */

typedef struct {
  uint64_t accesses, hits, stage_hits, misses, bypass, prefetch_issued, prefetch_late, prefetch_wasted, admit_rejects, evictions;
  uint64_t demand_bytes, prefetch_bytes, read_ns, wait_ns;
} cl_xs_stats;

typedef struct cl_xs cl_xs;

cl_xs *cl_xs_open(const char *path, const cl_xs_layer *layers, int n_layer, int n_expert, int n_used, uint64_t cache_bytes, int stage_slots,
                  cl_policy policy, int io_threads, int os_cache_bypass, char *err, size_t errlen);
void cl_xs_close(cl_xs *x);
uint64_t cl_xs_slot_bytes(const cl_xs *x);
uint32_t cl_xs_cache_slots(const cl_xs *x);
uint64_t cl_xs_buffer_bytes(const cl_xs *x);   /* memory the store holds for expert data and its tables */

/* Starts getting an expert. Returns a handle (>= 0) or -1 on failure. Misses are read in the background. */
int cl_xs_request(cl_xs *x, int layer, int expert);
/* Blocks until the expert is in memory; returns its bytes (gate, up, down in that order) or NULL if the read failed. */
const uint8_t *cl_xs_wait(cl_xs *x, int handle);
void cl_xs_release(cl_xs *x, int handle);
/* A hint: read this expert into staging if there is room and it is not already here. */
void cl_xs_prefetch(cl_xs *x, int layer, int expert);
cl_xs_stats cl_xs_get_stats(cl_xs *x);
void cl_xs_reset_stats(cl_xs *x);
#endif
