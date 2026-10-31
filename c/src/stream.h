#ifndef CL_STREAM_H
#define CL_STREAM_H
#include "cache.h"
#include "gguf.h"
#include "route.h"
#include <stdint.h>

enum { CL_COMPUTE_KERNEL = 0, CL_COMPUTE_CHECKSUM = 1 };
enum { CL_DENSE_AUTO = 0, CL_DENSE_RESIDENT = 1, CL_DENSE_RING = 2 };

typedef struct {
  int tokens, warmup;
  uint64_t cache_bytes;   /* budget for the expert cache proper */
  int stage_slots;        /* prefetch staging ("Buffer B"), in experts; 0 = twice the experts used per layer */
  cl_policy policy;
  int window_pct;         /* W-TinyLFU admission window as a share of the cache, 0 = default */
  cl_smart smart;
  int dense_mode;         /* CL_DENSE_AUTO, CL_DENSE_RESIDENT (read once, kept in RAM) or CL_DENSE_RING (two layers at a time, re-read every token) */
  int prefetch;           /* 1: predict layer L+1's experts while layer L computes */
  int io_threads;
  int os_cache_bypass;    /* ask the OS not to keep what we read (F_NOCACHE / fadvise), so reads hit the disk */
  uint64_t seed;
  double locality;        /* skew of the synthetic router's expert popularity (Zipf exponent) */
  double predict_corr;    /* how well the synthetic L+1 predictor matches the real L+1 router (0..1) */
  int compute_mode;       /* CL_COMPUTE_KERNEL: a Q4_0 x Q8_0 dot-product kernel over every loaded byte; CL_COMPUTE_CHECKSUM: a cheap hash, so the run is I/O only */
  int compute_us;         /* extra per-layer busy time, added to either mode */
  int shift_at, shift_len; /* scenario: measured tokens [shift_at, shift_at+shift_len) route uniformly at random (scan noise); every other measured token uses one fixed tight topic. shift_len 0 = off */
  const float *trace;     /* a recorded run of the REAL router (from `generate --trace`): trace_tokens x layers x experts raw scores; replayed cyclically. NULL = synthetic routing */
  int trace_tokens;
  int window_tokens;      /* record hits and peak RSS every N measured tokens; 0 = off (10 when a scenario is set) */
} cl_stream_opts;

typedef struct {
  uint64_t accesses, hits, stage_hits, misses, bypass, substituted;
  uint64_t prefetch_issued, prefetch_late, prefetch_wasted;
  uint64_t demand_bytes, prefetch_bytes, dense_bytes;
  uint64_t read_ns_total;        /* sum of time spent inside pread across all I/O threads */
  uint64_t io_wait_ns;           /* time the compute thread sat waiting for bytes */
  uint64_t compute_ns;           /* time spent in the compute step over the loaded bytes */
  uint64_t wall_ns;
  uint64_t admit_rejects, evictions;
  double dropped_mass, max_delta;
  uint64_t checksum;
} cl_stream_totals;

typedef struct {
  uint64_t accesses, hits, misses, substituted, io_wait_ns;
} cl_layer_stats;

typedef struct { int tok_end; uint64_t accesses, hits, peak_rss; } cl_window;

typedef struct {
  cl_stream_totals t;
  cl_window *windows; int n_windows;
  uint64_t rss_start, rss_peak; /* ru_maxrss in bytes: before the run allocates anything, and at the end */
  uint64_t rss_predicted;       /* rss_start + every buffer and table the run allocates, with every page touched: what rss_peak should reach once the cache is full */
  cl_layer_stats *layers; /* model->n_layers entries */
  uint32_t cache_slots, stage_slots;
  int dense_resident;       /* the mode actually used */
  uint64_t dense_preload_bytes; /* read once before the timed run when resident */
  uint64_t slot_bytes;
  int ok;
  char error[256];
} cl_stream_result;

int cl_stream_run(const cl_model *m, const char *path, const cl_stream_opts *o, cl_stream_result *r);
void cl_stream_result_free(cl_stream_result *r);
#endif
