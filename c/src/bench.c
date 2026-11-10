/* A disk speed measurement done the way the engine reads: positional reads of the model's own file with the operating system's cache kept out of the way where it can be.
 * The planner uses the result instead of an assumed speed, so "how fast could this go" starts from this computer's disk. */
#define _GNU_SOURCE
#include "plat.h"
#include "chat.h"
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + (double)t.tv_nsec / 1e9; }

/* Reads the file front to back in 4 MiB pieces for `seconds`/2 (starting over if it ends), then 1 MiB pieces at scattered offsets for `seconds`/2, which is how an expert is
 * fetched. Returns 0, with the two rates in bytes per second; on failure 1 and `err`. */
int cl_disk_bench(const char *path, double seconds, double *seq_bps, double *rand_bps, char *err, size_t errlen) {
  int fd = cl_open_ro(path);
  if (fd < 0) { snprintf(err, errlen, "cannot open %s", path); return 1; }
  uint64_t size = cl_fd_size(fd);
  if (size < 8u * 1024 * 1024) { cl_close(fd); snprintf(err, errlen, "the file is too small to time (under 8 MB)"); return 1; }
  cl_nocache(fd);
  const size_t seq_chunk = (size_t)4 << 20, rnd_chunk = (size_t)1 << 20;
  void *buf = cl_aligned_alloc(4096, seq_chunk);
  if (!buf) { cl_close(fd); snprintf(err, errlen, "out of memory"); return 1; }
  if (seconds < 0.5) seconds = 0.5;

  uint64_t total = 0, off = 0;
  double t0 = now_s(), t = t0;
  while ((t = now_s()) - t0 < seconds / 2) {
    long long n = cl_pread(fd, buf, seq_chunk, off);
    if (n <= 0) { off = 0; if (total == 0) { cl_aligned_free(buf); cl_close(fd); snprintf(err, errlen, "could not read the file"); return 1; } continue; }
    total += (uint64_t)n; cl_drop_cache(fd, off, (uint64_t)n); off += (uint64_t)n;
    if (off >= size) off = 0;
  }
  *seq_bps = (double)total / (t - t0 > 1e-6 ? t - t0 : 1e-6);

  uint64_t rtotal = 0, x = 0x9E3779B97F4A7C15ull ^ size;
  uint64_t span = size > rnd_chunk ? size - rnd_chunk : 1;
  t0 = now_s();
  while ((t = now_s()) - t0 < seconds / 2) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    uint64_t o = (x % span) & ~(uint64_t)4095;
    long long n = cl_pread(fd, buf, rnd_chunk, o);
    if (n > 0) { rtotal += (uint64_t)n; cl_drop_cache(fd, o, (uint64_t)n); }
  }
  *rand_bps = (double)rtotal / (t - t0 > 1e-6 ? t - t0 : 1e-6);
  cl_aligned_free(buf); cl_close(fd);
  return 0;
}
