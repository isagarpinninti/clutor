/*
 * The few operating-system calls the engine needs, behind one interface, so the rest of the code is the same on macOS, Linux, Android (bionic) and Windows (MinGW-w64).
 * POSIX builds use exactly the calls the engine always used; the Windows branch maps them to the Win32 API.
 */
#ifndef CL_PLAT_H
#define CL_PLAT_H
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <malloc.h>
#include <psapi.h>
#include <sys/stat.h>

/* Opens a file for reading by its UTF-8 path. Returns a file descriptor or -1. */
static inline int cl_open_ro(const char *path) {
  wchar_t w[4096];
  if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, w, 4096)) return -1;
  return _wopen(w, _O_RDONLY | _O_BINARY);
}
static inline uint64_t cl_fd_size(int fd) {
  struct _stat64 st;
  return _fstat64(fd, &st) == 0 ? (uint64_t)st.st_size : 0;
}
/* Reads up to n bytes at an absolute offset without moving any file position. Returns the bytes read, 0 at the end of the file, -1 on error. */
static inline long long cl_pread(int fd, void *buf, size_t n, uint64_t off) {
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) return -1;
  OVERLAPPED ov;
  memset(&ov, 0, sizeof ov);
  ov.Offset = (DWORD)(off & 0xffffffffu); ov.OffsetHigh = (DWORD)(off >> 32);
  DWORD got = 0, want = n > 0x40000000u ? 0x40000000u : (DWORD)n;
  if (!ReadFile(h, buf, want, &got, &ov)) return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
  return (long long)got;
}
static inline void cl_close(int fd) { _close(fd); }
static inline void cl_nocache(int fd) { (void)fd; }                          /* no portable equivalent without aligned unbuffered reads */
static inline void cl_drop_cache(int fd, uint64_t off, uint64_t len) { (void)fd; (void)off; (void)len; }
static inline void *cl_map_ro(int fd, size_t size) {
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) return NULL;
  HANDLE m = CreateFileMappingW(h, NULL, PAGE_READONLY, (DWORD)((uint64_t)size >> 32), (DWORD)((uint64_t)size & 0xffffffffu), NULL);
  if (!m) return NULL;
  void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(m);   /* the view keeps the mapping alive */
  return p;
}
static inline void cl_unmap(void *p, size_t size) { (void)size; if (p) UnmapViewOfFile(p); }
/* Pins pages in memory. Windows needs the working-set quota raised first. Returns 1 on success. */
static inline int cl_lock_memory(void *p, size_t n) {
  SetProcessWorkingSetSize(GetCurrentProcess(), n + ((size_t)256 << 20), n + ((size_t)512 << 20));
  return VirtualLock(p, n) != 0;
}
static inline uint64_t cl_phys_mem(void) { MEMORYSTATUSEX s; s.dwLength = sizeof s; return GlobalMemoryStatusEx(&s) ? (uint64_t)s.ullTotalPhys : 0; }
/* Memory that can be given to a program now without pushing others out to disk (0 when it cannot be told). */
static inline uint64_t cl_avail_mem(void) { MEMORYSTATUSEX s; s.dwLength = sizeof s; return GlobalMemoryStatusEx(&s) ? (uint64_t)s.ullAvailPhys : 0; }
static inline size_t cl_page_size(void) { SYSTEM_INFO i; GetSystemInfo(&i); return i.dwPageSize ? i.dwPageSize : 4096; }
static inline int cl_ncpu(void) { SYSTEM_INFO i; GetSystemInfo(&i); return i.dwNumberOfProcessors ? (int)i.dwNumberOfProcessors : 1; }
static inline uint64_t cl_peak_rss(void) { PROCESS_MEMORY_COUNTERS c; return GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof c) ? (uint64_t)c.PeakWorkingSetSize : 0; }
static inline void *cl_aligned_alloc(size_t align, size_t size) { return _aligned_malloc(size, align); }
static inline void cl_aligned_free(void *p) { _aligned_free(p); }
static inline void cl_setenv_(const char *k, const char *v) { _putenv_s(k, v); }
static inline void cl_advise_willneed(const void *p, size_t n) { (void)p; (void)n; }   /* no readahead hint on Windows */
/* True when a read on standard input would not block (bytes waiting, or the other end closed). */
static inline int cl_stdin_ready(void) {
  HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
  DWORD type = GetFileType(h), avail = 0;
  if (type == FILE_TYPE_PIPE) { if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) return 1; return avail > 0; }
  if (type == FILE_TYPE_DISK) return 1;
  return WaitForSingleObject(h, 0) == WAIT_OBJECT_0;
}
static inline void cl_stdio_binary(void) { _setmode(0, _O_BINARY); _setmode(1, _O_BINARY); _setmode(2, _O_BINARY); }
/* getline and memmem, which Windows lacks */
static inline long long cl_getline(char **line, size_t *cap, FILE *fp) {
  size_t n = 0;
  int ch;
  if (!*line || !*cap) { *cap = 256; *line = malloc(*cap); if (!*line) return -1; }
  while ((ch = fgetc(fp)) != EOF) {
    if (n + 2 > *cap) { *cap *= 2; char *g = realloc(*line, *cap); if (!g) return -1; *line = g; }
    (*line)[n++] = (char)ch;
    if (ch == '\n') break;
  }
  if (n == 0) return -1;
  (*line)[n] = 0;
  return (long long)n;
}
static inline void *cl_memmem(const void *h, size_t hl, const void *nd, size_t nl) {
  if (nl == 0) return (void *)h;
  for (size_t i = 0; i + nl <= hl; i++) if (!memcmp((const char *)h + i, nd, nl)) return (void *)((const char *)h + i);
  return NULL;
}


/* Windows built with MSVC or clang-cl has no pthreads. The engine uses a small part of them (mutexes, condition variables, creating and joining threads) and a monotonic
 * clock, so that part is provided here on the Win32 API. MinGW-w64 ships its own pthread.h and uses that. */
#if !defined(__MINGW32__)
#include <time.h>
#define CL_WIN_THREADS 1
typedef SRWLOCK pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;
typedef int pthread_mutexattr_t;
typedef int pthread_condattr_t;
typedef int pthread_attr_t;
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT
static inline int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) { (void)a; InitializeSRWLock(m); return 0; }
static inline int pthread_mutex_lock(pthread_mutex_t *m) { AcquireSRWLockExclusive(m); return 0; }
static inline int pthread_mutex_unlock(pthread_mutex_t *m) { ReleaseSRWLockExclusive(m); return 0; }
static inline int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a) { (void)a; InitializeConditionVariable(c); return 0; }
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) { SleepConditionVariableSRW(c, m, INFINITE, 0); return 0; }
static inline int pthread_cond_signal(pthread_cond_t *c) { WakeConditionVariable(c); return 0; }
static inline int pthread_cond_broadcast(pthread_cond_t *c) { WakeAllConditionVariable(c); return 0; }
typedef struct { void *(*fn)(void *); void *arg; } cl_thread_start;
static DWORD WINAPI cl_thread_main(LPVOID p) { cl_thread_start t = *(cl_thread_start *)p; free(p); t.fn(t.arg); return 0; }
static inline int pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
  (void)a;
  cl_thread_start *c = (cl_thread_start *)malloc(sizeof *c);
  if (!c) return 1;
  c->fn = fn; c->arg = arg;
  HANDLE h = CreateThread(NULL, 0, cl_thread_main, c, 0, NULL);
  if (!h) { free(c); return 1; }
  *t = h;
  return 0;
}
static inline int pthread_join(pthread_t t, void **ret) { (void)ret; WaitForSingleObject(t, INFINITE); CloseHandle(t); return 0; }
static inline int pthread_detach(pthread_t t) { CloseHandle(t); return 0; }
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
static inline int clock_gettime(int id, struct timespec *ts) {
  (void)id;
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
  ts->tv_sec = (time_t)(c.QuadPart / f.QuadPart);
  ts->tv_nsec = (long)((c.QuadPart % f.QuadPart) * 1000000000LL / f.QuadPart);
  return 0;
}
#endif

#else   /* ---------------------------------------------------------------- POSIX: macOS, Linux, Android */
#define _GNU_SOURCE
#include <fcntl.h>
#include <poll.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

static inline int cl_open_ro(const char *path) { return open(path, O_RDONLY); }
static inline uint64_t cl_fd_size(int fd) { struct stat st; return fstat(fd, &st) == 0 ? (uint64_t)st.st_size : 0; }
static inline long long cl_pread(int fd, void *buf, size_t n, uint64_t off) { return (long long)pread(fd, buf, n, (off_t)off); }
static inline void cl_close(int fd) { close(fd); }
static inline void cl_nocache(int fd) {
#ifdef F_NOCACHE
  fcntl(fd, F_NOCACHE, 1);   /* macOS */
#else
  (void)fd;
#endif
}
static inline void cl_drop_cache(int fd, uint64_t off, uint64_t len) {
#ifdef POSIX_FADV_DONTNEED
  posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);   /* Linux, Android */
#else
  (void)fd; (void)off; (void)len;
#endif
}
static inline void *cl_map_ro(int fd, size_t size) { void *p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0); return p == MAP_FAILED ? NULL : p; }
static inline void cl_unmap(void *p, size_t size) { if (p) munmap(p, size); }
static inline int cl_lock_memory(void *p, size_t n) { return mlock(p, n) == 0; }
static inline uint64_t cl_phys_mem(void) { long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE); return pages > 0 && psz > 0 ? (uint64_t)pages * (uint64_t)psz : 0; }
/* Memory that can be given to a program now without pushing others out to disk (0 when it cannot be told): free and reclaimable pages, which is what other programs have not taken. */
#if defined(__APPLE__)
#include <mach/mach.h>
static inline uint64_t cl_avail_mem(void) {
  vm_statistics64_data_t v; mach_msg_type_number_t n = HOST_VM_INFO64_COUNT; vm_size_t psz = 0;
  if (host_page_size(mach_host_self(), &psz) != KERN_SUCCESS) return 0;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &n) != KERN_SUCCESS) return 0;
  return ((uint64_t)v.free_count + (uint64_t)v.inactive_count + (uint64_t)v.speculative_count) * (uint64_t)psz;
}
#else
static inline uint64_t cl_avail_mem(void) {
  FILE *f = fopen("/proc/meminfo", "r");
  if (f) { char line[160]; unsigned long long kb = 0; while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { fclose(f); return (uint64_t)kb * 1024u; } fclose(f); }
  long pages = sysconf(_SC_AVPHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE);
  return pages > 0 && psz > 0 ? (uint64_t)pages * (uint64_t)psz : 0;
}
#endif
static inline void cl_setenv_(const char *k, const char *v) { setenv(k, v, 1); }
/* Asks the system to start reading these mapped pages now (they will be needed soon). The range is widened to whole pages. */
static inline void cl_advise_willneed(const void *p, size_t n) { const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE), a = (uintptr_t)p & ~(pg - 1), e = ((uintptr_t)p + n + pg - 1) & ~(pg - 1); madvise((void *)a, (size_t)(e - a), MADV_WILLNEED); }
static inline size_t cl_page_size(void) { long p = sysconf(_SC_PAGESIZE); return p > 0 ? (size_t)p : 4096; }
static inline int cl_ncpu(void) { long c = sysconf(_SC_NPROCESSORS_ONLN); return c > 0 ? (int)c : 1; }
static inline uint64_t cl_peak_rss(void) {
  struct rusage u;
  if (getrusage(RUSAGE_SELF, &u)) return 0;
#if defined(__APPLE__)
  return (uint64_t)u.ru_maxrss;            /* bytes */
#else
  return (uint64_t)u.ru_maxrss * 1024u;    /* kilobytes */
#endif
}
static inline void *cl_aligned_alloc(size_t align, size_t size) { return aligned_alloc(align, size); }
static inline void cl_aligned_free(void *p) { free(p); }
static inline int cl_stdin_ready(void) {
  struct pollfd p = {0, POLLIN, 0};
  int r;
  do r = poll(&p, 1, 0); while (r < 0 && errno == EINTR);
  return r > 0;
}
static inline void cl_stdio_binary(void) {}
static inline long long cl_getline(char **line, size_t *cap, FILE *fp) { return (long long)getline(line, cap, fp); }
static inline void *cl_memmem(const void *h, size_t hl, const void *nd, size_t nl) { return memmem(h, hl, nd, nl); }
#endif
#endif
