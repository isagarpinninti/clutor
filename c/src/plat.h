/*
 * The few operating-system calls the engine needs, behind one interface, so the rest of the code is the same on macOS, Linux, Android (bionic) and Windows (MinGW-w64).
 * POSIX builds use exactly the calls the engine always used; the Windows branch maps them to the Win32 API.
 */
#ifndef CL_PLAT_H
#define CL_PLAT_H
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
static inline size_t cl_page_size(void) { SYSTEM_INFO i; GetSystemInfo(&i); return i.dwPageSize ? i.dwPageSize : 4096; }
static inline int cl_ncpu(void) { SYSTEM_INFO i; GetSystemInfo(&i); return i.dwNumberOfProcessors ? (int)i.dwNumberOfProcessors : 1; }
static inline uint64_t cl_peak_rss(void) { PROCESS_MEMORY_COUNTERS c; return GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof c) ? (uint64_t)c.PeakWorkingSetSize : 0; }
static inline void *cl_aligned_alloc(size_t align, size_t size) { return _aligned_malloc(size, align); }
static inline void cl_aligned_free(void *p) { _aligned_free(p); }
/* True when a read on standard input would not block (bytes waiting, or the other end closed). */
static inline int cl_stdin_ready(void) {
  HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
  DWORD type = GetFileType(h), avail = 0;
  if (type == FILE_TYPE_PIPE) { if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) return 1; return avail > 0; }
  if (type == FILE_TYPE_DISK) return 1;
  return WaitForSingleObject(h, 0) == WAIT_OBJECT_0;
}
static inline void cl_stdio_binary(void) { _setmode(0, _O_BINARY); _setmode(1, _O_BINARY); _setmode(2, _O_BINARY); }

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
#endif
#endif
