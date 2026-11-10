/* Threads: pthreads everywhere, except Windows built with MSVC or clang-cl, where plat.h provides the few calls the engine uses. */
#ifndef CL_THREAD_H
#define CL_THREAD_H
#if defined(_WIN32) && !defined(__MINGW32__)
#include "plat.h"
#else
#include <pthread.h>
#endif
#endif
