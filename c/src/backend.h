#ifndef CL_BACKEND_H
#define CL_BACKEND_H
#include "quant.h"

/* A backend is where the matrix products run. The engine's every matrix-vector product goes through cl_matvec_batch (quant.h), which hands the products
 * to the active backend; products the backend does not handle (supports() returns 0 for their weight type) stay on the CPU, so a backend need not cover
 * every type. The CPU kernels are the default backend, named "cpu". Choose another with CLUTOR_BACKEND=<name> (read once, at the first product) or
 * cl_backend_select. No backend other than "cpu" is built into this engine; this is the seam where one would go (docs/DIRECTIONS.md). */
typedef struct cl_backend {
  const char *name;
  int (*supports)(uint32_t type);                       /* 1 when matvec_batch can compute weights of this type */
  void (*matvec_batch)(const cl_mv *jobs, int n);       /* run every job; the result must equal the CPU's within the tolerance the backend documents */
} cl_backend;

const cl_backend *cl_backend_cpu(void);
int cl_backend_register(const cl_backend *b);           /* 1 on success; at most 8 backends; the pointer must stay valid */
int cl_backend_select(const char *name);                /* 1 when a backend of that name exists and is now active; 0 leaves the active one unchanged */
const cl_backend *cl_backend_active(void);
#endif
