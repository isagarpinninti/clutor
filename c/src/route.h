/*
 * Expert routing with optional cache awareness (SMART-ROUTE).
 *
 * Plain routing takes the K experts with the highest router probability. SMART-ROUTE keeps the true top-J, and for each remaining
 * slot that would need an expert that is NOT in memory, substitutes an expert that IS in memory when it ranks within the top-M and
 * its router probability is within `tau` of the one it replaces. With tau = infinity this is the unbounded "cache-aware routing"
 * heuristic; with a small tau a substitution can only ever move a bounded amount of probability, which is the point.
 */
#ifndef CL_ROUTE_H
#define CL_ROUTE_H
#define CL_MAX_K 32
#define CL_MAX_E 1024

typedef struct { int enabled; int J, M; double tau; } cl_smart;

typedef struct {
  int selected[CL_MAX_K];
  int k;
  int substituted;      /* slots filled with a different expert than the router's own choice */
  double dropped_mass;  /* sum over substitutions of (p of the expert replaced - p of the one used) */
  double max_delta;     /* largest single such difference; never exceeds tau */
} cl_route_result;

typedef int (*cl_resident_fn)(void *ctx, int expert);
/* Returns 0, or -1 if the arguments are out of range (E > CL_MAX_E, K > CL_MAX_K, K > E). */
int cl_route(const float *logits, int E, int K, const cl_smart *smart, cl_resident_fn resident, void *ctx, cl_route_result *out);
#endif
