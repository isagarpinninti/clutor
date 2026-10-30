#include "route.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int e; double p; } ranked;
static int by_p_desc(const void *a, const void *b) {
  const ranked *x = a, *y = b;
  if (x->p != y->p) return x->p < y->p ? 1 : -1;
  return x->e - y->e; /* ties: lower expert id first, so routing is deterministic */
}

int cl_route(const float *logits, int E, int K, const cl_smart *smart, cl_resident_fn resident, void *ctx, cl_route_result *out) {
  if (E <= 0 || E > CL_MAX_E || K <= 0 || K > CL_MAX_K || K > E) return -1;
  ranked r[CL_MAX_E];
  double mx = logits[0], sum = 0;
  for (int i = 1; i < E; i++) if (logits[i] > mx) mx = logits[i];
  for (int i = 0; i < E; i++) { r[i].e = i; r[i].p = exp((double)logits[i] - mx); sum += r[i].p; }
  for (int i = 0; i < E; i++) r[i].p /= sum;
  qsort(r, (size_t)E, sizeof r[0], by_p_desc);
  memset(out, 0, sizeof *out);
  out->k = K;
  for (int i = 0; i < K; i++) out->selected[i] = r[i].e;
  if (!smart || !smart->enabled || !resident) return 0;

  int J = smart->J < 0 ? 0 : smart->J > K ? K : smart->J;
  int M = smart->M < K ? K : smart->M > E ? E : smart->M;
  char taken[CL_MAX_E] = {0};
  int cold[CL_MAX_K], ncold = 0;
  for (int i = 0; i < K; i++) {
    if (i < J || resident(ctx, r[i].e)) taken[r[i].e] = 1;
    else cold[ncold++] = i; /* rank of a true-top expert that would have to be read from disk */
  }
  for (int c = 0; c < ncold; c++) {
    int rank = cold[c], pick = -1;
    for (int j = 0; j < M; j++) { /* best-ranked resident expert not already used, within tau of the one it replaces */
      if (taken[r[j].e] || j == rank) continue;
      if (!resident(ctx, r[j].e)) continue;
      if (r[rank].p - r[j].p > smart->tau) continue;
      pick = j;
      break;
    }
    if (pick < 0) { taken[r[rank].e] = 1; continue; }
    double d = r[rank].p - r[pick].p;
    taken[r[pick].e] = 1;
    out->substituted++;
    out->dropped_mass += d;
    if (d > out->max_delta) out->max_delta = d;
    out->selected[rank] = r[pick].e;
  }
  return 0;
}
