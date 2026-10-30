/* GGUF v2/v3 reader: header, metadata we need, tensor table, and the layout the streamer needs (where each routed expert lives). */
#ifndef CL_GGUF_H
#define CL_GGUF_H
#include <stddef.h>
#include <stdint.h>

typedef struct { uint64_t offset, length; } cl_range;

typedef struct {
  char name[160];
  uint32_t ndims;
  uint64_t dims[4];
  uint32_t type;
  uint64_t offset;  /* absolute file offset of the tensor data */
  uint64_t nbytes;  /* 0 when the type or shape is not understood */
} cl_tensor;

typedef struct {
  cl_range *dense;      /* attention, norms, shared experts, dense feed-forward: read for every token */
  int n_dense;
  uint64_t dense_bytes;
  uint64_t router_bytes;
  /* Routed experts are fused: one tensor per gate/up/down holds every expert, experts slowest-varying, so each is one slab. */
  int n_exp;            /* 0 for a dense layer, else 1..3 */
  uint64_t exp_off[3];  /* file offset of expert 0 of each fused tensor */
  uint64_t exp_slab[3]; /* bytes of ONE expert in each fused tensor */
  uint64_t expert_bytes;/* sum of exp_slab */
} cl_layer;

typedef struct {
  char arch[64];
  char name[128];
  uint32_t version;
  uint64_t file_size, data_offset;
  uint32_t n_layers, n_experts, n_used;
  cl_tensor *tensors;
  uint64_t n_tensors;
  cl_layer *layers;
  uint64_t resident_bytes, max_dense_bytes, max_expert_bytes, total_bytes;
  uint32_t moe_layers;
  uint32_t unsized_tensors;
} cl_model;

/* Returns 0 on success. On failure returns nonzero and writes a message to err. */
int cl_model_open(cl_model *m, const char *path, char *err, size_t errlen);
void cl_model_close(cl_model *m);
const char *cl_type_name(uint32_t type);
#endif
