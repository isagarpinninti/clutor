/* Reads the key/value metadata of a memory-mapped GGUF file: the hyper-parameters and the tokenizer tables the forward pass needs. */
#ifndef CL_META_H
#define CL_META_H
#include <stddef.h>
#include <stdint.h>

typedef struct { const char *p; uint64_t len; } cl_str;   /* a view into the mapped file, not NUL-terminated */
typedef struct { char key[128]; uint32_t type, etype; uint64_t count; size_t off; } cl_kv; /* off: the value, or the first element of an array */
typedef struct { const uint8_t *base; size_t size; cl_kv *kv; int n; } cl_meta;

int cl_meta_parse(cl_meta *m, const uint8_t *base, size_t size, char *err, size_t errlen);
void cl_meta_free(cl_meta *m);
const cl_kv *cl_meta_get(const cl_meta *m, const char *key);
/* Scalars: return 1 and set *out when the key exists and has a numeric (or bool) type. */
int cl_meta_u64(const cl_meta *m, const char *key, uint64_t *out);
int cl_meta_f64(const cl_meta *m, const char *key, double *out);
/* A string value: returns 1 and fills *out. */
int cl_meta_str(const cl_meta *m, const char *key, cl_str *out);
/* Arrays: malloc'd, the caller frees. NULL when absent or of another type. */
cl_str *cl_meta_str_array(const cl_meta *m, const char *key, uint64_t *n);
int32_t *cl_meta_i32_array(const cl_meta *m, const char *key, uint64_t *n);
#endif
