#define _GNU_SOURCE
#include "gguf.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plat.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "GGUF is little-endian; this reader assumes a little-endian host"
#endif

/* [elements per block, bytes per block, name], by ggml type id. Same table as the TypeScript reader. */
static const struct { int id; uint32_t block, bytes; const char *name; } TYPES[] = {
  {0, 1, 4, "F32"}, {1, 1, 2, "F16"}, {2, 32, 18, "Q4_0"}, {3, 32, 20, "Q4_1"}, {6, 32, 22, "Q5_0"}, {7, 32, 24, "Q5_1"}, {8, 32, 34, "Q8_0"}, {9, 32, 36, "Q8_1"},
  {10, 256, 84, "Q2_K"}, {11, 256, 110, "Q3_K"}, {12, 256, 144, "Q4_K"}, {13, 256, 176, "Q5_K"}, {14, 256, 210, "Q6_K"}, {15, 256, 292, "Q8_K"},
  {16, 256, 66, "IQ2_XXS"}, {17, 256, 74, "IQ2_XS"}, {18, 256, 98, "IQ3_XXS"}, {19, 256, 50, "IQ1_S"}, {20, 32, 18, "IQ4_NL"}, {21, 256, 110, "IQ3_S"},
  {22, 256, 82, "IQ2_S"}, {23, 256, 136, "IQ4_XS"}, {24, 1, 1, "I8"}, {25, 1, 2, "I16"}, {26, 1, 4, "I32"}, {27, 1, 8, "I64"}, {28, 1, 8, "F64"},
  {29, 256, 56, "IQ1_M"}, {30, 1, 2, "BF16"}, {34, 256, 54, "TQ1_0"}, {35, 256, 66, "TQ2_0"}, {39, 32, 17, "MXFP4"},
};
#define NTYPES (sizeof TYPES / sizeof TYPES[0])

const char *cl_type_name(uint32_t type) {
  for (size_t i = 0; i < NTYPES; i++) if ((uint32_t)TYPES[i].id == type) return TYPES[i].name;
  return "unknown";
}

typedef struct { const uint8_t *p; size_t n, pos; int over, bad; } cur;

static int need(cur *c, size_t k) {
  if (c->over || c->n - c->pos < k) { c->over = 1; return 0; }
  return 1;
}
static uint32_t rd32(cur *c) { uint32_t v = 0; if (need(c, 4)) { memcpy(&v, c->p + c->pos, 4); c->pos += 4; } return v; }
static uint64_t rd64(cur *c) { uint64_t v = 0; if (need(c, 8)) { memcpy(&v, c->p + c->pos, 8); c->pos += 8; } return v; }

#define MAX_STR (1u << 24)
static void rdstr(cur *c, char *out, size_t cap) {
  uint64_t len = rd64(c);
  if (c->over) return;
  if (len > MAX_STR) { c->bad = 1; return; }
  if (!need(c, (size_t)len)) return;
  if (out && cap) { size_t k = len < cap - 1 ? (size_t)len : cap - 1; memcpy(out, c->p + c->pos, k); out[k] = 0; }
  c->pos += (size_t)len;
}

static size_t scalar_size(uint32_t t) {
  switch (t) { case 0: case 1: case 7: return 1; case 2: case 3: return 2; case 4: case 5: case 6: return 4; case 10: case 11: case 12: return 8; default: return 0; }
}
static int is_int(uint32_t t) { return t <= 5 || t == 10 || t == 11; }

static uint64_t read_int(cur *c, uint32_t t) {
  switch (t) {
    case 0: case 1: { if (!need(c, 1)) return 0; uint8_t v = c->p[c->pos++]; return t == 1 ? (uint64_t)(int64_t)(int8_t)v : v; }
    case 2: case 3: { if (!need(c, 2)) return 0; uint16_t v; memcpy(&v, c->p + c->pos, 2); c->pos += 2; return t == 3 ? (uint64_t)(int64_t)(int16_t)v : v; }
    case 4: return rd32(c);
    case 5: return (uint64_t)(int64_t)(int32_t)rd32(c);
    default: return rd64(c);
  }
}

static void skip_value(cur *c, uint32_t t, int depth) {
  if (c->over || c->bad) return;
  if (depth > 4) { c->bad = 1; return; }
  size_t s = scalar_size(t);
  if (s) { if (need(c, s)) c->pos += s; return; }
  if (t == 8) { rdstr(c, NULL, 0); return; }
  if (t == 9) {
    uint32_t et = rd32(c);
    uint64_t n = rd64(c);
    if (c->over) return;
    if (n > (1ull << 32)) { c->bad = 1; return; }
    size_t es = scalar_size(et);
    if (es) {
      if (n > (c->n - c->pos) / es + 1) { c->over = 1; return; }
      if (!need(c, (size_t)n * es)) return;
      c->pos += (size_t)n * es;
      return;
    }
    for (uint64_t i = 0; i < n && !c->over && !c->bad; i++) skip_value(c, et, depth + 1);
    return;
  }
  c->bad = 1;
}

typedef struct { char key[96]; uint64_t v; } kv_int;
#define MAX_KV_INT 24

static int tensor_nbytes(const cl_tensor *t, uint64_t *out) {
  const uint32_t *bs = NULL;
  uint32_t block = 0, bytes = 0;
  (void)bs;
  for (size_t i = 0; i < NTYPES; i++) if ((uint32_t)TYPES[i].id == t->type) { block = TYPES[i].block; bytes = TYPES[i].bytes; }
  if (!block || t->ndims == 0) return 0;
  if (t->dims[0] % block) return 0;
  unsigned __int128 n = (t->dims[0] / block) * (unsigned __int128)bytes;
  for (uint32_t i = 1; i < t->ndims; i++) { n *= t->dims[i]; if (n > ((unsigned __int128)1 << 62)) return 0; }
  *out = (uint64_t)n;
  return 1;
}

static int starts(const char *s, const char *p) { return strncmp(s, p, strlen(p)) == 0; }
static int ends(const char *s, const char *suf) { size_t a = strlen(s), b = strlen(suf); return a >= b && !strcmp(s + a - b, suf); }

static uint64_t find_int(const kv_int *kv, int n, const char *arch, const char *suffix) {
  char want[160];
  snprintf(want, sizeof want, "%s.%s", arch, suffix);
  for (int i = 0; i < n; i++) if (!strcmp(kv[i].key, want)) return kv[i].v;
  return 0;
}

/* 0 ok, 1 need a larger buffer, -1 bad file (message in err). */
static int parse(cl_model *m, const uint8_t *buf, size_t n, char *err, size_t errlen) {
  cur c = {buf, n, 0, 0, 0};
  if (n >= 4 && memcmp(buf, "GGUF", 4)) { snprintf(err, errlen, "not a GGUF file (bad magic)"); return -1; }
  c.pos = 4;
  m->version = rd32(&c);
  uint64_t nt = rd64(&c), nkv = rd64(&c);
  if (c.over) return 1;
  if (m->version != 2 && m->version != 3) { snprintf(err, errlen, "unsupported GGUF version %u", m->version); return -1; }
  if (nt > 1000000 || nkv > 1000000) { snprintf(err, errlen, "implausible tensor or metadata count"); return -1; }
  kv_int kv[MAX_KV_INT];
  int nk = 0;
  uint64_t alignment = 32;
  m->arch[0] = m->name[0] = 0;
  for (uint64_t i = 0; i < nkv; i++) {
    char key[160];
    rdstr(&c, key, sizeof key);
    uint32_t t = rd32(&c);
    if (c.over) return 1;
    if (c.bad) { snprintf(err, errlen, "corrupt metadata"); return -1; }
    if (!strcmp(key, "general.architecture") && t == 8) rdstr(&c, m->arch, sizeof m->arch);
    else if (!strcmp(key, "general.name") && t == 8) rdstr(&c, m->name, sizeof m->name);
    else if (is_int(t) && (ends(key, ".block_count") || ends(key, ".expert_count") || ends(key, ".expert_used_count") || !strcmp(key, "general.alignment"))) {
      uint64_t v = read_int(&c, t);
      if (!strcmp(key, "general.alignment")) { if (v > 0 && v <= (1u << 20)) alignment = v; }
      else if (nk < MAX_KV_INT) { snprintf(kv[nk].key, sizeof kv[nk].key, "%s", key); kv[nk++].v = v; }
    } else skip_value(&c, t, 0);
    if (c.over) return 1;
    if (c.bad) { snprintf(err, errlen, "corrupt metadata value in \"%s\"", key); return -1; }
  }
  m->n_tensors = nt;
  m->tensors = calloc(nt ? nt : 1, sizeof(cl_tensor));
  if (!m->tensors) { snprintf(err, errlen, "out of memory"); return -1; }
  uint64_t rel[1]; (void)rel;
  uint64_t *rels = calloc(nt ? nt : 1, sizeof(uint64_t));
  if (!rels) { snprintf(err, errlen, "out of memory"); return -1; }
  for (uint64_t i = 0; i < nt; i++) {
    cl_tensor *t = &m->tensors[i];
    rdstr(&c, t->name, sizeof t->name);
    t->ndims = rd32(&c);
    if (c.over) { free(rels); return 1; }
    if (t->ndims == 0 || t->ndims > 4 || c.bad) { free(rels); snprintf(err, errlen, "tensor %llu has %u dimensions", (unsigned long long)i, t->ndims); return -1; }
    for (uint32_t d = 0; d < t->ndims; d++) t->dims[d] = rd64(&c);
    t->type = rd32(&c);
    rels[i] = rd64(&c);
    if (c.over) { free(rels); return 1; }
  }
  m->data_offset = ((c.pos + alignment - 1) / alignment) * alignment;
  for (uint64_t i = 0; i < nt; i++) {
    cl_tensor *t = &m->tensors[i];
    t->offset = m->data_offset + rels[i];
    if (!tensor_nbytes(t, &t->nbytes)) { t->nbytes = 0; m->unsized_tensors++; continue; }
    if (t->offset > m->file_size || t->nbytes > m->file_size - t->offset) { free(rels); snprintf(err, errlen, "tensor \"%s\" points outside the file (truncated download?)", t->name); return -1; }
  }
  free(rels);

  m->n_layers = (uint32_t)find_int(kv, nk, m->arch, "block_count");
  if (m->n_layers == 0 && !strcmp(m->arch, "clip")) m->n_layers = (uint32_t)find_int(kv, nk, m->arch, "vision.block_count");   /* a vision projector keeps its block count under "vision." */
  m->n_experts = (uint32_t)find_int(kv, nk, m->arch, "expert_count");
  m->n_used = (uint32_t)find_int(kv, nk, m->arch, "expert_used_count");
  if (m->n_layers == 0 || m->n_layers > 4096) { snprintf(err, errlen, "the file does not say how many layers it has"); return -1; }
  m->layers = calloc(m->n_layers, sizeof(cl_layer));
  if (!m->layers) { snprintf(err, errlen, "out of memory"); return -1; }
  for (uint64_t i = 0; i < nt; i++) {
    const cl_tensor *t = &m->tensors[i];
    if (!t->nbytes) continue;
    m->total_bytes += t->nbytes;
    unsigned layer = 0;
    int off = 0;
    if (!starts(t->name, "blk.") || sscanf(t->name + 4, "%u.%n", &layer, &off) != 1 || layer >= m->n_layers) { m->resident_bytes += t->nbytes; continue; }
    const char *rest = t->name + 4 + off;
    cl_layer *L = &m->layers[layer];
    if (!strcmp(rest, "ffn_gate_inp.weight")) { L->router_bytes += t->nbytes; m->resident_bytes += t->nbytes; continue; }
    if (starts(rest, "ffn_") && ends(rest, "_exps.weight") && L->n_exp < 3 && t->ndims >= 2) {
      uint64_t ne = t->dims[t->ndims - 1];
      if (ne > 0 && t->nbytes % ne == 0) {
        if (m->n_experts < ne) m->n_experts = (uint32_t)ne;
        L->exp_off[L->n_exp] = t->offset;
        L->exp_slab[L->n_exp] = t->nbytes / ne;
        L->expert_bytes += t->nbytes / ne;
        L->n_exp++;
        continue;
      }
    }
    if (starts(rest, "ffn_") && strstr(rest, "_exps")) { snprintf(err, errlen, "tensor \"%s\": this expert layout is not supported (only fused per-layer expert tensors)", t->name); return -1; }
    L->dense = realloc(L->dense, (size_t)(L->n_dense + 1) * sizeof(cl_range));
    if (!L->dense) { snprintf(err, errlen, "out of memory"); return -1; }
    L->dense[L->n_dense].offset = t->offset;
    L->dense[L->n_dense].length = t->nbytes;
    L->n_dense++;
    L->dense_bytes += t->nbytes;
  }
  for (uint32_t l = 0; l < m->n_layers; l++) {
    if (m->layers[l].n_exp) m->moe_layers++;
    if (m->layers[l].dense_bytes > m->max_dense_bytes) m->max_dense_bytes = m->layers[l].dense_bytes;
    if (m->layers[l].expert_bytes > m->max_expert_bytes) m->max_expert_bytes = m->layers[l].expert_bytes;
  }
  return 0;
}

void cl_model_close(cl_model *m) {
  if (m->layers) for (uint32_t l = 0; l < m->n_layers; l++) free(m->layers[l].dense);
  free(m->layers);
  free(m->tensors);
  memset(m, 0, sizeof *m);
}

int cl_model_open(cl_model *m, const char *path, char *err, size_t errlen) {
  memset(m, 0, sizeof *m);
  int fd = cl_open_ro(path);
  if (fd < 0) { snprintf(err, errlen, "cannot open %s: %s", path, strerror(errno)); return 1; }
  const uint64_t fsize = cl_fd_size(fd);
  if (fsize < 24) { cl_close(fd); snprintf(err, errlen, "%s is too small to be a GGUF file", path); return 1; }
  m->file_size = fsize;
  size_t want = m->file_size < (1u << 20) ? (size_t)m->file_size : (1u << 20);
  for (;;) {
    uint8_t *buf = malloc(want);
    if (!buf) { cl_close(fd); snprintf(err, errlen, "out of memory"); return 1; }
    long long got = cl_pread(fd, buf, want, 0);
    if (got < 0 || (size_t)got != want) { free(buf); cl_close(fd); snprintf(err, errlen, "cannot read %s", path); return 1; }
    cl_model_close(m);
    m->file_size = fsize;
    int r = parse(m, buf, want, err, errlen);
    free(buf);
    if (r == 0) { cl_close(fd); return 0; }
    if (r < 0) { cl_close(fd); cl_model_close(m); return 1; }
    if (want >= m->file_size || want >= ((size_t)1 << 29)) { cl_close(fd); cl_model_close(m); snprintf(err, errlen, "the header is truncated or larger than 512 MiB"); return 1; }
    want = want * 4 > m->file_size ? (size_t)m->file_size : want * 4;
  }
}
