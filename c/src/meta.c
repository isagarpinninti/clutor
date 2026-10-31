#include "meta.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const uint8_t *p; size_t n, pos; int bad; } cur;
static int need(cur *c, size_t k) { if (c->bad || c->n - c->pos < k) { c->bad = 1; return 0; } return 1; }
static uint32_t r32(cur *c) { uint32_t v = 0; if (need(c, 4)) { memcpy(&v, c->p + c->pos, 4); c->pos += 4; } return v; }
static uint64_t r64(cur *c) { uint64_t v = 0; if (need(c, 8)) { memcpy(&v, c->p + c->pos, 8); c->pos += 8; } return v; }

static size_t scalar_size(uint32_t t) {
  switch (t) { case 0: case 1: case 7: return 1; case 2: case 3: return 2; case 4: case 5: case 6: return 4; case 10: case 11: case 12: return 8; default: return 0; }
}

static void skip_str(cur *c) { uint64_t len = r64(c); if (!c->bad && need(c, (size_t)len)) c->pos += (size_t)len; }

int cl_meta_parse(cl_meta *m, const uint8_t *base, size_t size, char *err, size_t errlen) {
  memset(m, 0, sizeof *m);
  m->base = base; m->size = size;
  cur c = {base, size, 0, 0};
  if (size < 24 || memcmp(base, "GGUF", 4)) { snprintf(err, errlen, "not a GGUF file"); return 1; }
  c.pos = 4;
  uint32_t ver = r32(&c);
  (void)r64(&c);
  uint64_t nkv = r64(&c);
  if (ver != 2 && ver != 3) { snprintf(err, errlen, "unsupported GGUF version %u", ver); return 1; }
  if (nkv > 100000) { snprintf(err, errlen, "implausible metadata count"); return 1; }
  m->kv = calloc(nkv ? nkv : 1, sizeof(cl_kv));
  if (!m->kv) { snprintf(err, errlen, "out of memory"); return 1; }
  for (uint64_t i = 0; i < nkv; i++) {
    cl_kv *k = &m->kv[m->n];
    uint64_t kl = r64(&c);
    if (c.bad || kl > 1u << 20 || !need(&c, (size_t)kl)) break;
    size_t cp = kl < sizeof k->key - 1 ? (size_t)kl : sizeof k->key - 1;
    memcpy(k->key, c.p + c.pos, cp); k->key[cp] = 0; c.pos += (size_t)kl;
    k->type = r32(&c);
    if (k->type == 9) { k->etype = r32(&c); k->count = r64(&c); }
    k->off = c.pos;
    if (c.bad) break;
    if (k->type == 9) {
      size_t es = scalar_size(k->etype);
      if (es) { if (k->count > (c.n - c.pos) / es) { c.bad = 1; break; } c.pos += (size_t)k->count * es; }
      else if (k->etype == 8) for (uint64_t j = 0; j < k->count && !c.bad; j++) skip_str(&c);
      else { c.bad = 1; break; }
    } else if (k->type == 8) skip_str(&c);
    else { size_t s = scalar_size(k->type); if (!s || !need(&c, s)) { c.bad = 1; break; } c.pos += s; }
    if (c.bad) break;
    m->n++;
  }
  if (c.bad) { cl_meta_free(m); snprintf(err, errlen, "corrupt metadata"); return 1; }
  return 0;
}

void cl_meta_free(cl_meta *m) { free(m->kv); memset(m, 0, sizeof *m); }

const cl_kv *cl_meta_get(const cl_meta *m, const char *key) {
  for (int i = 0; i < m->n; i++) if (!strcmp(m->kv[i].key, key)) return &m->kv[i];
  return NULL;
}

int cl_meta_f64(const cl_meta *m, const char *key, double *out) {
  const cl_kv *k = cl_meta_get(m, key);
  if (!k || k->type == 8 || k->type == 9) return 0;
  const uint8_t *p = m->base + k->off;
  switch (k->type) {
    case 0: *out = p[0]; return 1;
    case 1: *out = (int8_t)p[0]; return 1;
    case 2: { uint16_t v; memcpy(&v, p, 2); *out = v; return 1; }
    case 3: { int16_t v; memcpy(&v, p, 2); *out = v; return 1; }
    case 4: { uint32_t v; memcpy(&v, p, 4); *out = v; return 1; }
    case 5: { int32_t v; memcpy(&v, p, 4); *out = v; return 1; }
    case 6: { float v; memcpy(&v, p, 4); *out = v; return 1; }
    case 7: *out = p[0] != 0; return 1;
    case 10: { uint64_t v; memcpy(&v, p, 8); *out = (double)v; return 1; }
    case 11: { int64_t v; memcpy(&v, p, 8); *out = (double)v; return 1; }
    case 12: { double v; memcpy(&v, p, 8); *out = v; return 1; }
    default: return 0;
  }
}
int cl_meta_u64(const cl_meta *m, const char *key, uint64_t *out) {
  double d;
  if (!cl_meta_f64(m, key, &d) || d < 0) return 0;
  *out = (uint64_t)d;
  return 1;
}

int cl_meta_str(const cl_meta *m, const char *key, cl_str *out) {
  const cl_kv *k = cl_meta_get(m, key);
  if (!k || k->type != 8) return 0;
  uint64_t len; memcpy(&len, m->base + k->off, 8);
  out->p = (const char *)m->base + k->off + 8; out->len = len;
  return 1;
}

cl_str *cl_meta_str_array(const cl_meta *m, const char *key, uint64_t *n) {
  const cl_kv *k = cl_meta_get(m, key);
  if (!k || k->type != 9 || k->etype != 8) return NULL;
  cl_str *a = calloc(k->count ? k->count : 1, sizeof *a);
  if (!a) return NULL;
  size_t pos = k->off;
  for (uint64_t i = 0; i < k->count; i++) {
    uint64_t len; memcpy(&len, m->base + pos, 8);
    a[i].p = (const char *)m->base + pos + 8; a[i].len = len;
    pos += 8 + (size_t)len;
  }
  *n = k->count;
  return a;
}

int32_t *cl_meta_i32_array(const cl_meta *m, const char *key, uint64_t *n) {
  const cl_kv *k = cl_meta_get(m, key);
  if (!k || k->type != 9 || (k->etype != 5 && k->etype != 4)) return NULL;
  int32_t *a = malloc((k->count ? k->count : 1) * sizeof *a);
  if (!a) return NULL;
  memcpy(a, m->base + k->off, (size_t)k->count * 4);
  *n = k->count;
  return a;
}
