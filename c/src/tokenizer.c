#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- string -> int hash table (open addressing; keys are views into the mapped file or into our own buffer) ---- */
typedef struct { const char *k; uint32_t len; int32_t v; } slot;
typedef struct { slot *s; size_t cap; } smap;

static uint64_t hash(const char *p, size_t n) { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; i++) { h ^= (uint8_t)p[i]; h *= 1099511628211ull; } return h; }
static int smap_init(smap *m, size_t n) { m->cap = 1; while (m->cap < n * 2 + 2) m->cap <<= 1; m->s = calloc(m->cap, sizeof(slot)); return m->s != NULL; }
static void smap_put(smap *m, const char *k, size_t len, int32_t v) {
  size_t i = hash(k, len) & (m->cap - 1);
  while (m->s[i].k) { if (m->s[i].len == len && !memcmp(m->s[i].k, k, len)) return; i = (i + 1) & (m->cap - 1); }   /* first definition wins, as in llama.cpp */
  m->s[i].k = k; m->s[i].len = (uint32_t)len; m->s[i].v = v;
}
static int32_t smap_get(const smap *m, const char *k, size_t len) {
  size_t i = hash(k, len) & (m->cap - 1);
  while (m->s[i].k) { if (m->s[i].len == len && !memcmp(m->s[i].k, k, len)) return m->s[i].v; i = (i + 1) & (m->cap - 1); }
  return -1;
}

struct cl_tok {
  cl_str *tokens; int32_t *types; uint64_t n;
  smap vocab, merges;
  char *merge_buf;
  int bos, eos, add_bos;
  int *special; int n_special;       /* ids of control / user-defined tokens, longest text first */
  int byte_cp[256];                  /* GPT-2 byte -> code point */
  int cp_byte[512];                  /* code point (< 512) -> byte, or -1 */
};

static int put_utf8(char *o, int cp) {
  if (cp < 0x80) { o[0] = (char)cp; return 1; }
  if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 63)); return 2; }
  o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 63)); o[2] = (char)(0x80 | (cp & 63)); return 3;
}
/* Decodes one code point; an invalid byte gives U+FFFD and length 1. */
static int get_utf8(const uint8_t *p, size_t avail, int *cp) {
  uint8_t c = p[0];
  if (c < 0x80) { *cp = c; return 1; }
  int n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
  if (!n || (size_t)n > avail) { *cp = 0xFFFD; return 1; }
  int v = c & (0x3F >> (n - 1));
  for (int i = 1; i < n; i++) { if ((p[i] & 0xC0) != 0x80) { *cp = 0xFFFD; return 1; } v = (v << 6) | (p[i] & 63); }
  *cp = v;
  return n;
}

/* ---- character classes: exact for ASCII, approximate above it ---- */
static int is_space(int c) { return c == ' ' || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000 || c == 0x1680; }
static int is_digit(int c) { return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) || (c >= 0x660 && c <= 0x669) || (c >= 0xFF10 && c <= 0xFF19); }
static int is_punct(int c) {   /* Unicode P*: ASCII punctuation except the symbols $ + < = > ^ ` | ~ */
  if (c < 0x80) return c && strchr("!\"#%&'()*,-./:;?@[\\]_{}", c) != NULL;
  return c == 0xA1 || c == 0xA7 || c == 0xAB || c == 0xB6 || c == 0xB7 || c == 0xBB || c == 0xBF || (c >= 0x2010 && c <= 0x2027) || (c >= 0x2030 && c <= 0x205E) || (c >= 0x3001 && c <= 0x3003) || (c >= 0x3008 && c <= 0x3011) || (c >= 0xFF01 && c <= 0xFF0F && c != 0xFF04 && c != 0xFF0B);
}
static int is_letter(int c) {
  if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  if (c == 0xAA || c == 0xB5 || c == 0xBA) return 1;
  if (c < 0xC0 || c == 0xD7 || c == 0xF7) return 0;   /* Latin-1 symbols */
  if (is_space(c) || is_digit(c) || is_punct(c)) return 0;
  if ((c >= 0x2000 && c <= 0x2BFF) || (c >= 0x3000 && c <= 0x303F) || (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0xFF00 && c <= 0xFF0F)) return 0;   /* punctuation, symbols, emoji */
  return 1;
}

/* ---- the three splitting patterns of the "olmo" pre-tokenizer; each returns the length of the match at s[i] (0 = no match) ---- */
static int cp_at(const uint8_t *s, size_t i, size_t end, int *cp) { return get_utf8(s + i, end - i, cp); }

static size_t run(const uint8_t *s, size_t i, size_t end, int (*in)(int), int negate_all_three) {
  size_t j = i;
  while (j < end) {
    int cp, n = cp_at(s, j, end, &cp);
    int ok = negate_all_three ? !(is_space(cp) || is_letter(cp) || is_digit(cp)) : in(cp);
    if (!ok) break;
    j += (size_t)n;
  }
  return j - i;
}
static size_t m_gpt2(const uint8_t *s, size_t i, size_t end) {   /* 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S) */
  static const char *contr[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  for (size_t c = 0; c < sizeof contr / sizeof *contr; c++) { size_t n = strlen(contr[c]); if (end - i >= n && !memcmp(s + i, contr[c], n)) return n; }
  size_t j = i;
  if (s[j] == ' ') j++;
  if (j < end) {
    size_t n;
    if ((n = run(s, j, end, is_letter, 0))) return j - i + n;
    if ((n = run(s, j, end, is_digit, 0))) return j - i + n;
    if ((n = run(s, j, end, NULL, 1))) return j - i + n;
  }
  /* \s+(?!\S): the longest whitespace run that is followed by whitespace or the end; a run followed by text gives back its last character */
  size_t k = i, last = i;
  while (k < end) { int cp, n = cp_at(s, k, end, &cp); if (!is_space(cp)) break; last = k; k += (size_t)n; }
  if (k == i) return 0;
  if (k == end) return k - i;
  return last > i ? last - i : 0;
}

typedef size_t (*matcher)(const uint8_t *, size_t, size_t);
typedef struct { size_t a, b; } seg;
typedef struct { seg *v; size_t n, cap; } segs;
static int segs_add(segs *l, size_t a, size_t b) {
  if (a >= b) return 1;
  if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2 : 64; l->v = realloc(l->v, l->cap * sizeof(seg)); if (!l->v) return 0; }
  l->v[l->n++] = (seg){a, b};
  return 1;
}
/* Applies one pattern to every segment: matches become their own segments, and what lies between matches stays a segment for the next pattern. */
static int split(const uint8_t *s, const segs *in, segs *out, matcher m) {
  for (size_t k = 0; k < in->n; k++) {
    size_t a = in->v[k].a, end = in->v[k].b, gap = a, i = a;
    while (i < end) {
      size_t n = m(s, i, end);
      if (n) { if (!segs_add(out, gap, i) || !segs_add(out, i, i + n)) return 0; i += n; gap = i; }
      else { int cp; i += (size_t)cp_at(s, i, end, &cp); }
    }
    if (!segs_add(out, gap, end)) return 0;
  }
  return 1;
}

cl_tok *cl_tok_load(const cl_meta *m, char *err, size_t errlen) {
  cl_str mdl, pre;
  if (!cl_meta_str(m, "tokenizer.ggml.model", &mdl) || mdl.len != 4 || memcmp(mdl.p, "gpt2", 4)) { snprintf(err, errlen, "only the gpt2 (byte-level BPE) tokenizer is supported"); return NULL; }
  if (!cl_meta_str(m, "tokenizer.ggml.pre", &pre) || pre.len != 4 || memcmp(pre.p, "olmo", 4)) { snprintf(err, errlen, "only the \"olmo\" pre-tokenizer is implemented"); return NULL; }
  cl_tok *t = calloc(1, sizeof *t);
  if (!t) { snprintf(err, errlen, "out of memory"); return NULL; }
  t->tokens = cl_meta_str_array(m, "tokenizer.ggml.tokens", &t->n);
  uint64_t nt = 0, nm = 0;
  t->types = cl_meta_i32_array(m, "tokenizer.ggml.token_type", &nt);
  cl_str *merges = cl_meta_str_array(m, "tokenizer.ggml.merges", &nm);
  if (!t->tokens || !merges) { snprintf(err, errlen, "the model has no tokenizer tables"); free(merges); cl_tok_free(t); return NULL; }
  if (!smap_init(&t->vocab, t->n) || !smap_init(&t->merges, nm)) { snprintf(err, errlen, "out of memory"); free(merges); cl_tok_free(t); return NULL; }
  for (uint64_t i = 0; i < t->n; i++) smap_put(&t->vocab, t->tokens[i].p, t->tokens[i].len, (int32_t)i);
  for (uint64_t i = 0; i < nm; i++) smap_put(&t->merges, merges[i].p, merges[i].len, (int32_t)i);   /* "left right": the rank is the line number */
  free(merges);
  uint64_t v;
  t->bos = cl_meta_u64(m, "tokenizer.ggml.bos_token_id", &v) ? (int)v : -1;
  t->eos = cl_meta_u64(m, "tokenizer.ggml.eos_token_id", &v) ? (int)v : -1;
  t->add_bos = cl_meta_u64(m, "tokenizer.ggml.add_bos_token", &v) ? (int)v : 0;
  int n = 0;
  for (int i = 0; i < 512; i++) t->cp_byte[i] = -1;
  for (int b = 0; b < 256; b++) {
    int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    t->byte_cp[b] = printable ? b : 256 + n++;
    t->cp_byte[t->byte_cp[b]] = b;
  }
  t->special = malloc((t->n + 1) * sizeof(int));
  if (t->types && t->special) for (uint64_t i = 0; i < t->n && i < nt; i++) if ((t->types[i] == 3 || t->types[i] == 4) && t->tokens[i].len > 0) t->special[t->n_special++] = (int)i;
  for (int a = 0; a < t->n_special; a++) for (int b = a + 1; b < t->n_special; b++)   /* longest first */
    if (t->tokens[t->special[b]].len > t->tokens[t->special[a]].len) { int x = t->special[a]; t->special[a] = t->special[b]; t->special[b] = x; }
  return t;
}

void cl_tok_free(cl_tok *t) {
  if (!t) return;
  free(t->tokens); free(t->types); free(t->vocab.s); free(t->merges.s); free(t->special);
  free(t);
}
int cl_tok_vocab_size(const cl_tok *t) { return (int)t->n; }
int cl_tok_bos(const cl_tok *t) { return t->bos; }
int cl_tok_eos(const cl_tok *t) { return t->eos; }
int cl_tok_add_bos(const cl_tok *t) { return t->add_bos; }

/* Byte-pair-encodes one pre-token and appends its ids. */
static int bpe(const cl_tok *t, const uint8_t *s, size_t len, int32_t *out, int cap, int *n) {
  if (len == 0) return 1;
  char *buf = malloc(len * 3 + 1);
  size_t *st = malloc((len + 1) * sizeof(size_t));   /* start offset of each symbol in buf, plus the end */
  if (!buf || !st) { free(buf); free(st); return 0; }
  size_t pos = 0, ns = 0;
  for (size_t i = 0; i < len; i++) { st[ns++] = pos; pos += (size_t)put_utf8(buf + pos, t->byte_cp[s[i]]); }
  st[ns] = pos;
  for (;;) {   /* merge the adjacent pair with the lowest rank until none is in the merge table */
    int32_t best = -1; size_t bi = 0;
    for (size_t i = 0; i + 1 < ns; i++) {
      size_t la = st[i + 1] - st[i], lb = st[i + 2] - st[i + 1];
      char key[256];
      if (la + lb + 1 > sizeof key) continue;
      memcpy(key, buf + st[i], la); key[la] = ' '; memcpy(key + la + 1, buf + st[i + 1], lb);
      int32_t r = smap_get(&t->merges, key, la + 1 + lb);
      if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
    }
    if (best < 0) break;
    memmove(&st[bi + 1], &st[bi + 2], (ns - bi - 1) * sizeof(size_t));   /* drop the boundary between the pair */
    ns--;
  }
  for (size_t i = 0; i < ns; i++) {
    int32_t id = smap_get(&t->vocab, buf + st[i], st[i + 1] - st[i]);
    if (id < 0) { free(buf); free(st); return 0; }
    if (*n >= cap) { free(buf); free(st); return -1; }
    out[(*n)++] = id;
  }
  free(buf); free(st);
  return 1;
}

static int encode_plain(const cl_tok *t, const uint8_t *s, size_t len, int32_t *out, int cap, int *n) {
  segs a = {0}, b = {0}, c = {0}, d = {0};
  int rc = 1;
  if (!segs_add(&a, 0, len) || !split(s, &a, &d, m_gpt2)) rc = 0;
  for (size_t i = 0; rc == 1 && i < d.n; i++) rc = bpe(t, s + d.v[i].a, d.v[i].b - d.v[i].a, out, cap, n);
  free(a.v); free(b.v); free(c.v); free(d.v);
  return rc;
}

int cl_tok_encode(const cl_tok *t, const char *text, size_t len, int parse_special, int32_t *out, int cap) {
  const uint8_t *s = (const uint8_t *)text;
  int n = 0;
  size_t from = 0, i = 0;
  while (parse_special && i < len) {
    int hit = -1;
    for (int k = 0; k < t->n_special && hit < 0; k++) {
      const cl_str *tk = &t->tokens[t->special[k]];
      if (len - i >= tk->len && !memcmp(s + i, tk->p, tk->len)) hit = t->special[k];
    }
    if (hit < 0) { i++; continue; }
    int rc = encode_plain(t, s + from, i - from, out, cap, &n);
    if (rc != 1) return -1;
    if (n >= cap) return -1;
    out[n++] = hit;
    i += t->tokens[hit].len; from = i;
  }
  return encode_plain(t, s + from, len - from, out, cap, &n) == 1 ? n : -1;
}

int cl_tok_piece(const cl_tok *t, int id, char *out, int cap) {
  if (id < 0 || (uint64_t)id >= t->n) return 0;
  const cl_str *tk = &t->tokens[id];
  int ctl = t->types && (t->types[id] == 3 || t->types[id] == 4);
  int w = 0;
  for (size_t i = 0; i < tk->len && w < cap;) {
    if (ctl) { out[w++] = tk->p[i++]; continue; }
    int cp, n = get_utf8((const uint8_t *)tk->p + i, tk->len - i, &cp);
    i += (size_t)n;
    out[w++] = (char)(cp < 512 && t->cp_byte[cp] >= 0 ? t->cp_byte[cp] : '?');
  }
  return w;
}
