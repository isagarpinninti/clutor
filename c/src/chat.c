#define _GNU_SOURCE
#include "chat.h"
#include "plat.h"
#include "forward.h"
#include "tokenizer.h"
#include "quant.h"
#include "vision.h"
#include "jsongate.h"
#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <time.h>

/*
 * A conversation with a model as a library: the same operations the `chat` command serves over stdin and stdout, behind a handle and a callback, so a program
 * (the graphical app, on every operating system, Android included) can link the engine instead of starting it as a separate process. Each operation runs to
 * completion on the calling thread and reports through the emit callback, one JSON object per call; cl_chat_cancel may be called from any thread.
 */

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + (double)t.tv_nsec / 1e9; }

/* ---- a growing text buffer for the events ---- */
typedef struct { char *p; size_t n, cap; } ob_t;
static void ob_reserve(ob_t *o, size_t more) {
  if (o->n + more + 1 <= o->cap) return;
  size_t cap = o->cap ? o->cap : 512;
  while (cap < o->n + more + 1) cap *= 2;
  o->p = realloc(o->p, cap); o->cap = cap;
}
static void ob_put(ob_t *o, const char *s, size_t len) { ob_reserve(o, len); memcpy(o->p + o->n, s, len); o->n += len; o->p[o->n] = 0; }
static void ob_fmt(ob_t *o, const char *fmt, ...) {
  va_list ap, ap2;
  va_start(ap, fmt); va_copy(ap2, ap);
  int need = vsnprintf(NULL, 0, fmt, ap); va_end(ap);
  if (need > 0) { ob_reserve(o, (size_t)need); vsnprintf(o->p + o->n, (size_t)need + 1, fmt, ap2); o->n += (size_t)need; }
  va_end(ap2);
}
/* A JSON string from bytes that may not be valid UTF-8 (a byte-level token can be half a character): bad bytes become U+FFFD, so the output always parses. */
static void ob_json(ob_t *o, const char *p, int n) {
  ob_put(o, "\"", 1);
  for (int i = 0; i < n;) {
    unsigned char c = (unsigned char)p[i];
    if (c == '"' || c == '\\') { char b[2] = {'\\', (char)c}; ob_put(o, b, 2); i++; }
    else if (c < 0x20) { ob_fmt(o, "\\u%04x", c); i++; }
    else if (c < 0x80) { ob_put(o, p + i, 1); i++; }
    else {
      int len = c >= 0xF0 && c < 0xF8 ? 4 : c >= 0xE0 ? 3 : c >= 0xC2 && c < 0xE0 ? 2 : 0, ok = len > 0 && i + len <= n && c < 0xF8;
      for (int k = 1; ok && k < len; k++) if (((unsigned char)p[i + k] & 0xC0) != 0x80) ok = 0;
      if (ok) { ob_put(o, p + i, (size_t)len); i += len; }
      else { ob_put(o, "\xef\xbf\xbd", 3); i++; }
    }
  }
  ob_put(o, "\"", 1);
}

/* Finds the value of a top-level key in one JSON object line. Walks the object, so a key name inside a string value is never mistaken for a key. */
static const char *jfind(const char *j, const char *key) {
  size_t kl = strlen(key);
  while (*j && *j != '{') j++;
  if (!*j) return NULL;
  j++;
  for (;;) {
    while (*j == ' ' || *j == ',' || *j == '\t') j++;
    if (*j != '"') return NULL;
    const char *ks = ++j;
    while (*j && *j != '"') j += *j == '\\' && j[1] ? 2 : 1;
    if (!*j) return NULL;
    size_t n = (size_t)(j - ks);
    j++;
    while (*j == ' ') j++;
    if (*j != ':') return NULL;
    j++;
    while (*j == ' ') j++;
    if (n == kl && !strncmp(ks, key, kl)) return j;
    if (*j == '"') { j++; while (*j && *j != '"') j += *j == '\\' && j[1] ? 2 : 1; if (!*j) return NULL; j++; }
    else { int depth = 0; while (*j && (depth > 0 || (*j != ',' && *j != '}'))) { if (*j == '{' || *j == '[') depth++; if (*j == '}' || *j == ']') depth--; j++; } }
  }
}
static double jnum(const char *j, const char *key, double d) { const char *v = jfind(j, key); return v && (*v == '-' || (*v >= '0' && *v <= '9')) ? atof(v) : d; }

static int hexv(char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }
static int put_utf8(char *o, unsigned cp) {
  if (cp < 0x80) { o[0] = (char)cp; return 1; }
  if (cp < 0x800) { o[0] = (char)(0xC0 | cp >> 6); o[1] = (char)(0x80 | (cp & 63)); return 2; }
  if (cp < 0x10000) { o[0] = (char)(0xE0 | cp >> 12); o[1] = (char)(0x80 | ((cp >> 6) & 63)); o[2] = (char)(0x80 | (cp & 63)); return 3; }
  o[0] = (char)(0xF0 | cp >> 18); o[1] = (char)(0x80 | ((cp >> 12) & 63)); o[2] = (char)(0x80 | ((cp >> 6) & 63)); o[3] = (char)(0x80 | (cp & 63)); return 4;
}
/* A JSON string value decoded to UTF-8 (malloc'd), or NULL when the key is missing or not a string. */
static char *jstr(const char *j, const char *key, size_t *outlen) {
  const char *v = jfind(j, key);
  if (!v || *v != '"') return NULL;
  v++;
  size_t cap = strlen(v) + 1, n = 0;
  char *o = malloc(cap);
  while (*v && *v != '"') {
    if (*v != '\\') { o[n++] = *v++; continue; }
    v++;
    switch (*v) {
      case 'n': o[n++] = '\n'; v++; break; case 't': o[n++] = '\t'; v++; break; case 'r': o[n++] = '\r'; v++; break;
      case 'b': o[n++] = '\b'; v++; break; case 'f': o[n++] = '\f'; v++; break;
      case 'u': {
        unsigned cp = 0; int ok = 1;
        for (int i = 1; i <= 4; i++) { int h = hexv(v[i]); if (h < 0) ok = 0; else cp = cp << 4 | (unsigned)h; }
        if (!ok) { o[n++] = '?'; v++; break; }
        v += 5;
        if (cp >= 0xD800 && cp < 0xDC00 && v[0] == '\\' && v[1] == 'u') {
          unsigned lo = 0; int ok2 = 1;
          for (int i = 2; i <= 5; i++) { int h = hexv(v[i]); if (h < 0) ok2 = 0; else lo = lo << 4 | (unsigned)h; }
          if (ok2 && lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); v += 6; }
        }
        n += (size_t)put_utf8(o + n, cp);
        break;
      }
      case 0: break;
      default: o[n++] = *v++; break;
    }
  }
  o[n] = 0;
  if (outlen) *outlen = n;
  return o;
}

static uint64_t rng_s = 88172645463325252ull;
static double rnd(void) { rng_s ^= rng_s >> 12; rng_s ^= rng_s << 25; rng_s ^= rng_s >> 27; return (double)((rng_s * 2685821657736338717ull) >> 11) / 9007199254740992.0; }

/* Constrained decoding: `ok` says whether a token may follow (the JSON checker below). It is asked only about tokens that would enter the short list, so a few hundred
 * calls at most per token, never one per vocabulary entry. */
typedef int (*tok_ok_fn)(void *ctx, int tok);

/* temp <= 0: the most likely token. Otherwise: repetition penalty over `recent`, the top_k best, temperature, min_p and top_p, then a draw. Returns -1 when `ok` refuses every token. */
static int sample_token(float *logits, int n, double temp, int top_k, double top_p, double min_p, double rep, const int32_t *recent, int nrecent, tok_ok_fn ok, void *okctx) {
  if (rep > 1.0) for (int i = 0; i < nrecent; i++) { float *l = &logits[recent[i]]; *l = *l > 0 ? (float)(*l / rep) : (float)(*l * rep); }
  if (temp <= 0) {
    int b = -1;
    for (int i = 0; i < n; i++) if ((b < 0 || logits[i] > logits[b]) && (!ok || ok(okctx, i))) b = i;
    return b;
  }
  if (top_k < 1) top_k = 1;
  if (top_k > 200) top_k = 200;
  int idx[200]; float val[200]; int k = 0;
  for (int i = 0; i < n; i++) {
    float v = logits[i];
    if (k == top_k && v <= val[k - 1]) continue;
    if (ok && !ok(okctx, i)) continue;
    int p = k < top_k ? k++ : k - 1;
    while (p > 0 && val[p - 1] < v) { val[p] = val[p - 1]; idx[p] = idx[p - 1]; p--; }
    val[p] = v; idx[p] = i;
  }
  if (k == 0) return -1;
  double mx = val[0], sum = 0, pr[200];
  for (int i = 0; i < k; i++) { pr[i] = exp((val[i] - mx) / temp); sum += pr[i]; }
  double cum = 0; int keep = k;
  for (int i = 0; i < k; i++) { cum += pr[i] / sum; if (cum >= top_p) { keep = i + 1; break; } }
  if (min_p > 0) for (int i = 1; i < keep; i++) if (pr[i] < min_p * pr[0]) { keep = i; break; }   /* min-p: drop tokens far less likely than the best one */
  double tot = 0;
  for (int i = 0; i < keep; i++) tot += pr[i];
  double r = rnd() * tot;
  for (int i = 0; i < keep; i++) { r -= pr[i]; if (r <= 0) return idx[i]; }
  return idx[keep - 1];
}

/* Prompt lookup: guess the next tokens from the text itself. If the last `ngram` tokens (the history and `last`) occurred before, the tokens that followed them then are the guess.
 * Code and edited text repeat themselves a lot, so the guess is often right, and a right guess is verified in one pass over the weights instead of one pass per token. */
static int lookup_draft(const int32_t *h, int n, int last, int32_t *out, int kmax, int ngram) {
  const int m = n + 1;
  if (m < ngram + 1) return 0;
  #define SEQ(i) ((i) == n ? last : h[i])
  for (int i = m - ngram - 1; i >= 0; i--) {
    int ok = 1;
    for (int j = 0; j < ngram && ok; j++) ok = SEQ(i + j) == SEQ(m - ngram + j);
    if (!ok) continue;
    int k = 0;
    for (int p = i + ngram; p < m && k < kmax; p++) out[k++] = SEQ(p);
    return k;
  }
  #undef SEQ
  return 0;
}

typedef struct { const cl_tok *tok; cl_json st; } json_ctx;
/* A token is allowed when its text keeps the reply a valid beginning of a JSON object. Control tokens (end of reply included) are not: the reply ends when the object does. */
static int json_tok_ok(void *vc, int tok) {
  json_ctx *c = vc;
  if (cl_tok_is_eog(c->tok, tok)) return 0;
  char piece[256];
  int pl = cl_tok_piece(c->tok, tok, piece, sizeof piece);
  if (pl <= 0) return 0;
  cl_json t = c->st;
  return cl_json_feed(&t, piece, (size_t)pl);
}

/* How many bytes of p[0..n) end on a character boundary: an incomplete UTF-8 sequence at the end is held back until its remaining bytes arrive. */
static int utf8_complete(const unsigned char *p, int n) {
  for (int back = 1; back <= 3 && back <= n; back++) {
    unsigned char c = p[n - back];
    if ((c & 0xC0) == 0x80) continue;
    int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    return need > back ? n - back : n;
  }
  return n;
}


struct cl_chat {
  cl_fwd f;
  cl_meta meta; uint8_t *tokmap; size_t tokmap_size; cl_tok *tok;
  cl_vision *vis; int pad_id;   /* the Qwen3-VL picture encoder (when a projector file was given) and the id of the <|image_pad|> token */
  int ctx, nhist, have_logits;
  float *logits; int32_t *hist;
  cl_chat_emit emit; void *user;
  cl_chat_poll poll; void *poll_user;
  atomic_int cancel;
  ob_t out;
};

static void flush_event(cl_chat *c) { if (c->out.n) c->emit(c->user, c->out.p); c->out.n = 0; if (c->out.p) c->out.p[0] = 0; }
static void ev_error(cl_chat *c, const char *msg) { ob_put(&c->out, "{\"ev\":\"error\",\"msg\":", 20); ob_json(&c->out, msg, (int)strlen(msg)); ob_put(&c->out, "}", 1); flush_event(c); }
static int cancelled(cl_chat *c) { return atomic_load(&c->cancel) || (c->poll && c->poll(c->poll_user)); }

static const char *arg_of(int argc, const char *const *argv, const char *name) { const char *r = NULL; for (int i = 0; i < argc - 1; i++) if (!strcmp(argv[i], name)) r = argv[i + 1]; return r; }
static int flag_of(int argc, const char *const *argv, const char *name) { for (int i = 0; i < argc; i++) if (!strcmp(argv[i], name)) return 1; return 0; }
static double num_of(int argc, const char *const *argv, const char *name, double d) { const char *v = arg_of(argc, argv, name); return v ? atof(v) : d; }

cl_fwd_opts cl_fwd_opts_from_args(int argc, const char *const *argv, char *err, size_t errlen) {
  cl_fwd_opts o = {0};
  o.cache_bytes = (uint64_t)(num_of(argc, argv, "--cache-mb", 0) * 1048576.0);
  const char *pf = arg_of(argc, argv, "--prefetch");
  o.prefetch = pf ? (!strcmp(pf, "on") || !strcmp(pf, "prev") ? 1 : !strcmp(pf, "ahead") ? 2 : 0) : 0;   /* off by default: previous-token prefetch lost on every disk-bound measurement (docs/NATIVE_ENGINE.md) */
  const char *pol = arg_of(argc, argv, "--policy");
  o.policy = !pol || !strcmp(pol, "adaptive") ? CL_ADAPTIVE : !strcmp(pol, "lru") ? CL_LRU : CL_WTINYLFU;
  o.io_threads = (int)num_of(argc, argv, "--io-threads", 3);
  o.stage_slots = (int)num_of(argc, argv, "--stage", 0);
  o.prefetch_n = (int)num_of(argc, argv, "--prefetch-n", 0);
  o.os_bypass = flag_of(argc, argv, "--no-os-cache");
  o.lock = flag_of(argc, argv, "--lock");
  const char *sm = arg_of(argc, argv, "--smart");
  if (sm && strcmp(sm, "off")) {
    int J, M; double tau;
    if (sscanf(sm, "%d,%d,%lf", &J, &M, &tau) != 3 || J < 0 || M < 1 || tau < 0) { snprintf(err, errlen, "--smart wants J,M,TAU (for example 4,16,0.02) or off"); o.policy = (cl_policy)-1; return o; }
    o.smart.enabled = 1; o.smart.J = J; o.smart.M = M; o.smart.tau = tau;
  }
  return o;
}

/* Reads the metadata at the front of the file (the first 64 MiB at most) and builds the tokenizer from it. */
int cl_load_tokenizer(const char *path, cl_meta *meta, uint8_t **map, size_t *msize, cl_tok **tok, char *err, size_t errlen) {
  int fd = cl_open_ro(path);
  if (fd < 0) { snprintf(err, errlen, "cannot open %s", path); return 1; }
  uint64_t sz = cl_fd_size(fd);
  size_t want = sz < ((uint64_t)64 << 20) ? (size_t)sz : ((size_t)64 << 20);
  *map = malloc(want ? want : 1); *msize = want;
  size_t got = 0;
  while (*map && got < want) { long long r = cl_pread(fd, *map + got, want - got, got); if (r <= 0) break; got += (size_t)r; }
  cl_close(fd);
  if (!*map || got != want) { snprintf(err, errlen, "cannot read %s", path); return 1; }
  if (cl_meta_parse(meta, *map, want, err, errlen)) return 1;
  *tok = cl_tok_load(meta, err, errlen);
  return *tok ? 0 : 1;
}

cl_chat *cl_chat_open(int argc, const char *const *argv, cl_chat_emit emit, void *user, char *err, size_t errlen) {
  if (argc < 1 || !emit) { snprintf(err, errlen, "a model path and an emit callback are required"); return NULL; }
  cl_chat *c = calloc(1, sizeof *c);
  if (!c) { snprintf(err, errlen, "out of memory"); return NULL; }
  c->emit = emit; c->user = user;
  if (cl_load_tokenizer(argv[0], &c->meta, &c->tokmap, &c->tokmap_size, &c->tok, err, errlen)) { free(c); return NULL; }
  int ctx = 4096;
  for (int i = 0; i < argc - 1; i++) if (!strcmp(argv[i], "--ctx")) ctx = atoi(argv[i + 1]);
  if (ctx < 64) ctx = 64;
  cl_fwd_opts fo = cl_fwd_opts_from_args(argc, argv, err, errlen);
  if ((int)fo.policy == -1) { free(c); return NULL; }
  { const char *kv = arg_of(argc, argv, "--kv"); if (kv && !strcmp(kv, "q8")) cl_setenv_("CLUTOR_KV", "q8"); }
  if (cl_fwd_open(&c->f, argv[0], ctx, &fo, err, errlen)) { cl_tok_free(c->tok); free(c->tokmap); free(c); return NULL; }
  if (c->f.n_ctx_max > 0 && ctx > c->f.n_ctx_max) ctx = c->f.n_ctx_max;   /* the model's own limit; the caller reads the real size from the ready event */
  c->ctx = ctx;
  c->pad_id = -1;
  { const char *mm = arg_of(argc, argv, "--mmproj");
    if (mm && c->f.n_deepstack) {   /* a projector given for a model that does not take pictures is ignored, so an app can pass one whenever it finds one next to the model */
      c->vis = cl_vision_open(mm, c->f.n_embd, err, errlen);
      if (!c->vis) { cl_fwd_close(&c->f); cl_tok_free(c->tok); free(c->tokmap); free(c); return NULL; }
      int32_t pid[4];
      if (cl_tok_encode(c->tok, "<|image_pad|>", 13, 1, pid, 4) == 1) c->pad_id = pid[0];
    } }
  c->logits = malloc((size_t)c->f.n_vocab * sizeof *c->logits);
  c->hist = malloc((size_t)ctx * sizeof *c->hist);   /* every token in the KV cache, for the repetition penalty */
  if (!c->logits || !c->hist) { snprintf(err, errlen, "out of memory"); cl_chat_close(c); return NULL; }
  cl_str tpl = {0};
  cl_meta_str(&c->meta, "tokenizer.chat_template", &tpl);
  ob_fmt(&c->out, "{\"ev\":\"ready\",\"arch\":\"%s\",\"name\":\"%s\",\"nCtx\":%d,\"nVocab\":%d,\"nLayer\":%d,\"threads\":%d,\"streaming\":%s,\"autoStream\":%s,\"locked\":%s,\"addBos\":%s,\"nCtxTrain\":%d,\"kvBytesPerToken\":%zu,\"vision\":%s,\"pinnedLayers\":%d,\"template\":", c->f.arch, c->f.model.name, ctx, c->f.n_vocab, c->f.n_layer, cl_get_threads(), c->f.xs ? "true" : "false", c->f.auto_stream ? "true" : "false", c->f.locked ? "true" : "false", cl_tok_add_bos(c->tok) ? "true" : "false", c->f.n_ctx_train, c->f.kv8 ? (size_t)2 * (size_t)c->f.n_kv_layers * ((size_t)c->f.n_kvw + 4 * (size_t)c->f.n_head_kv) : (size_t)4 * (size_t)c->f.n_kv_layers * (size_t)c->f.n_kvw, c->vis ? "true" : "false", c->f.stream_from < c->f.n_layer ? c->f.pinned_layers : -1);
  ob_json(&c->out, tpl.p ? tpl.p : "", tpl.p ? (int)(tpl.len < 12000 ? tpl.len : 12000) : 0);
  ob_put(&c->out, "}", 1);
  flush_event(c);
  return c;
}

void cl_chat_set_poll(cl_chat *c, cl_chat_poll poll, void *user) { c->poll = poll; c->poll_user = user; }
void cl_chat_cancel(cl_chat *c) { atomic_store(&c->cancel, 1); }

void cl_chat_close(cl_chat *c) {
  if (!c) return;
  cl_fwd_close(&c->f); cl_vision_close(c->vis);
  cl_tok_free(c->tok); cl_meta_free(&c->meta); free(c->tokmap); free(c->logits); free(c->hist); free(c->out.p);
  free(c);
}

static int b64v(int ch) { return ch >= 'A' && ch <= 'Z' ? ch - 'A' : ch >= 'a' && ch <= 'z' ? ch - 'a' + 26 : ch >= '0' && ch <= '9' ? ch - '0' + 52 : ch == '+' ? 62 : ch == '/' ? 63 : -1; }
static uint8_t *b64_decode(const char *s, size_t len, size_t *outn) {
  uint8_t *o = malloc(len / 4 * 3 + 4);
  if (!o) return NULL;
  size_t n = 0; unsigned acc = 0; int bits = 0;
  for (size_t i = 0; i < len; i++) { int v = b64v((unsigned char)s[i]); if (v < 0) continue; acc = acc << 6 | (unsigned)v; bits += 6; if (bits >= 8) { bits -= 8; o[n++] = (uint8_t)(acc >> bits); acc &= (1u << bits) - 1; } }
  *outn = n;
  return o;
}

/* The pictures of an append: {"images":[{"w":448,"h":448,"rgb":"<base64 of w*h*3 bytes>"} or {"path":"picture.ppm"}, ...]}, each encoded to the vectors the model reads. Returns how many (or -1 after an error event). */
static int load_images(cl_chat *c, const char *line, cl_vision_out **outs) {
  const char *v = jfind(line, "images");
  *outs = NULL;
  if (!v || *v != '[') return 0;
  int cap = 0, n = 0;
  cl_vision_out *list = NULL;
  const char *p = v + 1;
  char err[256];
  while (*p) {
    while (*p && *p != '{' && *p != ']') p++;
    if (*p != '{') break;
    const char *e = p;
    while (*e && *e != '}') e++;
    if (!*e) break;
    char *obj = malloc((size_t)(e - p) + 2);
    memcpy(obj, p, (size_t)(e - p) + 1); obj[e - p + 1] = 0;
    p = e + 1;
    int w = (int)jnum(obj, "w", 0), h = (int)jnum(obj, "h", 0);
    uint8_t *rgb = NULL;
    size_t blen = 0;
    char *b64 = jstr(obj, "rgb", &blen), *path = b64 ? NULL : jstr(obj, "path", NULL);
    if (b64) { size_t got = 0; rgb = b64_decode(b64, blen, &got); if (rgb && (w < 1 || h < 1 || got != (size_t)w * h * 3)) { free(rgb); rgb = NULL; } }
    else if (path) rgb = cl_image_load_ppm(path, &w, &h);
    free(b64); free(path); free(obj);
    if (!c->vis) { ev_error(c, "this model was opened without a vision projector (--mmproj), so it cannot take pictures"); free(rgb); goto bad; }
    if (!rgb) { ev_error(c, "a picture could not be read (it needs w, h and rgb, or the path of a P6 PPM file)"); goto bad; }
    int ow, oh;
    cl_vision_fit(w, h, 1024, &ow, &oh);   /* at most 1,024 tokens: one 1,024 x 1,024 picture */
    uint8_t *fit = (ow == w && oh == h) ? rgb : cl_image_resize(rgb, w, h, ow, oh);
    if (fit != rgb) free(rgb);
    if (!fit) { ev_error(c, "out of memory"); goto bad; }
    if (n == cap) { cap = cap ? cap * 2 : 2; cl_vision_out *nl = realloc(list, (size_t)cap * sizeof *list); if (!nl) { free(fit); ev_error(c, "out of memory"); goto bad; } list = nl; }
    double t0 = now_s();
    if (cl_vision_encode(c->vis, fit, ow, oh, &list[n], err, sizeof err)) { free(fit); ev_error(c, err); goto bad; }
    free(fit);
    ob_fmt(&c->out, "{\"ev\":\"vision\",\"tokens\":%d,\"width\":%d,\"height\":%d,\"seconds\":%.3f}", list[n].n, ow, oh, now_s() - t0); flush_event(c);
    n++;
  }
  *outs = list;
  return n;
bad:
  for (int i = 0; i < n; i++) cl_vision_out_free(&list[i]);
  free(list);
  return -1;
}

int cl_chat_run(cl_chat *c, const char *line) {
  cl_fwd *f = &c->f;
  char err[256];
  int rc = 0;
  char *op = jstr(line, "op", NULL);
  if (!op) { ev_error(c, "every line needs an \"op\""); return 0; }
  if (!strcmp(op, "reset")) {
    cl_fwd_reset(f); c->nhist = 0; c->have_logits = 0;
    ob_put(&c->out, "{\"ev\":\"reset\"}", 14); flush_event(c);
  } else if (!strcmp(op, "append")) {
    atomic_store(&c->cancel, 0);
    size_t len = 0;
    char *text = jstr(line, "text", &len);
    if (!text) { ev_error(c, "append needs \"text\""); goto done; }
    cl_vision_out *imgs = NULL;
    int n_img = load_images(c, line, &imgs);
    if (n_img < 0) { free(text); goto done; }
    int cap = (int)len + 8;
    int32_t *ids = malloc((size_t)cap * sizeof *ids);
    int n = 0;
    if (f->pos == 0 && cl_tok_add_bos(c->tok) && cl_tok_bos(c->tok) >= 0) ids[n++] = cl_tok_bos(c->tok);
    int k = cl_tok_encode(c->tok, text, len, 1, ids + n, cap - n);
    if (k < 0) { ev_error(c, "could not tokenize that text"); free(ids); free(text); for (int i = 0; i < n_img; i++) cl_vision_out_free(&imgs[i]); free(imgs); goto done; }
    n += k;
    int n_pads = 0, extra = 0;   /* the picture tokens replace one <|image_pad|> each */
    if (c->vis && c->pad_id >= 0) for (int i = 0; i < n; i++) if (ids[i] == c->pad_id && n_pads < n_img) { extra += imgs[n_pads].n - 1; n_pads++; }
    if (n_pads != n_img || (n_img > 0 && ids[n - 1] == c->pad_id)) {
      ev_error(c, n_pads != n_img ? "the text must hold one <|image_pad|> for each picture" : "a picture cannot be the last thing in a prompt: some text must follow it");
      free(ids); free(text); for (int i = 0; i < n_img; i++) cl_vision_out_free(&imgs[i]); free(imgs); goto done;
    }
    n += extra;
    if (f->pos + n + 1 > c->ctx) {
      ob_fmt(&c->out, "{\"ev\":\"error\",\"msg\":\"the conversation does not fit in the context window\",\"code\":\"context\",\"pos\":%d,\"need\":%d,\"nCtx\":%d}", f->pos, n, c->ctx);
      flush_event(c); free(ids); free(text); for (int i = 0; i < n_img; i++) cl_vision_out_free(&imgs[i]); free(imgs); goto done;
    }
    double t0 = now_s();
    cl_xs_stats a0 = f->xs ? cl_xs_get_stats(f->xs) : (cl_xs_stats){0};
    int fed = 0, cut = 0, at = 0, img_at = 0;   /* `at` walks the encoded text; `fed` counts every token in the cache, picture tokens included */
    const int n_text = n - extra;
    while (at < n_text && !cut) {
      if (c->vis && ids[at] == c->pad_id && img_at < n_img) {   /* a picture: its vectors go in as one chunk */
        const cl_vision_out *im = &imgs[img_at++];
        cl_mm_image mi = {im->n, im->gh, im->gw, im->n_ds, im->emb, {im->ds[0], im->ds[1], im->ds[2]}};
        if (cl_fwd_image(f, &mi, err, sizeof err)) { ev_error(c, err); c->have_logits = 0; cut = 1; break; }
        for (int i = 0; i < im->n; i++) c->hist[c->nhist++] = c->pad_id;
        fed += im->n; at++;
      } else {
        int run = 1;   /* the text up to the next picture */
        while (at + run < n_text && !(c->vis && ids[at + run] == c->pad_id)) run++;
        const int piece = f->xs && !f->hybrid ? CL_PS_B : CL_PF_B;   /* streamed experts: a bigger piece shares each disk read among more tokens */
        int nb = run < piece ? run : piece;
        if (cl_fwd_prefill(f, ids + at, nb, at + nb == n_text ? c->logits : NULL, err, sizeof err)) { ev_error(c, err); c->have_logits = 0; cut = 1; break; }
        for (int i = 0; i < nb; i++) c->hist[c->nhist++] = ids[at + i];
        fed += nb; at += nb;
      }
      if (n > 16) {
        ob_fmt(&c->out, "{\"ev\":\"progress\",\"done\":%d,\"total\":%d}", fed, n); flush_event(c);
        if (at < n_text && cancelled(c)) { cut = 1; break; }
      }
    }
    for (int i = 0; i < n_img; i++) cl_vision_out_free(&imgs[i]);
    free(imgs);
    c->have_logits = !cut && fed > 0;
    ob_fmt(&c->out, "{\"ev\":\"appended\",\"tokens\":%d,\"requested\":%d,\"pos\":%d,\"seconds\":%.3f,\"cancelled\":%s", fed, n, f->pos, now_s() - t0, cut ? "true" : "false");
    if (f->xs) {   /* what the prompt cost in disk reads: the hit rate, the bytes read, and how long the compute thread sat waiting for them */
      cl_xs_stats a1 = cl_xs_get_stats(f->xs);
      uint64_t acc = a1.accesses - a0.accesses, hit = a1.hits - a0.hits;
      ob_fmt(&c->out, ",\"cache\":{\"hitRate\":%.4f,\"gbRead\":%.4f,\"waitSeconds\":%.3f}", acc ? (double)hit / (double)acc : 0.0, (double)((a1.demand_bytes + a1.prefetch_bytes) - (a0.demand_bytes + a0.prefetch_bytes)) / 1e9, (double)(a1.wait_ns - a0.wait_ns) / 1e9);
    }
    ob_put(&c->out, "}", 1);
    flush_event(c);
    free(ids); free(text);
  } else if (!strcmp(op, "generate")) {
    atomic_store(&c->cancel, 0);
    if (!c->have_logits) { ev_error(c, "nothing to continue from: append the prompt first"); goto done; }
    int maxn = (int)jnum(line, "max", 512), top_k = (int)jnum(line, "topK", 40);
    double temp = jnum(line, "temp", 0.7), top_p = jnum(line, "topP", 0.95), rep = jnum(line, "repeat", 1.0);
    double seed = jnum(line, "seed", -1), min_p = jnum(line, "minP", 0);
    const char *jv = jfind(line, "json");
    json_ctx jc = {c->tok, {0}};
    int json_on = jv && *jv == 't', json_complete = 0;
    if (json_on) {   /* JSON mode: the reply must be one JSON object. jsonPrefix is the part of it already written (a reply being continued). */
      cl_json_init(&jc.st, 1);
      size_t pl = 0; char *pre = jstr(line, "jsonPrefix", &pl);
      if (pre && pl && !cl_json_feed(&jc.st, pre, pl)) json_on = 0;   /* a prefix that is not JSON cannot be continued as JSON: leave the reply unconstrained */
      free(pre);
      if (json_on && cl_json_done(&jc.st)) json_complete = 1;   /* already a whole object: nothing more to write */
    }
    if (seed >= 0) rng_s = (uint64_t)seed * 0x9E3779B97F4A7C15ull + 88172645463325252ull;
    unsigned char pend[8]; int np = 0;
    double t0 = now_s(), tfirst = 0;
    int gen = 0, desync = 0;
    int32_t *gids = malloc((size_t)(maxn > 0 ? maxn : 1) * sizeof *gids);
    const char *finish = "length";
    cl_xs_stats s0 = f->xs ? cl_xs_get_stats(f->xs) : (cl_xs_stats){0};
    /* Speculation by prompt lookup (see lookup_draft). Off unless asked for ("spec":1 or CLUTOR_SPEC=1): it gives exactly the same tokens, but in the measurements it was not faster (see the changelog). */
    enum { SPEC_K = 6, SPEC_N = 3 };
    const char *sv = jfind(line, "spec"); const char *se = getenv("CLUTOR_SPEC");
    const int spec_on = !f->hybrid && ((sv && *sv == '1') || (se && *se == '1')) && f->ps_cap >= SPEC_K + 1;
    float *Lall = NULL;
    int32_t q[SPEC_K + 2]; int qfed[SPEC_K + 2], qn = 0, qi = 0;   /* tokens already decided, to be emitted next: accepted guesses (already in the model) and the token that ended the guess (not yet) */
    uint64_t sp_drafted = 0, sp_accepted = 0, sp_runs = 0; int sp_miss = 0, sp_pause = 0;
    for (;;) {
      if (json_complete) { finish = "stop"; break; }
      int already_fed = 0;
      int b = qi < qn ? (already_fed = qfed[qi], q[qi++]) : sample_token(c->logits, f->n_vocab, temp, top_k, top_p, min_p, rep, c->hist + (c->nhist > 64 ? c->nhist - 64 : 0), c->nhist > 64 ? 64 : c->nhist, json_on ? json_tok_ok : NULL, &jc);
      if (b < 0 || cl_tok_is_eog(c->tok, b)) { finish = "stop"; break; }
      char piece[256];
      int pl = cl_tok_piece(c->tok, b, piece, sizeof piece);
      int json_done = 0;
      if (json_on) { cl_json_feed(&jc.st, piece, (size_t)pl); json_done = cl_json_done(&jc.st); }
      unsigned char tmp[272];
      memcpy(tmp, pend, (size_t)np); memcpy(tmp + np, piece, (size_t)pl);
      int total = np + pl, whole = utf8_complete(tmp, total);
      if (whole > 0) { ob_put(&c->out, "{\"ev\":\"t\",\"s\":", 14); ob_json(&c->out, (const char *)tmp, whole); ob_put(&c->out, "}", 1); flush_event(c); }
      np = total - whole; memcpy(pend, tmp + whole, (size_t)np);
      if (gen < maxn) gids[gen] = b;
      gen++;
      if (!tfirst) tfirst = now_s();
      if (already_fed) {   /* an accepted guess: the model already holds it (and the history was extended when it was accepted) */
        if (json_done) { finish = "stop"; break; }
        if (gen >= maxn) { finish = "length"; break; }
        if (cancelled(c)) { finish = "cancel"; break; }
        continue;
      }
      if (f->pos + 1 >= c->ctx) { finish = "context"; desync = 1; break; }
      /* a guess for what follows this token, if the text has repeated itself and the guess has been paying off */
      int32_t dr[SPEC_K]; int k = 0;
      if (spec_on && !json_done && gen < maxn) {
        if (sp_pause > 0) sp_pause--;
        else {
          k = lookup_draft(c->hist, c->nhist, b, dr, SPEC_K, SPEC_N);
          if (k > maxn - gen - 1) k = maxn - gen - 1;
          if (k > c->ctx - f->pos - 3) k = c->ctx - f->pos - 3;
          if (k < 1) k = 0;
          for (int i = 0; i < k; i++) if (cl_tok_is_eog(c->tok, dr[i])) { k = i; break; }
        }
      }
      if (k == 0) {
        if (cl_fwd_step(f, b, c->logits, NULL, err, sizeof err)) { ev_error(c, err); finish = "error"; desync = 1; c->have_logits = 0; break; }
        c->hist[c->nhist++] = b;
      } else {
        if (!Lall) Lall = malloc((size_t)(SPEC_K + 1) * (size_t)f->n_vocab * sizeof *Lall);
        int batch[SPEC_K + 1]; batch[0] = b; for (int i = 0; i < k; i++) batch[i + 1] = dr[i];
        if (!Lall) { snprintf(err, sizeof err, "out of memory"); ev_error(c, err); finish = "error"; desync = 1; c->have_logits = 0; break; }
        const int pos0 = f->pos, nh0 = c->nhist;
        c->hist[c->nhist++] = b;
        if (cl_fwd_verify(f, batch, k + 1, Lall, err, sizeof err)) { ev_error(c, err); finish = "error"; desync = 1; c->have_logits = 0; break; }
        /* walk the distributions: the token drawn after each one either is the guessed token (kept, it is already in the model) or is not (it ends the guess and is emitted next) */
        json_ctx jw = jc;
        qn = qi = 0; int acc = 0;
        for (int i = 0; i <= k; i++) {
          memcpy(c->logits, Lall + (size_t)i * f->n_vocab, (size_t)f->n_vocab * sizeof *Lall);
          const int sm = sample_token(c->logits, f->n_vocab, temp, top_k, top_p, min_p, rep, c->hist + (c->nhist > 64 ? c->nhist - 64 : 0), c->nhist > 64 ? 64 : c->nhist, json_on ? json_tok_ok : NULL, &jw);
          if (i < k && sm == dr[i]) {
            q[qn] = sm; qfed[qn++] = 1; c->hist[c->nhist++] = sm; acc++;
            if (json_on) { char pc[256]; int pn = cl_tok_piece(c->tok, sm, pc, sizeof pc); if (pn > 0) cl_json_feed(&jw.st, pc, (size_t)pn); }
            continue;
          }
          q[qn] = sm; qfed[qn++] = 0;   /* a token not yet in the model (or -1 when the constraint refused everything: the loop stops on it) */
          break;
        }
        f->pos = pos0 + 1 + acc;   /* take back what was fed beyond the accepted guesses */
        (void)nh0;
        sp_drafted += (uint64_t)k; sp_accepted += (uint64_t)acc; sp_runs++;
        if (acc == 0) { if (++sp_miss >= 3) { sp_pause = 16; sp_miss = 0; } } else sp_miss = 0;
      }
      if (json_done) { finish = "stop"; break; }   /* the object is complete: nothing more may follow it */
      if (gen >= maxn) { finish = "length"; break; }
      if (cancelled(c)) { finish = "cancel"; break; }
    }
    /* tokens that were accepted but never emitted (the reply ended first) are taken back, so the model holds exactly what was sent */
    if (qi < qn) {
      for (int j = qi; j < qn; j++) if (qfed[j]) { f->pos--; c->nhist--; }
      /* the logits held now belong to a different position: run the last token that stays once more to get the right ones (a later "generate" may continue from them) */
      if (finish[0] != 'e' && c->nhist > 0 && f->pos > 0) {
        f->pos--; c->nhist--;
        if (cl_fwd_step(f, c->hist[c->nhist], c->logits, NULL, err, sizeof err)) { ev_error(c, err); desync = 1; c->have_logits = 0; }
        else c->nhist++;
      }
    }
    free(Lall);
    if (np > 0) { ob_put(&c->out, "{\"ev\":\"t\",\"s\":", 14); ob_json(&c->out, (const char *)pend, np); ob_put(&c->out, "}", 1); flush_event(c); }
    double secs = now_s() - t0;
    double dec = tfirst && gen > 1 ? now_s() - tfirst : 0;
    ob_fmt(&c->out, "{\"ev\":\"done\",\"finish\":\"%s\",\"tokens\":%d,\"seconds\":%.3f,\"tokensPerSecond\":%.3f,\"pos\":%d,\"desync\":%s", finish, gen, secs, dec > 0 ? (gen - 1) / dec : (secs > 0 ? gen / secs : 0.0), f->pos, desync ? "true" : "false");
    if (sp_runs) ob_fmt(&c->out, ",\"spec\":{\"runs\":%llu,\"drafted\":%llu,\"accepted\":%llu}", (unsigned long long)sp_runs, (unsigned long long)sp_drafted, (unsigned long long)sp_accepted);
    ob_put(&c->out, ",\"ids\":[", 8);
    for (int i = 0; i < gen && i < maxn; i++) ob_fmt(&c->out, "%s%d", i ? "," : "", gids[i]);
    ob_put(&c->out, "]", 1);
    free(gids);
    if (f->xs) {
      cl_xs_stats s1 = cl_xs_get_stats(f->xs);
      uint64_t acc = s1.accesses - s0.accesses, hit = s1.hits - s0.hits;
      ob_fmt(&c->out, ",\"cache\":{\"hitRate\":%.4f,\"gbRead\":%.4f}", acc ? (double)hit / (double)acc : 0.0, (double)((s1.demand_bytes + s1.prefetch_bytes) - (s0.demand_bytes + s0.prefetch_bytes)) / 1e9);
    }
    ob_put(&c->out, "}", 1); flush_event(c);
  } else if (!strcmp(op, "cancel")) {
    /* nothing is running; a cancel that arrives late is ignored */
  } else if (!strcmp(op, "quit")) {
    rc = 1;
  } else ev_error(c, "unknown op");
done:
  free(op);
  return rc;
}

uint64_t cl_system_ram(void) { return cl_phys_mem(); }
uint64_t cl_system_avail(void) { return cl_avail_mem(); }
int cl_system_cores(void) { return cl_ncpu(); }
