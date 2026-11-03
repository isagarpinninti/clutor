#define _GNU_SOURCE
#include "chat.h"
#include "plat.h"
#include "forward.h"
#include "tokenizer.h"
#include "quant.h"
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

/* temp <= 0: the most likely token. Otherwise: repetition penalty over `recent`, the top_k best, temperature, top_p, then a draw. */
static int sample_token(float *logits, int n, double temp, int top_k, double top_p, double rep, const int32_t *recent, int nrecent) {
  if (rep > 1.0) for (int i = 0; i < nrecent; i++) { float *l = &logits[recent[i]]; *l = *l > 0 ? (float)(*l / rep) : (float)(*l * rep); }
  if (temp <= 0) { int b = 0; for (int i = 1; i < n; i++) if (logits[i] > logits[b]) b = i; return b; }
  if (top_k < 1) top_k = 1;
  if (top_k > 200) top_k = 200;
  int idx[200]; float val[200]; int k = 0;
  for (int i = 0; i < n; i++) {
    float v = logits[i];
    if (k == top_k && v <= val[k - 1]) continue;
    int p = k < top_k ? k++ : k - 1;
    while (p > 0 && val[p - 1] < v) { val[p] = val[p - 1]; idx[p] = idx[p - 1]; p--; }
    val[p] = v; idx[p] = i;
  }
  double mx = val[0], sum = 0, pr[200];
  for (int i = 0; i < k; i++) { pr[i] = exp((val[i] - mx) / temp); sum += pr[i]; }
  double cum = 0; int keep = k;
  for (int i = 0; i < k; i++) { cum += pr[i] / sum; if (cum >= top_p) { keep = i + 1; break; } }
  double tot = 0;
  for (int i = 0; i < keep; i++) tot += pr[i];
  double r = rnd() * tot;
  for (int i = 0; i < keep; i++) { r -= pr[i]; if (r <= 0) return idx[i]; }
  return idx[keep - 1];
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
  if (cl_fwd_open(&c->f, argv[0], ctx, &fo, err, errlen)) { cl_tok_free(c->tok); free(c->tokmap); free(c); return NULL; }
  if (c->f.n_ctx_max > 0 && ctx > c->f.n_ctx_max) ctx = c->f.n_ctx_max;   /* the model's own limit; the caller reads the real size from the ready event */
  c->ctx = ctx;
  c->logits = malloc((size_t)c->f.n_vocab * sizeof *c->logits);
  c->hist = malloc((size_t)ctx * sizeof *c->hist);   /* every token in the KV cache, for the repetition penalty */
  if (!c->logits || !c->hist) { snprintf(err, errlen, "out of memory"); cl_chat_close(c); return NULL; }
  cl_str tpl = {0};
  cl_meta_str(&c->meta, "tokenizer.chat_template", &tpl);
  ob_fmt(&c->out, "{\"ev\":\"ready\",\"arch\":\"%s\",\"name\":\"%s\",\"nCtx\":%d,\"nVocab\":%d,\"nLayer\":%d,\"threads\":%d,\"streaming\":%s,\"locked\":%s,\"addBos\":%s,\"nCtxTrain\":%d,\"kvBytesPerToken\":%zu,\"template\":", c->f.arch, c->f.model.name, ctx, c->f.n_vocab, c->f.n_layer, cl_get_threads(), c->f.xs ? "true" : "false", c->f.locked ? "true" : "false", cl_tok_add_bos(c->tok) ? "true" : "false", c->f.n_ctx_train, (size_t)8 * (size_t)c->f.n_kv_layers * (size_t)c->f.n_kvw);
  ob_json(&c->out, tpl.p ? tpl.p : "", tpl.p ? (int)(tpl.len < 12000 ? tpl.len : 12000) : 0);
  ob_put(&c->out, "}", 1);
  flush_event(c);
  return c;
}

void cl_chat_set_poll(cl_chat *c, cl_chat_poll poll, void *user) { c->poll = poll; c->poll_user = user; }
void cl_chat_cancel(cl_chat *c) { atomic_store(&c->cancel, 1); }

void cl_chat_close(cl_chat *c) {
  if (!c) return;
  cl_fwd_close(&c->f);
  cl_tok_free(c->tok); free(c->tokmap); free(c->logits); free(c->hist); free(c->out.p);
  free(c);
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
    int cap = (int)len + 8;
    int32_t *ids = malloc((size_t)cap * sizeof *ids);
    int n = 0;
    if (f->pos == 0 && cl_tok_add_bos(c->tok) && cl_tok_bos(c->tok) >= 0) ids[n++] = cl_tok_bos(c->tok);
    int k = cl_tok_encode(c->tok, text, len, 1, ids + n, cap - n);
    if (k < 0) { ev_error(c, "could not tokenize that text"); free(ids); free(text); goto done; }
    n += k;
    if (f->pos + n + 1 > c->ctx) {
      ob_fmt(&c->out, "{\"ev\":\"error\",\"msg\":\"the conversation does not fit in the context window\",\"code\":\"context\",\"pos\":%d,\"need\":%d,\"nCtx\":%d}", f->pos, n, c->ctx);
      flush_event(c); free(ids); free(text); goto done;
    }
    double t0 = now_s();
    int fed = 0, cut = 0;
    while (fed < n) {
      int nb = n - fed < CL_PF_B ? n - fed : CL_PF_B;
      if (cl_fwd_prefill(f, ids + fed, nb, fed + nb == n ? c->logits : NULL, err, sizeof err)) { ev_error(c, err); c->have_logits = 0; cut = 1; break; }
      for (int i = 0; i < nb; i++) c->hist[c->nhist++] = ids[fed + i];
      fed += nb;
      if (n > 16) {
        ob_fmt(&c->out, "{\"ev\":\"progress\",\"done\":%d,\"total\":%d}", fed, n); flush_event(c);
        if (fed < n && cancelled(c)) { cut = 1; break; }
      }
    }
    c->have_logits = !cut && fed > 0;
    ob_fmt(&c->out, "{\"ev\":\"appended\",\"tokens\":%d,\"requested\":%d,\"pos\":%d,\"seconds\":%.3f,\"cancelled\":%s}", fed, n, f->pos, now_s() - t0, cut ? "true" : "false");
    flush_event(c);
    free(ids); free(text);
  } else if (!strcmp(op, "generate")) {
    atomic_store(&c->cancel, 0);
    if (!c->have_logits) { ev_error(c, "nothing to continue from: append the prompt first"); goto done; }
    int maxn = (int)jnum(line, "max", 512), top_k = (int)jnum(line, "topK", 40);
    double temp = jnum(line, "temp", 0.7), top_p = jnum(line, "topP", 0.95), rep = jnum(line, "repeat", 1.0);
    double seed = jnum(line, "seed", -1);
    if (seed >= 0) rng_s = (uint64_t)seed * 0x9E3779B97F4A7C15ull + 88172645463325252ull;
    unsigned char pend[8]; int np = 0;
    double t0 = now_s(), tfirst = 0;
    int gen = 0, desync = 0;
    int32_t *gids = malloc((size_t)(maxn > 0 ? maxn : 1) * sizeof *gids);
    const char *finish = "length";
    cl_xs_stats s0 = f->xs ? cl_xs_get_stats(f->xs) : (cl_xs_stats){0};
    for (;;) {
      int b = sample_token(c->logits, f->n_vocab, temp, top_k, top_p, rep, c->hist + (c->nhist > 64 ? c->nhist - 64 : 0), c->nhist > 64 ? 64 : c->nhist);
      if (cl_tok_is_eog(c->tok, b)) { finish = "stop"; break; }
      char piece[256];
      int pl = cl_tok_piece(c->tok, b, piece, sizeof piece);
      unsigned char tmp[272];
      memcpy(tmp, pend, (size_t)np); memcpy(tmp + np, piece, (size_t)pl);
      int total = np + pl, whole = utf8_complete(tmp, total);
      if (whole > 0) { ob_put(&c->out, "{\"ev\":\"t\",\"s\":", 14); ob_json(&c->out, (const char *)tmp, whole); ob_put(&c->out, "}", 1); flush_event(c); }
      np = total - whole; memcpy(pend, tmp + whole, (size_t)np);
      if (gen < maxn) gids[gen] = b;
      gen++;
      if (!tfirst) tfirst = now_s();
      if (f->pos + 1 >= c->ctx) { finish = "context"; desync = 1; break; }
      if (cl_fwd_step(f, b, c->logits, NULL, err, sizeof err)) { ev_error(c, err); finish = "error"; desync = 1; c->have_logits = 0; break; }
      c->hist[c->nhist++] = b;
      if (gen >= maxn) { finish = "length"; break; }
      if (cancelled(c)) { finish = "cancel"; break; }
    }
    if (np > 0) { ob_put(&c->out, "{\"ev\":\"t\",\"s\":", 14); ob_json(&c->out, (const char *)pend, np); ob_put(&c->out, "}", 1); flush_event(c); }
    double secs = now_s() - t0;
    double dec = tfirst && gen > 1 ? now_s() - tfirst : 0;
    ob_fmt(&c->out, "{\"ev\":\"done\",\"finish\":\"%s\",\"tokens\":%d,\"seconds\":%.3f,\"tokensPerSecond\":%.3f,\"pos\":%d,\"desync\":%s", finish, gen, secs, dec > 0 ? (gen - 1) / dec : (secs > 0 ? gen / secs : 0.0), f->pos, desync ? "true" : "false");
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
int cl_system_cores(void) { return cl_ncpu(); }
