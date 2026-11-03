#define _GNU_SOURCE
#include "gguf.h"
#include "stream.h"
#include "kernel.h"
#include "forward.h"
#include "tokenizer.h"
#include "quant.h"
#include <math.h>
#include <time.h>
#include <poll.h>
#include <unistd.h>
#include <errno.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VERSION "0.3.1"

static void usage(void) {
  puts("clutor-engine " VERSION "\n"
       "  info <model.gguf> [--json]       Read the header and print where every layer's bytes and experts live\n"
       "  tokenize <model.gguf> TEXT [--no-special]    Print the token ids of TEXT (byte-level BPE built from the model's own tables)\n"
       "  logits <model.gguf> PROMPT [--top N] [--ctx N] [--trace FILE]   Run the REAL forward pass over PROMPT and print the next-token logprobs (olmoe models); --trace writes the router's choices\n"
       "  generate <model.gguf> PROMPT [--n N] [--trace FILE] [--quiet] [--cache-mb N --prefetch off|prev|ahead (off) --prefetch-n N --policy lru|wtinylfu|adaptive --io-threads N --no-os-cache]   Greedy decoding with the real forward pass; --trace records the router's scores for every token; --cache-mb N streams the experts from disk through an N MiB cache instead of mapping the whole file\n"
       "  chat <model.gguf> [--ctx N] [--cache-mb N ...]   A long-lived session over stdin/stdout, one JSON object per line: {\"op\":\"append\",\"text\":...} feeds text into the conversation (the KV cache is kept between calls), {\"op\":\"generate\",\"max\":N,\"temp\":T} streams the reply as {\"ev\":\"t\",\"s\":piece} lines and ends with {\"ev\":\"done\",...}, {\"op\":\"cancel\"} stops it, {\"op\":\"reset\"} forgets the conversation\n"
       "  quality <model.gguf> TEXTFILE --cache-mb N [--smart J,M,TAU] [--max-tokens N]   Teacher-forced perplexity, KL divergence and top-1 agreement of a streamed (optionally SMART-ROUTE) run against exact routing\n"
       "  stream <model.gguf> [options]    Stream experts from disk through the cache for N tokens of routing (synthetic, or the real router with --trace FILE) and report what was measured\n"
       "\n"
       "stream options:\n  --trace FILE   replay the REAL router recorded by `generate --trace` (cyclically) instead of the synthetic one; prefetch then guesses from the previous token's choices\n"
       "  --tokens N (200)  --warmup N (20)  --cache-mb N (256)  --stage N (twice the experts used)\n"
       "  --policy lru|wtinylfu|adaptive (adaptive)   --window-pct N   --smart J,M,TAU | off (off)   --prefetch on|off (on)\n"
       "  --dense auto|resident|ring (auto: resident when the dense weights fit in a quarter of RAM)\n  --io-threads N (2)  --seed N (1)  --locality X (1.0)  --predict-corr X (0.8)  --compute-us N (0)\n  --compute kernel|checksum (kernel: a Q4_0 x Q8_0 dot-product kernel over the loaded bytes; checksum: I/O only)\n"
       "  --shift AT,LEN   scenario: measured tokens AT..AT+LEN-1 are uniform scan noise, the rest one tight fixed topic; --window N  hit rate and peak RSS every N tokens\n  --no-os-cache   ask the OS not to cache what is read, so reads come from the disk\n"
       "  --json");
}

static const char *arg(int argc, char **argv, const char *name) {
  for (int i = 0; i < argc - 1; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
  return NULL;
}
static int flag(int argc, char **argv, const char *name) { for (int i = 0; i < argc; i++) if (!strcmp(argv[i], name)) return 1; return 0; }
static double num(int argc, char **argv, const char *name, double dflt) { const char *v = arg(argc, argv, name); return v ? atof(v) : dflt; }

static int cmd_info(int argc, char **argv) {
  if (argc < 1) { usage(); return 2; }
  cl_model m; char err[256];
  if (cl_model_open(&m, argv[0], err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
  if (flag(argc, argv, "--json")) {
    printf("{\"arch\":\"%s\",\"name\":\"%s\",\"layers\":%u,\"experts\":%u,\"expertsUsed\":%u,\"moeLayers\":%u,\"residentBytes\":%llu,\"maxDenseBytes\":%llu,\"maxExpertBytes\":%llu,\"totalBytes\":%llu,\"unsizedTensors\":%u,\"layerTable\":[",
           m.arch, m.name, m.n_layers, m.n_experts, m.n_used, m.moe_layers, (unsigned long long)m.resident_bytes, (unsigned long long)m.max_dense_bytes, (unsigned long long)m.max_expert_bytes, (unsigned long long)m.total_bytes, m.unsized_tensors);
    for (uint32_t l = 0; l < m.n_layers; l++) printf("%s{\"denseBytes\":%llu,\"expertBytes\":%llu,\"routerBytes\":%llu}", l ? "," : "", (unsigned long long)m.layers[l].dense_bytes, (unsigned long long)m.layers[l].expert_bytes, (unsigned long long)m.layers[l].router_bytes);
    puts("]}");
  } else {
    printf("%s (%s): %u layers, %u experts (%u used), %u of them routed layers\n", m.name, m.arch, m.n_layers, m.n_experts, m.n_used, m.moe_layers);
    printf("resident %.1f MiB, largest layer's dense weights %.1f MiB, one expert %.2f MiB, whole file %.2f GiB\n", m.resident_bytes / 1048576.0, m.max_dense_bytes / 1048576.0, m.max_expert_bytes / 1048576.0, m.total_bytes / 1073741824.0);
  }
  cl_model_close(&m);
  return 0;
}

static void print_json(const cl_model *m, const cl_stream_opts *o, const cl_stream_result *r) {
  const cl_stream_totals *t = &r->t;
  double wall = t->wall_ns / 1e9, readS = t->read_ns_total / 1e9, wait = t->io_wait_ns / 1e9;
  double overlap = readS > 0 ? 1.0 - wait / readS : 0;
  if (overlap < 0) overlap = 0;
  const char *routing = o->trace ? "real" : "synthetic";
  if (o->compute_mode == CL_COMPUTE_KERNEL) printf("{\"provenance\":{\"routing\":\"%s\",\"disk\":\"measured\",\"compute\":\"Q4_0 x Q8_0 dot-product kernel (%s) run over the loaded bytes, read as Q4_0 whatever they are: a real kernel, not this model's maths\"},", routing, cl_kernel_name());
  else printf("{\"provenance\":{\"routing\":\"%s\",\"disk\":\"measured\",\"compute\":\"checksum over the loaded bytes, not matrix maths\"},", routing);
  printf("\"model\":{\"name\":\"%s\",\"arch\":\"%s\",\"layers\":%u,\"experts\":%u,\"expertsUsed\":%u,\"expertBytes\":%llu},", m->name, m->arch, m->n_layers, m->n_experts, m->n_used, (unsigned long long)r->slot_bytes);
  printf("\"config\":{\"policy\":\"%s\",\"cacheSlots\":%u,\"stageSlots\":%u,\"smartRoute\":%s,\"J\":%d,\"M\":%d,\"tau\":%g,\"denseMode\":\"%s\",\"densePreloadBytes\":%llu,\"prefetch\":%s,\"ioThreads\":%d,\"osCacheBypass\":%s,\"computeMode\":\"%s\",\"computeKernel\":\"%s\",\"tokens\":%d,\"warmup\":%d,\"seed\":%llu,\"locality\":%g,\"predictCorr\":%g},",
         o->policy == CL_LRU ? "lru" : o->policy == CL_WTINYLFU ? "wtinylfu" : "adaptive", r->cache_slots, r->stage_slots, o->smart.enabled ? "true" : "false", o->smart.J, o->smart.M, o->smart.tau, r->dense_resident ? "resident" : "ring", (unsigned long long)r->dense_preload_bytes, o->prefetch ? "true" : "false", o->io_threads, o->os_cache_bypass ? "true" : "false", o->compute_mode == CL_COMPUTE_KERNEL ? "kernel" : "checksum", o->compute_mode == CL_COMPUTE_KERNEL ? cl_kernel_name() : "none", o->tokens, o->warmup, (unsigned long long)o->seed, o->locality, o->predict_corr);
  printf("\"totals\":{\"accesses\":%llu,\"hits\":%llu,\"stageHits\":%llu,\"misses\":%llu,\"bypass\":%llu,\"hitRate\":%.4f,\"substituted\":%llu,\"droppedMass\":%.6f,\"maxDeltaP\":%.6f,\"admitRejects\":%llu,\"evictions\":%llu,",
         (unsigned long long)t->accesses, (unsigned long long)t->hits, (unsigned long long)t->stage_hits, (unsigned long long)t->misses, (unsigned long long)t->bypass, t->accesses ? (double)t->hits / t->accesses : 0.0, (unsigned long long)t->substituted, t->dropped_mass, t->max_delta, (unsigned long long)t->admit_rejects, (unsigned long long)t->evictions);
  printf("\"prefetchIssued\":%llu,\"prefetchLate\":%llu,\"prefetchWasted\":%llu,\"demandBytes\":%llu,\"prefetchBytes\":%llu,\"denseBytes\":%llu,",
         (unsigned long long)t->prefetch_issued, (unsigned long long)t->prefetch_late, (unsigned long long)t->prefetch_wasted, (unsigned long long)t->demand_bytes, (unsigned long long)t->prefetch_bytes, (unsigned long long)t->dense_bytes);
  printf("\"wallSeconds\":%.4f,\"ioWaitSeconds\":%.4f,\"computeSeconds\":%.4f,\"readSecondsTotal\":%.4f,\"overlap\":%.4f,\"tokensPerSecond\":%.3f,\"gbPerToken\":%.5f,\"missesPerToken\":%.3f,\"checksum\":\"%llx\"},",
         wall, wait, t->compute_ns / 1e9, readS, overlap, wall > 0 ? o->tokens / wall : 0.0, o->tokens ? (double)(t->demand_bytes + t->prefetch_bytes + t->dense_bytes) / o->tokens / 1e9 : 0.0, o->tokens ? (double)t->misses / o->tokens : 0.0, (unsigned long long)t->checksum);
  printf("\"rss\":{\"startBytes\":%llu,\"peakBytes\":%llu,\"predictedBytes\":%llu},", (unsigned long long)r->rss_start, (unsigned long long)r->rss_peak, (unsigned long long)r->rss_predicted);
  printf("\"scenario\":{\"shiftAt\":%d,\"shiftLen\":%d},\"windows\":[", o->shift_at, o->shift_len);
  for (int i = 0; i < r->n_windows; i++) printf("%s{\"tokEnd\":%d,\"accesses\":%llu,\"hits\":%llu,\"peakRssBytes\":%llu}", i ? "," : "", r->windows[i].tok_end, (unsigned long long)r->windows[i].accesses, (unsigned long long)r->windows[i].hits, (unsigned long long)r->windows[i].peak_rss);
  printf("],");
  printf("\"layers\":[");
  for (uint32_t l = 0; l < m->n_layers; l++) {
    const cl_layer_stats *s = &r->layers[l];
    printf("%s{\"layer\":%u,\"accesses\":%llu,\"hits\":%llu,\"misses\":%llu,\"substituted\":%llu,\"ioWaitSeconds\":%.5f}", l ? "," : "", l, (unsigned long long)s->accesses, (unsigned long long)s->hits, (unsigned long long)s->misses, (unsigned long long)s->substituted, s->io_wait_ns / 1e9);
  }
  puts("]}");
}

static void print_text(const cl_model *m, const cl_stream_opts *o, const cl_stream_result *r) {
  const cl_stream_totals *t = &r->t;
  double wall = t->wall_ns / 1e9;
  printf("%s: %u layers, %u experts (%u used), expert %.2f MiB\n", m->name, m->n_layers, m->n_experts, m->n_used, r->slot_bytes / 1048576.0);
  printf("policy %s, cache %u experts (%.0f MiB) + %u staged, smart-route %s, prefetch %s, dense weights %s, %d io threads%s\n", o->policy == CL_LRU ? "lru" : o->policy == CL_WTINYLFU ? "wtinylfu" : "adaptive", r->cache_slots, r->cache_slots * (double)r->slot_bytes / 1048576.0, r->stage_slots, o->smart.enabled ? "on" : "off", o->prefetch ? "on" : "off", r->dense_resident ? "resident" : "ring (re-read every token)", o->io_threads, o->os_cache_bypass ? ", OS cache bypass requested" : "");
  printf("MEASURED  disk reads: %.2f GB in %.2f s of read time; the compute thread waited %.2f s of %.2f s wall\n", (t->demand_bytes + t->prefetch_bytes + t->dense_bytes) / 1e9, t->read_ns_total / 1e9, t->io_wait_ns / 1e9, wall);
  printf("MEASURED  cache: %.1f%% hits (%llu of %llu), %llu from prefetch, %llu admission rejects, %.1f misses per token\n", t->accesses ? 100.0 * t->hits / t->accesses : 0.0, (unsigned long long)t->hits, (unsigned long long)t->accesses, (unsigned long long)t->stage_hits, (unsigned long long)t->admit_rejects, o->tokens ? (double)t->misses / o->tokens : 0.0);
  printf("MEASURED  prefetch: %llu issued, %llu arrived late, %llu never used\n", (unsigned long long)t->prefetch_issued, (unsigned long long)t->prefetch_late, (unsigned long long)t->prefetch_wasted);
  if (o->smart.enabled) printf("MEASURED  smart-route: %llu substitutions, largest probability moved %.4f (limit %g), total %.4f\n", (unsigned long long)t->substituted, t->max_delta, o->smart.tau, t->dropped_mass);
  if (o->compute_mode == CL_COMPUTE_KERNEL) printf("compute: Q4_0 x Q8_0 dot-product kernel (%s) over every loaded byte, %.2f s in total. The bytes are read as Q4_0 whatever their real type.\n", cl_kernel_name(), t->compute_ns / 1e9);
  printf("routing: %s\n", o->trace ? "REAL: the router scores recorded from this model's own forward pass" : "SYNTHETIC");
  printf("passes: %d tokens in %.2f s = %.2f passes/s. This is a streaming pass over %s routing; the compute is %s, NOT this model's transformer, so this is NOT model tokens per second.\n", o->tokens, wall, wall > 0 ? o->tokens / wall : 0.0, o->trace ? "REAL (replayed)" : "SYNTHETIC", o->compute_mode == CL_COMPUTE_KERNEL ? "a real quantized kernel on bytes that are not this model's weights" : "a checksum");
}


/* Reads a v2 route trace: "token layer score0 ... scoreE-1" per line. Returns tokens x layers x experts floats, or NULL (message printed). */
static float *load_trace(const char *path, int L, int E, int *tokens) {
  FILE *fp = fopen(path, "r");
  if (!fp) { fprintf(stderr, "cannot open trace %s\n", path); return NULL; }
  char *line = NULL; size_t cap = 0;
  int hl = 0, he = 0, maxtok = -1;
  float *buf = NULL; size_t used = 0, alloc = 0;
  while (getline(&line, &cap, fp) > 0) {
    if (line[0] == '#') {
      if (!strstr(line, "route trace v2")) { fprintf(stderr, "%s is not a v2 route trace (make one with `generate --trace`)\n", path); free(line); fclose(fp); return NULL; }
      char *p = strstr(line, "layers="); if (p) hl = atoi(p + 7);
      p = strstr(line, "experts="); if (p) he = atoi(p + 8);
      if (hl != L || he != E) { fprintf(stderr, "the trace is for %d layers x %d experts, this model has %d x %d\n", hl, he, L, E); free(line); fclose(fp); return NULL; }
      continue;
    }
    if (!hl) { fprintf(stderr, "%s has no header line\n", path); free(line); fclose(fp); return NULL; }
    char *p = line; int t = (int)strtol(p, &p, 10), l = (int)strtol(p, &p, 10);
    if (t != maxtok && t != maxtok + 1) { fprintf(stderr, "trace lines out of order\n"); free(line); fclose(fp); free(buf); return NULL; }
    if (t > maxtok) { maxtok = t; }
    size_t need = ((size_t)t * L + (size_t)l + 1) * E;
    if (need > alloc) { alloc = need * 2; buf = realloc(buf, alloc * sizeof(float)); if (!buf) { free(line); fclose(fp); return NULL; } }
    for (int e = 0; e < E; e++) buf[((size_t)t * L + (size_t)l) * E + e] = strtof(p, &p);
    if (need > used) used = need;
  }
  free(line); fclose(fp);
  if (maxtok < 0 || used != (size_t)(maxtok + 1) * L * E) { fprintf(stderr, "the trace is empty or incomplete\n"); free(buf); return NULL; }
  *tokens = maxtok + 1;
  return buf;
}

static int cmd_stream(int argc, char **argv) {
  if (argc < 1) { usage(); return 2; }
  cl_model m; char err[256];
  if (cl_model_open(&m, argv[0], err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
  cl_stream_opts o = {0};
  o.tokens = (int)num(argc, argv, "--tokens", 200); o.warmup = (int)num(argc, argv, "--warmup", 20);
  o.cache_bytes = (uint64_t)(num(argc, argv, "--cache-mb", 256) * 1048576.0);
  o.stage_slots = (int)num(argc, argv, "--stage", 0);
  const char *pol = arg(argc, argv, "--policy");
  o.policy = !pol || !strcmp(pol, "adaptive") ? CL_ADAPTIVE : !strcmp(pol, "lru") ? CL_LRU : CL_WTINYLFU;
  if (pol && strcmp(pol, "lru") && strcmp(pol, "wtinylfu") && strcmp(pol, "adaptive")) { fprintf(stderr, "--policy must be lru, wtinylfu or adaptive\n"); return 2; }
  o.window_pct = (int)num(argc, argv, "--window-pct", 0);
  const char *sm = arg(argc, argv, "--smart");
  if (sm && strcmp(sm, "off")) {
    int J, M; double tau;
    if (sscanf(sm, "%d,%d,%lf", &J, &M, &tau) != 3 || J < 0 || M < 1 || tau < 0) { fprintf(stderr, "--smart wants J,M,TAU (for example 2,12,0.015) or off\n"); return 2; }
    o.smart.enabled = 1; o.smart.J = J; o.smart.M = M; o.smart.tau = tau;
  }
  const char *dn = arg(argc, argv, "--dense");
  o.dense_mode = !dn || !strcmp(dn, "auto") ? CL_DENSE_AUTO : !strcmp(dn, "resident") ? CL_DENSE_RESIDENT : !strcmp(dn, "ring") ? CL_DENSE_RING : -1;
  if (o.dense_mode < 0) { fprintf(stderr, "--dense must be auto, resident or ring\n"); return 2; }
  const char *pf = arg(argc, argv, "--prefetch");
  o.prefetch = !(pf && !strcmp(pf, "off"));
  o.io_threads = (int)num(argc, argv, "--io-threads", 2);
  o.seed = (uint64_t)num(argc, argv, "--seed", 1);
  o.locality = num(argc, argv, "--locality", 1.0);
  o.predict_corr = num(argc, argv, "--predict-corr", 0.8);
  o.compute_us = (int)num(argc, argv, "--compute-us", 0);
  const char *cm = arg(argc, argv, "--compute");
  o.compute_mode = !cm || !strcmp(cm, "kernel") ? CL_COMPUTE_KERNEL : !strcmp(cm, "checksum") ? CL_COMPUTE_CHECKSUM : -1;
  if (o.compute_mode < 0) { fprintf(stderr, "--compute must be kernel or checksum\n"); return 2; }
  o.window_tokens = (int)num(argc, argv, "--window", 0);
  const char *sh = arg(argc, argv, "--shift");
  if (sh) {
    int a, b;
    if (sscanf(sh, "%d,%d", &a, &b) != 2 || a < 0 || b < 1) { fprintf(stderr, "--shift wants AT,LEN (for example 80,10): measured tokens AT..AT+LEN-1 are uniform scan noise\n"); return 2; }
    o.shift_at = a; o.shift_len = b;
  }
  o.os_cache_bypass = flag(argc, argv, "--no-os-cache");
  if (o.tokens < 1 || o.tokens > 1000000 || o.warmup < 0 || o.predict_corr < 0 || o.predict_corr > 1) { fprintf(stderr, "bad --tokens, --warmup or --predict-corr\n"); return 2; }
  float *trace = NULL;
  const char *tp = arg(argc, argv, "--trace");
  if (tp) {
    trace = load_trace(tp, (int)m.n_layers, (int)m.n_experts, &o.trace_tokens);
    if (!trace) return 1;
    if (o.shift_len > 0) { fprintf(stderr, "--shift builds synthetic scan noise and cannot be combined with --trace\n"); return 2; }
    o.trace = trace;
  }
  cl_stream_result r;
  int rc = cl_stream_run(&m, argv[0], &o, &r);
  if (rc) { fprintf(stderr, "%s\n", r.error); cl_stream_result_free(&r); cl_model_close(&m); return 1; }
  if (flag(argc, argv, "--json")) print_json(&m, &o, &r); else print_text(&m, &o, &r);
  cl_stream_result_free(&r);
  cl_model_close(&m);
  return 0;
}


static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

/* A JSON string from bytes that may not be valid UTF-8 (a byte-level token can be half a character): bad bytes become U+FFFD, so the output always parses. */
static void json_str(const char *p, int n) {
  putchar('"');
  for (int i = 0; i < n;) {
    unsigned char c = (unsigned char)p[i];
    if (c == '"' || c == '\\') { printf("\\%c", c); i++; }
    else if (c < 0x20) { printf("\\u%04x", c); i++; }
    else if (c < 0x80) { putchar(c); i++; }
    else {
      int len = c >= 0xF0 && c < 0xF8 ? 4 : c >= 0xE0 ? 3 : c >= 0xC2 && c < 0xE0 ? 2 : 0, ok = len > 0 && i + len <= n && c < 0xF8;
      for (int k = 1; ok && k < len; k++) if (((unsigned char)p[i + k] & 0xC0) != 0x80) ok = 0;
      if (ok) { fwrite(p + i, 1, (size_t)len, stdout); i += len; }
      else { fputs("\xef\xbf\xbd", stdout); i++; }
    }
  }
  putchar('"');
}

static int load_tok(const char *path, cl_meta *meta, uint8_t **map, size_t *msize, cl_tok **tok) {
  char err[256];
  FILE *fp = fopen(path, "rb");
  if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 1; }
  fseek(fp, 0, SEEK_END); long sz = ftell(fp); fclose(fp);
  /* the metadata sits at the front of the file; read only the first part that holds it */
  size_t want = (size_t)sz < ((size_t)64 << 20) ? (size_t)sz : ((size_t)64 << 20);
  *map = malloc(want); *msize = want;
  fp = fopen(path, "rb");
  if (!*map || !fp || fread(*map, 1, want, fp) != want) { if (fp) fclose(fp); fprintf(stderr, "cannot read %s\n", path); return 1; }
  fclose(fp);
  if (cl_meta_parse(meta, *map, want, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
  *tok = cl_tok_load(meta, err, sizeof err);
  if (!*tok) { fprintf(stderr, "%s\n", err); return 1; }
  return 0;
}

static int cmd_tokenize(int argc, char **argv) {
  if (argc < 2) { usage(); return 2; }
  cl_meta meta; uint8_t *map; size_t ms; cl_tok *tok;
  if (load_tok(argv[0], &meta, &map, &ms, &tok)) return 1;
  size_t len = strlen(argv[1]);
  int32_t *ids = malloc((len + 8) * sizeof *ids);
  int n = cl_tok_encode(tok, argv[1], len, !flag(argc, argv, "--no-special"), ids, (int)len + 8);
  if (n < 0) { fprintf(stderr, "could not tokenize that text\n"); return 1; }
  for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", ids[i]);
  putchar('\n');
  return 0;
}

/* --cache-mb N turns on streaming: experts come from disk through an N MiB cache. --prefetch on|off, --policy, --io-threads, --no-os-cache tune it. */
static cl_fwd_opts fwd_opts(int argc, char **argv) {
  cl_fwd_opts o = {0};
  o.cache_bytes = (uint64_t)(num(argc, argv, "--cache-mb", 0) * 1048576.0);
  const char *pf = arg(argc, argv, "--prefetch");
  o.prefetch = pf ? (!strcmp(pf, "on") || !strcmp(pf, "prev") ? 1 : !strcmp(pf, "ahead") ? 2 : 0) : 0;   /* off by default: previous-token prefetch lost on every disk-bound measurement (docs/NATIVE_ENGINE.md) */
  const char *pol = arg(argc, argv, "--policy");
  o.policy = !pol || !strcmp(pol, "adaptive") ? CL_ADAPTIVE : !strcmp(pol, "lru") ? CL_LRU : CL_WTINYLFU;
  o.io_threads = (int)num(argc, argv, "--io-threads", 3);
  o.stage_slots = (int)num(argc, argv, "--stage", 0);
  o.prefetch_n = (int)num(argc, argv, "--prefetch-n", 0);
  o.os_bypass = flag(argc, argv, "--no-os-cache");
  const char *sm = arg(argc, argv, "--smart");
  if (sm && strcmp(sm, "off")) {
    int J, M; double tau;
    if (sscanf(sm, "%d,%d,%lf", &J, &M, &tau) != 3 || J < 0 || M < 1 || tau < 0) { fprintf(stderr, "--smart wants J,M,TAU (for example 4,16,0.02) or off\n"); exit(2); }
    o.smart.enabled = 1; o.smart.J = J; o.smart.M = M; o.smart.tau = tau;
  }
  return o;
}

static void report_xs(cl_fwd *f, double secs, int tokens) {
  if (!f->xs) return;
  cl_xs_stats s = cl_xs_get_stats(f->xs);
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  double rss = (double)ru.ru_maxrss / 1048576.0;
#else
  double rss = (double)ru.ru_maxrss / 1024.0;
#endif
  if (f->smart.enabled) fprintf(stderr, "[smart-route %d,%d,%g: %llu of %llu expert slots (%.1f%%) used a resident expert instead of the router's own choice; probability moved: largest %.4f (limit %g), total %.3f]\n", f->smart.J, f->smart.M, f->smart.tau, (unsigned long long)f->subs, (unsigned long long)f->expert_slots, f->expert_slots ? 100.0 * f->subs / f->expert_slots : 0.0, f->max_delta, f->smart.tau, f->dropped_mass);
  fprintf(stderr, "[streaming: cache %u experts of %llu bytes; %llu accesses, %.1f%% hits (%llu from prefetch), %llu misses, prefetch %llu issued / %llu never used (%.0f%% of predicted experts were chosen); read %.2f GB (%.2f GB/token) in %.2f s of read time; compute thread waited %.2f s of %.2f s; peak RSS %.0f MiB]\n",
          cl_xs_cache_slots(f->xs), (unsigned long long)cl_xs_slot_bytes(f->xs), (unsigned long long)s.accesses, s.accesses ? 100.0 * s.hits / s.accesses : 0.0, (unsigned long long)s.stage_hits, (unsigned long long)s.misses,
          (unsigned long long)s.prefetch_issued, (unsigned long long)s.prefetch_wasted, f->pred_total ? 100.0 * f->pred_hit / f->pred_total : 0.0, (s.demand_bytes + s.prefetch_bytes) / 1e9, tokens ? (s.demand_bytes + s.prefetch_bytes) / 1e9 / tokens : 0.0, s.read_ns / 1e9, s.wait_ns / 1e9, secs, rss);
}

/* One line per (token, layer): "token layer" then the router's raw score for every expert. */
static void write_trace(FILE *tf, const cl_fwd *f, const cl_fwd_route *rt, int tok) {
  for (int l = 0; l < f->n_layer; l++) {
    fprintf(tf, "%d %d", tok, l);
    for (int e = 0; e < f->n_expert; e++) fprintf(tf, " %.6g", rt->logits[(size_t)l * f->n_expert + e]);
    fputc('\n', tf);
  }
}

static int cmd_logits(int argc, char **argv) {
  if (argc < 2) { usage(); return 2; }
  char err[256];
  cl_meta meta; uint8_t *map; size_t ms; cl_tok *tok;
  if (load_tok(argv[0], &meta, &map, &ms, &tok)) return 1;
  size_t len = strlen(argv[1]);
  int cap = (int)len + 8;
  int32_t *ids = malloc((size_t)cap * sizeof *ids);
  int n = 0;
  if (cl_tok_add_bos(tok) && cl_tok_bos(tok) >= 0) ids[n++] = cl_tok_bos(tok);
  int k = cl_tok_encode(tok, argv[1], len, 1, ids + n, cap - n);
  if (k < 0) { fprintf(stderr, "could not tokenize the prompt\n"); return 1; }
  n += k;
  if (n == 0) { fprintf(stderr, "the prompt has no tokens\n"); return 1; }
  int ctx = (int)num(argc, argv, "--ctx", n + 8), top = (int)num(argc, argv, "--top", 20);
  cl_fwd f;
  cl_fwd_opts fo = fwd_opts(argc, argv);
  if (cl_fwd_open(&f, argv[0], ctx, &fo, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
  float *logits = malloc((size_t)f.n_vocab * sizeof *logits);
  cl_fwd_route rt = {malloc((size_t)f.n_layer * f.n_used * sizeof(int)), malloc((size_t)f.n_layer * f.n_used * sizeof(float)), malloc((size_t)f.n_layer * f.n_expert * sizeof(float)), f.n_layer, f.n_used};
  const char *tr = arg(argc, argv, "--trace");
  FILE *tf = tr ? fopen(tr, "w") : NULL;
  if (tf) fprintf(tf, "# clutor route trace v2 model=%s layers=%d experts=%d used=%d\n", f.model.name, f.n_layer, f.n_expert, f.n_used);
  double t0 = now_s();
  for (int i = 0; i < n; i++) {
    if (cl_fwd_step(&f, ids[i], logits, &rt, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    if (tf) write_trace(tf, &f, &rt, i);
  }
  double secs = now_s() - t0;
  if (tf) fclose(tf);
  double mx = logits[0];
  for (int i = 1; i < f.n_vocab; i++) if (logits[i] > mx) mx = logits[i];
  double sum = 0;
  for (int i = 0; i < f.n_vocab; i++) sum += exp((double)logits[i] - mx);
  double lse = mx + log(sum);
  printf("{\"tokens\":[");
  for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
  printf("],\"seconds\":%.3f,\"secondsPerToken\":%.3f,\"threads\":%d,\"top\":[", secs, secs / n, cl_get_threads());
  char *used = calloc((size_t)f.n_vocab, 1);
  for (int r = 0; r < top && r < f.n_vocab; r++) {
    int b = -1;
    for (int i = 0; i < f.n_vocab; i++) if (!used[i] && (b < 0 || logits[i] > logits[b])) b = i;
    used[b] = 1;
    char piece[256];
    int pl = cl_tok_piece(tok, b, piece, sizeof piece);
    printf("%s{\"id\":%d,\"logit\":%.4f,\"logprob\":%.4f,\"piece\":", r ? "," : "", b, logits[b], logits[b] - lse);
    json_str(piece, pl);
    putchar('}');
  }
  puts("]}");
  cl_fwd_close(&f);
  return 0;
}

/* Greedy decoding with the real forward pass: the text and the routing both come from the model itself. */
static int cmd_generate(int argc, char **argv) {
  if (argc < 2) { usage(); return 2; }
  char err[256];
  cl_meta meta; uint8_t *map; size_t ms; cl_tok *tok;
  if (load_tok(argv[0], &meta, &map, &ms, &tok)) return 1;
  size_t len = strlen(argv[1]);
  int want = (int)num(argc, argv, "--n", 32), cap = (int)len + 8;
  int32_t *ids = malloc((size_t)cap * sizeof *ids);
  int n = 0;
  if (cl_tok_add_bos(tok) && cl_tok_bos(tok) >= 0) ids[n++] = cl_tok_bos(tok);
  int k = cl_tok_encode(tok, argv[1], len, 1, ids + n, cap - n);
  if (k < 0) { fprintf(stderr, "could not tokenize the prompt\n"); return 1; }
  n += k;
  if (n == 0 || want < 1) { fprintf(stderr, "need a prompt and --n >= 1\n"); return 1; }
  cl_fwd f;
  cl_fwd_opts fo = fwd_opts(argc, argv);
  if (cl_fwd_open(&f, argv[0], n + want + 1, &fo, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
  float *logits = malloc((size_t)f.n_vocab * sizeof *logits);
  cl_fwd_route rt = {malloc((size_t)f.n_layer * f.n_used * sizeof(int)), malloc((size_t)f.n_layer * f.n_used * sizeof(float)), malloc((size_t)f.n_layer * f.n_expert * sizeof(float)), f.n_layer, f.n_used};
  const char *tr = arg(argc, argv, "--trace");
  FILE *tf = tr ? fopen(tr, "w") : NULL;
  if (tf) fprintf(tf, "# clutor route trace v2 model=%s layers=%d experts=%d used=%d\n", f.model.name, f.n_layer, f.n_expert, f.n_used);
  int quiet = flag(argc, argv, "--quiet");
  if (!quiet) { fputs(argv[1], stdout); fflush(stdout); }
  double t0 = now_s();
  int fed = 0, gen = 0, cur = ids[0];
  int *outids = malloc((size_t)(want + 2) * sizeof(int));
  int nout = 0;
  for (;;) {
    if (cl_fwd_step(&f, cur, logits, &rt, err, sizeof err)) { fprintf(stderr, "\n%s\n", err); return 1; }
    if (tf) write_trace(tf, &f, &rt, fed);
    fed++;
    if (fed < n) { cur = ids[fed]; continue; }       /* still reading the prompt */
    int b = 0;
    for (int i = 1; i < f.n_vocab; i++) if (logits[i] > logits[b]) b = i;
    gen++;
    if (cl_tok_is_eog(tok, b) || gen > want) break;
    char piece[256];
    int pl = cl_tok_piece(tok, b, piece, sizeof piece);
    if (!quiet) { fwrite(piece, 1, (size_t)pl, stdout); fflush(stdout); }
    cur = b;
    outids[nout++] = b;
    if (gen == want) break;
  }
  double secs = now_s() - t0;
  if (tf) fclose(tf);
  const char *idf = arg(argc, argv, "--ids");   /* the token ids in the format colibri's olmoe harness reads (ref.json): prompt and prompt + generated */
  if (idf) {
    FILE *jf = fopen(idf, "w");
    if (jf) {
      fprintf(jf, "{\n  \"prompt_ids\": [");
      for (int i = 0; i < n; i++) fprintf(jf, "%s%d", i ? ", " : "", ids[i]);
      fprintf(jf, "],\n  \"full_ids\": [");
      for (int i = 0; i < n; i++) fprintf(jf, "%s%d", i ? ", " : "", ids[i]);
      for (int i = 0; i < nout; i++) fprintf(jf, ", %d", outids[i]);
      fprintf(jf, "]\n}\n");
      fclose(jf);
    }
  }
  fprintf(stderr, "\n[%d tokens fed (%d prompt + %d generated) in %.1f s = %.2f s/token on %d threads, %s]\n", fed, n, fed - n, secs, secs / fed, cl_get_threads(), tr ? "route trace written" : "no trace");
  report_xs(&f, secs, fed);
  cl_fwd_close(&f);
  return 0;
}

static double logprob_of(const float *logits, int n, int tok, double *lse_out) {
  double mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  double s = 0;
  for (int i = 0; i < n; i++) s += exp((double)logits[i] - mx);
  double lse = mx + log(s);
  if (lse_out) *lse_out = lse;
  return (double)logits[tok] - lse;
}

/* Teacher-forced comparison of exact routing (the whole file mapped) with the configured run (streamed experts, optionally SMART-ROUTE) on a text file. */
static int cmd_quality(int argc, char **argv) {
  if (argc < 2) { usage(); return 2; }
  char err[256];
  cl_meta meta; uint8_t *map; size_t ms; cl_tok *tok;
  if (load_tok(argv[0], &meta, &map, &ms, &tok)) return 1;
  FILE *tf = fopen(argv[1], "rb");
  if (!tf) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  fseek(tf, 0, SEEK_END); long tsz = ftell(tf); fseek(tf, 0, SEEK_SET);
  char *text = malloc((size_t)tsz + 1);
  if (!text || fread(text, 1, (size_t)tsz, tf) != (size_t)tsz) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  fclose(tf);
  int maxtok = (int)num(argc, argv, "--max-tokens", 512);
  int32_t *ids = malloc(((size_t)tsz + 8) * sizeof *ids);
  int n = cl_tok_encode(tok, text, (size_t)tsz, 0, ids, (int)tsz + 8);
  if (n < 2) { fprintf(stderr, "the text has fewer than 2 tokens\n"); return 1; }
  if (n > maxtok) n = maxtok;
  cl_fwd a, b;
  cl_fwd_opts none = {0}, fo = fwd_opts(argc, argv);
  if (!fo.cache_bytes) { fprintf(stderr, "quality compares against streamed runs: give --cache-mb N\n"); return 2; }
  if (cl_fwd_open(&a, argv[0], n + 1, &none, err, sizeof err) || cl_fwd_open(&b, argv[0], n + 1, &fo, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
  float *la = malloc((size_t)a.n_vocab * sizeof *la), *lb = malloc((size_t)a.n_vocab * sizeof *lb);
  double nllA = 0, nllB = 0, kl = 0;
  int agree = 0, used = 0;
  double t0 = now_s();
  for (int i = 0; i + 1 < n; i++) {
    if (cl_fwd_step(&a, ids[i], la, NULL, err, sizeof err) || cl_fwd_step(&b, ids[i], lb, NULL, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    double lseA, lseB;
    double pa = logprob_of(la, a.n_vocab, ids[i + 1], &lseA), pb = logprob_of(lb, a.n_vocab, ids[i + 1], &lseB);
    nllA -= pa; nllB -= pb;
    double k = 0;
    int ba = 0, bb = 0;
    for (int v = 0; v < a.n_vocab; v++) {
      double lpa = (double)la[v] - lseA, lpb = (double)lb[v] - lseB;
      k += exp(lpa) * (lpa - lpb);
      if (la[v] > la[ba]) ba = v;
      if (lb[v] > lb[bb]) bb = v;
    }
    kl += k; agree += ba == bb; used++;
  }
  double secs = now_s() - t0;
  cl_xs_stats s = cl_xs_get_stats(b.xs);
  printf("{\"tokensScored\":%d,\"perplexityExact\":%.4f,\"perplexityRun\":%.4f,\"perplexityRatio\":%.4f,\"meanKL\":%.5f,\"top1Agreement\":%.4f,", used, exp(nllA / used), exp(nllB / used), exp((nllB - nllA) / used), kl / used, (double)agree / used);
  printf("\"smart\":%s,\"substitutions\":%llu,\"expertSlots\":%llu,\"maxProbMoved\":%.5f,\"totalProbMoved\":%.4f,", b.smart.enabled ? "true" : "false", (unsigned long long)b.subs, (unsigned long long)b.expert_slots, b.max_delta, b.dropped_mass);
  printf("\"cacheSlots\":%u,\"hitRate\":%.4f,\"gbPerToken\":%.4f,\"seconds\":%.1f,\"note\":\"the exact run maps the whole file, so the OS cache is warm: seconds here say nothing about disk speed\"}\n",
         cl_xs_cache_slots(b.xs), s.accesses ? (double)s.hits / s.accesses : 0.0, (s.demand_bytes + s.prefetch_bytes) / 1e9 / used, secs);
  return 0;
}


/* ---- chat: a long-lived conversation over stdin/stdout ------------------------------------------------------------------------------------------------
 * The caller (the clutor terminal UI) formats the chat template and sends text; this process tokenizes it, keeps the KV cache between calls so a
 * turn only pays for its new tokens, samples the reply and streams it. Lines in, lines out, each one JSON object. */
static struct { char *buf; size_t len, cap; int eof; } ib;

/* Reads whatever is available (block: wait for at least one byte). Returns 1 if bytes arrived, 0 if none, -1 at end of input. */
static int ib_fill(int block) {
  if (ib.eof) return -1;
  if (!block) { struct pollfd p = {0, POLLIN, 0}; int r = poll(&p, 1, 0); if (r < 0 && errno == EINTR) return 0; if (r <= 0) return 0; }
  while (ib.cap - ib.len < 65536) {
    if (ib.cap > ((size_t)256 << 20)) return -1;
    ib.cap = ib.cap ? ib.cap * 2 : (1u << 20);
    ib.buf = realloc(ib.buf, ib.cap);
    if (!ib.buf) return -1;
  }
  ssize_t n;
  do n = read(0, ib.buf + ib.len, ib.cap - ib.len); while (n < 0 && errno == EINTR);
  if (n <= 0) { ib.eof = 1; return -1; }
  ib.len += (size_t)n;
  return 1;
}

/* The next complete line (malloc'd, no newline), waiting for it; NULL at end of input. */
static char *ib_line(void) {
  for (;;) {
    char *nl = ib.len ? memchr(ib.buf, '\n', ib.len) : NULL;
    if (nl) {
      size_t n = (size_t)(nl - ib.buf);
      char *s = malloc(n + 1);
      memcpy(s, ib.buf, n); s[n] = 0;
      memmove(ib.buf, nl + 1, ib.len - n - 1); ib.len -= n + 1;
      return s;
    }
    if (ib_fill(1) < 0) return NULL;
  }
}

/* True if a {"op":"cancel"} line has arrived (it is removed); input that has ended counts as a cancel too. Other lines wait where they are. */
static int ib_cancelled(void) {
  int r = ib_fill(0);
  if (r < 0 && ib.len == 0) return 1;
  size_t start = 0;
  while (start < ib.len) {
    char *nl = memchr(ib.buf + start, '\n', ib.len - start);
    if (!nl) break;
    size_t n = (size_t)(nl - (ib.buf + start));
    if (n < 80 && memmem(ib.buf + start, n, "\"cancel\"", 8)) {
      memmove(ib.buf + start, nl + 1, ib.len - start - n - 1); ib.len -= n + 1;
      return 1;
    }
    start += n + 1;
  }
  return 0;
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

static void ev_error(const char *msg) { fputs("{\"ev\":\"error\",\"msg\":", stdout); json_str(msg, (int)strlen(msg)); puts("}"); fflush(stdout); }

static int cmd_chat(int argc, char **argv) {
  if (argc < 1) { usage(); return 2; }
  char err[256];
  cl_meta meta; uint8_t *map; size_t ms; cl_tok *tok;
  if (load_tok(argv[0], &meta, &map, &ms, &tok)) return 1;
  int ctx = (int)num(argc, argv, "--ctx", 4096);
  if (ctx < 64) ctx = 64;
  cl_fwd f;
  cl_fwd_opts fo = fwd_opts(argc, argv);
  if (cl_fwd_open(&f, argv[0], ctx, &fo, err, sizeof err)) { ev_error(err); return 1; }
  if (f.n_ctx_max > 0 && ctx > f.n_ctx_max) ctx = f.n_ctx_max;   /* the model's own limit; the caller reads the real size from the ready line */
  float *logits = malloc((size_t)f.n_vocab * sizeof *logits);
  int32_t *hist = malloc((size_t)ctx * sizeof *hist);   /* every token in the KV cache, for the repetition penalty */
  int nhist = 0, have_logits = 0;
  cl_str tpl = {0};
  cl_meta_str(&meta, "tokenizer.chat_template", &tpl);
  printf("{\"ev\":\"ready\",\"arch\":\"%s\",\"name\":\"%s\",\"nCtx\":%d,\"nVocab\":%d,\"nLayer\":%d,\"threads\":%d,\"streaming\":%s,\"addBos\":%s,\"template\":", f.arch, f.model.name, ctx, f.n_vocab, f.n_layer, cl_get_threads(), f.xs ? "true" : "false", cl_tok_add_bos(tok) ? "true" : "false");
  json_str(tpl.p ? tpl.p : "", tpl.p ? (int)(tpl.len < 12000 ? tpl.len : 12000) : 0);
  puts("}");
  fflush(stdout);

  char *line;
  while ((line = ib_line())) {
    char *op = jstr(line, "op", NULL);
    if (!op) { ev_error("every line needs an \"op\""); free(line); continue; }
    if (!strcmp(op, "reset")) {
      cl_fwd_reset(&f); nhist = 0; have_logits = 0;
      puts("{\"ev\":\"reset\"}"); fflush(stdout);
    } else if (!strcmp(op, "append")) {
      size_t len = 0;
      char *text = jstr(line, "text", &len);
      if (!text) { ev_error("append needs \"text\""); goto next; }
      int cap = (int)len + 8;
      int32_t *ids = malloc((size_t)cap * sizeof *ids);
      int n = 0;
      if (f.pos == 0 && cl_tok_add_bos(tok) && cl_tok_bos(tok) >= 0) ids[n++] = cl_tok_bos(tok);
      int k = cl_tok_encode(tok, text, len, 1, ids + n, cap - n);
      if (k < 0) { ev_error("could not tokenize that text"); free(ids); free(text); goto next; }
      n += k;
      if (f.pos + n + 1 > ctx) {
        printf("{\"ev\":\"error\",\"msg\":\"the conversation does not fit in the context window\",\"code\":\"context\",\"pos\":%d,\"need\":%d,\"nCtx\":%d}\n", f.pos, n, ctx);
        fflush(stdout); free(ids); free(text); goto next;
      }
      double t0 = now_s();
      int fed = 0, cancelled = 0;
      for (; fed < n; fed++) {
        if (cl_fwd_step(&f, ids[fed], logits, NULL, err, sizeof err)) { ev_error(err); have_logits = 0; cancelled = 1; break; }
        hist[nhist++] = ids[fed];
        if (n > 16 && (fed % 8 == 7 || fed + 1 == n)) {
          printf("{\"ev\":\"progress\",\"done\":%d,\"total\":%d}\n", fed + 1, n); fflush(stdout);
          if (fed + 1 < n && ib_cancelled()) { fed++; cancelled = 1; break; }
        }
      }
      have_logits = !cancelled && fed > 0;
      printf("{\"ev\":\"appended\",\"tokens\":%d,\"requested\":%d,\"pos\":%d,\"seconds\":%.3f,\"cancelled\":%s}\n", fed, n, f.pos, now_s() - t0, cancelled ? "true" : "false");
      fflush(stdout);
      free(ids); free(text);
    } else if (!strcmp(op, "generate")) {
      if (!have_logits) { ev_error("nothing to continue from: append the prompt first"); goto next; }
      int maxn = (int)jnum(line, "max", 512), top_k = (int)jnum(line, "topK", 40);
      double temp = jnum(line, "temp", 0.7), top_p = jnum(line, "topP", 0.95), rep = jnum(line, "repeat", 1.0);
      double seed = jnum(line, "seed", -1);
      if (seed >= 0) rng_s = (uint64_t)seed * 0x9E3779B97F4A7C15ull + 88172645463325252ull;
      unsigned char pend[8]; int np = 0;
      double t0 = now_s(), tfirst = 0;
      int gen = 0, desync = 0;
      int32_t *gids = malloc((size_t)(maxn > 0 ? maxn : 1) * sizeof *gids);
      const char *finish = "length";
      cl_xs_stats s0 = f.xs ? cl_xs_get_stats(f.xs) : (cl_xs_stats){0};
      for (;;) {
        int b = sample_token(logits, f.n_vocab, temp, top_k, top_p, rep, hist + (nhist > 64 ? nhist - 64 : 0), nhist > 64 ? 64 : nhist);
        if (cl_tok_is_eog(tok, b)) { finish = "stop"; break; }
        char piece[256];
        int pl = cl_tok_piece(tok, b, piece, sizeof piece);
        unsigned char tmp[272];
        memcpy(tmp, pend, (size_t)np); memcpy(tmp + np, piece, (size_t)pl);
        int total = np + pl, whole = utf8_complete(tmp, total);
        if (whole > 0) { fputs("{\"ev\":\"t\",\"s\":", stdout); json_str((const char *)tmp, whole); puts("}"); fflush(stdout); }
        np = total - whole; memcpy(pend, tmp + whole, (size_t)np);
        if (gen < maxn) gids[gen] = b;
        gen++;
        if (!tfirst) tfirst = now_s();
        if (f.pos + 1 >= ctx) { finish = "context"; desync = 1; break; }
        if (cl_fwd_step(&f, b, logits, NULL, err, sizeof err)) { ev_error(err); finish = "error"; desync = 1; have_logits = 0; break; }
        hist[nhist++] = b;
        if (gen >= maxn) { finish = "length"; break; }
        if (ib_cancelled()) { finish = "cancel"; break; }
      }
      if (np > 0) { fputs("{\"ev\":\"t\",\"s\":", stdout); json_str((const char *)pend, np); puts("}"); }
      double secs = now_s() - t0;
      double dec = tfirst && gen > 1 ? now_s() - tfirst : 0;
      printf("{\"ev\":\"done\",\"finish\":\"%s\",\"tokens\":%d,\"seconds\":%.3f,\"tokensPerSecond\":%.3f,\"pos\":%d,\"desync\":%s", finish, gen, secs, dec > 0 ? (gen - 1) / dec : (secs > 0 ? gen / secs : 0.0), f.pos, desync ? "true" : "false");
      printf(",\"ids\":[");
      for (int i = 0; i < gen && i < maxn; i++) printf("%s%d", i ? "," : "", gids[i]);
      putchar(']');
      free(gids);
      if (f.xs) {
        cl_xs_stats s1 = cl_xs_get_stats(f.xs);
        uint64_t acc = s1.accesses - s0.accesses, hit = s1.hits - s0.hits;
        printf(",\"cache\":{\"hitRate\":%.4f,\"gbRead\":%.4f}", acc ? (double)hit / (double)acc : 0.0, (double)((s1.demand_bytes + s1.prefetch_bytes) - (s0.demand_bytes + s0.prefetch_bytes)) / 1e9);
      }
      puts("}"); fflush(stdout);
    } else if (!strcmp(op, "cancel")) {
      /* nothing is running; a cancel that arrives late is ignored */
    } else if (!strcmp(op, "quit")) {
      free(op); free(line); break;
    } else ev_error("unknown op");
next:
    free(op); free(line);
  }
  cl_fwd_close(&f);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) { usage(); return 2; }
  if (!strcmp(argv[1], "--version")) { puts(VERSION); return 0; }
  if (!strcmp(argv[1], "info")) return cmd_info(argc - 2, argv + 2);
  if (!strcmp(argv[1], "tokenize")) return cmd_tokenize(argc - 2, argv + 2);
  if (!strcmp(argv[1], "quality")) return cmd_quality(argc - 2, argv + 2);
  if (!strcmp(argv[1], "chat")) return cmd_chat(argc - 2, argv + 2);
  if (!strcmp(argv[1], "generate")) return cmd_generate(argc - 2, argv + 2);
  if (!strcmp(argv[1], "logits")) return cmd_logits(argc - 2, argv + 2);
  if (!strcmp(argv[1], "stream")) return cmd_stream(argc - 2, argv + 2);
  usage();
  return strcmp(argv[1], "--help") ? 2 : 0;
}
