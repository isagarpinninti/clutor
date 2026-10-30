#define _GNU_SOURCE
#include "gguf.h"
#include "stream.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VERSION "0.1.0"

static void usage(void) {
  puts("clutor-engine " VERSION "\n"
       "  info <model.gguf> [--json]       Read the header and print where every layer's bytes and experts live\n"
       "  stream <model.gguf> [options]    Stream experts from disk through the cache for N tokens of SYNTHETIC routing and report what was measured\n"
       "\n"
       "stream options:\n"
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
  if (o->compute_mode == CL_COMPUTE_KERNEL) printf("{\"provenance\":{\"routing\":\"synthetic\",\"disk\":\"measured\",\"compute\":\"Q4_0 x Q8_0 dot-product kernel (%s) run over the loaded bytes, read as Q4_0 whatever they are: a real kernel, not this model's maths\"},", cl_kernel_name());
  else printf("{\"provenance\":{\"routing\":\"synthetic\",\"disk\":\"measured\",\"compute\":\"checksum over the loaded bytes, not matrix maths\"},");
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
  printf("passes: %d tokens in %.2f s = %.2f passes/s. This is a streaming pass over SYNTHETIC routing; the compute is %s, NOT a transformer, so this is NOT model tokens per second.\n", o->tokens, wall, wall > 0 ? o->tokens / wall : 0.0, o->compute_mode == CL_COMPUTE_KERNEL ? "a real quantized kernel on bytes that are not this model's weights" : "a checksum");
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
  cl_stream_result r;
  int rc = cl_stream_run(&m, argv[0], &o, &r);
  if (rc) { fprintf(stderr, "%s\n", r.error); cl_stream_result_free(&r); cl_model_close(&m); return 1; }
  if (flag(argc, argv, "--json")) print_json(&m, &o, &r); else print_text(&m, &o, &r);
  cl_stream_result_free(&r);
  cl_model_close(&m);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) { usage(); return 2; }
  if (!strcmp(argv[1], "--version")) { puts(VERSION); return 0; }
  if (!strcmp(argv[1], "info")) return cmd_info(argc - 2, argv + 2);
  if (!strcmp(argv[1], "stream")) return cmd_stream(argc - 2, argv + 2);
  usage();
  return strcmp(argv[1], "--help") ? 2 : 0;
}
