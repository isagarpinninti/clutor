#!/usr/bin/env python3
"""Clutor against Colibri's OLMoE engine, same model weights (OLMoE-1B-7B-0924-Instruct Q4_K_M; Colibri's copy is that GGUF converted by Colibri's own
converter, experts re-quantized to int8), same 20-token prompt, 80 greedy tokens, one machine, runs one after another. Memory-bounded modes read from
fresh copies of the files that nothing has mapped or read through the OS cache."""
import json, os, re, subprocess, sys, time
# Paths come from the environment so the script runs anywhere; the defaults assume this layout next to the repo:
#   $BENCH_DIR/colibri (a checkout with `make olmoe` built), $BENCH_DIR/olmoe-colibri-cold (Colibri's converted copy), $BENCH_DIR/ref_story.json
#   $CLUTOR_GGUF_COLD: a fresh copy of the GGUF that nothing has mapped (copy it with `fastcopy`, or any tool that bypasses the OS cache)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
B = os.environ.get("BENCH_DIR", os.path.join(REPO, "..", "bench"))
CL = os.environ.get("CLUTOR_ENGINE", os.path.join(REPO, "c", "build", "clutor-engine"))
GGUF = os.environ.get("CLUTOR_GGUF_COLD", os.path.join(REPO, "..", "clutor-models", "olmoe-cold.gguf"))
COLI = os.path.join(B, "colibri", "c", "olmoe")
SNAP = os.path.join(B, "olmoe-colibri-cold")
PROMPT = "Once upon a time, in a small village by the sea, there lived an old fisherman who"
rows = []

def clutor(label, *extra):
    t = time.time()
    r = subprocess.run([CL, "generate", GGUF, PROMPT, "--n", "80", "--quiet", *extra], capture_output=True, text=True)
    wall = time.time() - t
    e = r.stderr
    m = re.search(r"in ([\d.]+) s = ([\d.]+) s/token", e)
    rss = re.search(r"peak RSS (\d+) MiB", e)
    gb = re.search(r"\(([\d.]+) GB/token", e)
    hit = re.search(r"([\d.]+)% hits", e)
    rows.append({"engine": "Clutor", "mode": label, "job_seconds": float(m.group(1)) if m else None, "peak_rss_mib": int(rss.group(1)) if rss else None,
                 "gb_per_token": float(gb.group(1)) if gb else None, "hit_rate": float(hit.group(1)) if hit else None, "wall": round(wall, 1)})
    print(rows[-1], flush=True)

def colibri(label, cap, env):
    e = dict(os.environ, SNAP=SNAP, **env)
    t = time.time()
    r = subprocess.run([COLI, str(cap), "8", os.path.join(B, "ref_story.json")], capture_output=True, text=True, env=e)
    wall = time.time() - t
    o = r.stdout
    sp = re.search(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)", o)
    rss = re.search(r"PEAK RSS: ([\d.]+) GB", o)
    hit = re.search(r"hit rate: ([\d.]+)%", o)
    rows.append({"engine": "Colibri", "mode": label, "job_seconds": float(sp.group(2)) if sp else None, "peak_rss_mib": int(float(rss.group(1)) * 1024) if rss else None,
                 "gb_per_token": None, "hit_rate": float(hit.group(1)) if hit else None, "wall": round(wall, 1)})
    print(rows[-1], flush=True)

# memory-bounded, cold files (nothing here maps the files before they are read with the OS cache bypassed)
for mb in (128, 256, 512, 1024):
    clutor(f"stream {mb} MiB cache", "--cache-mb", str(mb), "--no-os-cache")
for cap in (4, 8, 16, 32):
    colibri(f"cache {cap}/layer, EXPERT_DROP=1", cap, {"EXPERT_DROP": "1", "PILOT": "0"})
colibri("cache 16/layer, EXPERT_DROP=1, PILOT=1 (its prefetch)", 16, {"EXPERT_DROP": "1", "PILOT": "1"})
json.dump(rows, open(os.path.join(B, "bench_results.json"), "w"), indent=1)
print("done", flush=True)
