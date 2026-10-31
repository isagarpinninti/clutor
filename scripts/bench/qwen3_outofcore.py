#!/usr/bin/env python3
"""Qwen3-30B-A3B Q4_K_M (17.3 GiB) on a machine with less RAM than that: this engine streaming experts through a cache against llama.cpp memory-mapping the file.

Workload: a one-token prompt ("Hello") and 48 greedy tokens, so the run is decode, not a batched prefill. Three runs per setting.
  phase 1  this engine, F_NOCACHE reads (every byte from the disk), expert caches of 2, 4 and 8 GiB   (handicapped: llama.cpp below may use the OS cache)
  phase 2  llama.cpp: a fresh llama-server per run, default settings (mmap), -c 512, CPU only; decode speed from its own timings, RSS sampled
  phase 3  this engine with the OS cache allowed, 4 GiB cache (NOT comparable with phase 1: llama.cpp's runs have just warmed the OS cache for it)
Environment: QWEN_GGUF = a fresh copy of the model that nothing has read yet (write it with `fastcopy`, or any copy that bypasses the OS cache),
CLUTOR_ENGINE = path of clutor-engine (default c/build/clutor-engine), BENCH_OUT = where to write the JSON (default ./qwen3_outofcore_results.json)."""
import json, os, re, signal, statistics, subprocess, sys, threading, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GGUF = os.environ["QWEN_GGUF"]
CL = os.environ.get("CLUTOR_ENGINE", os.path.join(REPO, "c", "build", "clutor-engine"))
OUT = os.environ.get("BENCH_OUT", "qwen3_outofcore_results.json")
PROMPT, N, RUNS, COOL = "Hello", 48, 3, 20
rows = []

def clutor(label, cache_mb, nocache):
    args = [CL, "generate", GGUF, PROMPT, "--n", str(N), "--quiet", "--cache-mb", str(cache_mb)] + (["--no-os-cache"] if nocache else [])
    t = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    e = r.stderr
    m = re.search(r"in ([\d.]+) s = ([\d.]+) s/token", e)
    row = {"engine": "clutor", "setting": label, "ok": r.returncode == 0 and m is not None, "wall_s": round(time.time() - t, 1)}
    if m:
        row.update(sec_per_token=float(m.group(2)), tokens_per_s=round(1 / float(m.group(2)), 3),
                   hit_rate=float(re.search(r"([\d.]+)% hits", e).group(1)), gb_per_token=float(re.search(r"\(([\d.]+) GB/token", e).group(1)),
                   peak_rss_mib=int(re.search(r"peak RSS (\d+) MiB", e).group(1)))
    else:
        row["error"] = e[-300:]
    rows.append(row); print(json.dumps(row), flush=True)

def llama(label):
    port = 8097
    srv = subprocess.Popen(["llama-server", "-m", GGUF, "-c", "512", "-ngl", "0", "--no-webui", "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = [0]; stop = threading.Event()
    def sample():
        while not stop.is_set():
            o = subprocess.run(["ps", "-o", "rss=", "-p", str(srv.pid)], capture_output=True, text=True).stdout.strip()
            if o: peak[0] = max(peak[0], int(o))
            time.sleep(0.5)
    th = threading.Thread(target=sample, daemon=True); th.start()
    t0 = time.time(); row = {"engine": "llama.cpp", "setting": label, "ok": False}
    try:
        for _ in range(3600):
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read(): break
            except Exception: time.sleep(1)
        row["time_to_ready_s"] = round(time.time() - t0, 1)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps({"prompt": PROMPT, "n_predict": N, "temperature": 0, "cache_prompt": False}).encode(), headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        tm = d["timings"]
        row.update(ok=True, tokens_per_s=round(tm["predicted_per_second"], 3), sec_per_token=round(tm["predicted_ms"] / tm["predicted_n"] / 1000, 3), prompt_ms=round(tm["prompt_ms"], 1), predicted_n=tm["predicted_n"])
    except Exception as ex:
        row["error"] = str(ex)[:300]
    finally:
        stop.set(); th.join(); row["peak_rss_mib"] = peak[0] // 1024
        srv.send_signal(signal.SIGTERM)
        try: srv.wait(timeout=60)
        except subprocess.TimeoutExpired: srv.kill()
    rows.append(row); print(json.dumps(row), flush=True)

def cool(): time.sleep(COOL)

for mb in (2048, 4096, 8192):
    for i in range(RUNS):
        clutor(f"phase1 cache {mb} MiB, F_NOCACHE, run {i + 1}", mb, True); cool()
for i in range(RUNS):
    llama(f"phase2 mmap defaults, run {i + 1}"); cool()
for i in range(RUNS):
    clutor(f"phase3 cache 4096 MiB, OS cache allowed (biased: warmed by phase 2), run {i + 1}", 4096, False); cool()
json.dump(rows, open(OUT, "w"), indent=1)
print("DONE", flush=True)
