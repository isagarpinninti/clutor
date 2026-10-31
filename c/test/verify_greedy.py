#!/usr/bin/env python3
"""Greedy continuations from llama.cpp and from this engine, compared token by token. Logprobs of two engines never agree exactly (8-bit activations,
accumulation order, f16 against f32 caches), and a deep model can amplify that; identical text is the test that matters in practice.
Usage: verify_greedy.py MODEL.gguf [--engine build/clutor-engine] [--engine-args "--cache-mb 4096"] [--n 24]   (needs llama-server on PATH)"""
import json, os, signal, subprocess, sys, tempfile, time, urllib.request

model = sys.argv[1]
engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else os.path.join(os.path.dirname(__file__), "..", "build", "clutor-engine")
extra = sys.argv[sys.argv.index("--engine-args") + 1].split() if "--engine-args" in sys.argv else []
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 24
PROMPTS = ["The capital of France is", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", "Once upon a time, in a small village by the sea, there lived an old fisherman who"]

srv = subprocess.Popen(["llama-server", "-m", model, "-c", "1024", "--port", "8098", "-ngl", "0", "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
bad = 0
try:
    for _ in range(900):
        try:
            if b"ok" in urllib.request.urlopen("http://127.0.0.1:8098/health").read(): break
        except Exception: time.sleep(1)
    for p in PROMPTS:
        req = urllib.request.Request("http://127.0.0.1:8098/completion", data=json.dumps({"prompt": p, "n_predict": N, "temperature": 0, "return_tokens": True, "cache_prompt": False}).encode(), headers={"Content-Type": "application/json"})
        ref = json.load(urllib.request.urlopen(req))["tokens"]
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            subprocess.run([engine, "generate", model, p, "--n", str(N), "--quiet", "--ids", f.name, *extra], capture_output=True, check=True)
            d = json.load(open(f.name)); mine = d["full_ids"][len(d["prompt_ids"]):]
        k = next((i for i in range(min(len(ref), len(mine))) if ref[i] != mine[i]), None)
        same = k is None and len(ref) == len(mine)
        bad += not same
        print("OK  " if same else "DIFF", "%d tokens identical" % len(ref) if same else "first difference at generated token %d of %d (llama %s, engine %s)" % (k if k is not None else min(len(ref), len(mine)), min(len(ref), len(mine)), ref[k] if k is not None else "-", mine[k] if k is not None else "-"), repr(p[:36]))
finally:
    srv.send_signal(signal.SIGTERM)
print("prompts with a difference:", bad)
sys.exit(1 if bad else 0)
