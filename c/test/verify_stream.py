#!/usr/bin/env python3
"""Streaming the experts from disk must not change a single output token: greedy text from the mapped file and from several cache sizes,
policies and prefetch settings must be byte-identical. Usage: verify_stream.py MODEL.gguf [--engine build/clutor-engine]"""
import subprocess, sys, os

model = sys.argv[1]
engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else os.path.join(os.path.dirname(__file__), "..", "build", "clutor-engine")
prompt, n = "The three primary colors are", "16"

def text(*extra):
    r = subprocess.run([engine, "generate", model, prompt, "--n", n, *extra], capture_output=True, text=True, check=True)
    return r.stdout

base = text()
bad = 0
for extra in (["--cache-mb", "32"], ["--cache-mb", "64", "--policy", "lru"], ["--cache-mb", "128", "--prefetch", "on"], ["--cache-mb", "128", "--prefetch", "on", "--prefetch-n", "2"], ["--cache-mb", "256", "--io-threads", "1", "--no-os-cache"], ["--cache-mb", "1024"]):
    same = text(*extra) == base
    bad += not same
    print("OK  " if same else "FAIL", "identical to the mapped run:", " ".join(extra))
print("failed:", bad)
sys.exit(1 if bad else 0)
