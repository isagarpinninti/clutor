#!/usr/bin/env python3
"""Compares the engine's real forward pass with llama.cpp on the same model: tokenization, then the next-token distribution.

Needs `llama-server` and `llama-tokenize` on PATH. Usage: verify_llama.py MODEL.gguf [--engine build/clutor-engine] [--engine-args "--cache-mb 4096"]
(--engine-args is required for a model larger than memory: the engine refuses to map it. llama.cpp itself maps the file, so on such a model it is slow.)
llama.cpp runs with a Q8_K-quantized activation path and Metal/NEON kernels; this engine uses float32 activations, so the two agree closely but
not bit for bit. What is checked: the same token ids, the same top-1 token, top-5 sets that overlap, and logprobs within a stated tolerance.
"""
import json, subprocess, sys, time, urllib.request, os, signal

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Water boils at 100 degrees Celsius at sea level. At the top of Mount Everest it boils at",
    "Once upon a time, in a small village by the sea, there lived",
    "1 + 1 = 2, 2 + 2 = 4, 4 + 4 =",
    "<|user|>\nName three primary colors.\n<|assistant|>\n",
]
TOL_TOP1 = 0.35      # |delta logprob| allowed on the top-1 token
TOL_MEAN = 0.35      # mean |delta| over the tokens both list in their top 20

def main():
    model = sys.argv[1]
    engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else os.path.join(os.path.dirname(__file__), "..", "build", "clutor-engine")
    extra = sys.argv[sys.argv.index("--engine-args") + 1].split() if "--engine-args" in sys.argv else []
    port = 8099
    srv = subprocess.Popen(["llama-server", "-m", model, "-c", "1024", "--port", str(port), "-ngl", "0", "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(900):
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health").read(): break
            except Exception: time.sleep(1)
        else:
            print("llama-server did not start"); return 2
        failed = 0
        for p in PROMPTS:
            ids = subprocess.run(["llama-tokenize", "-m", model, "-p", p, "--ids"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
            mine = json.loads(subprocess.run([engine, "logits", model, p, "--top", "20", *extra], capture_output=True, text=True, check=True).stdout)
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps({"prompt": p, "n_predict": 1, "n_probs": 20, "temperature": 0, "samplers": []}).encode(), headers={"Content-Type": "application/json"})
            ref = json.load(urllib.request.urlopen(req))["completion_probabilities"][0]["top_logprobs"]
            r = {t["id"]: t["logprob"] for t in ref}
            m = {t["id"]: t["logprob"] for t in mine["top"]}
            same_tokens = json.loads(ids) == mine["tokens"]
            top1 = mine["top"][0]["id"] == ref[0]["id"]
            both = [i for i in m if i in r]
            deltas = [abs(m[i] - r[i]) for i in both]
            top5 = len(set(list(m)[:5]) & set(list(r)[:5]))
            d1 = abs(m[mine["top"][0]["id"]] - r[ref[0]["id"]]) if top1 else 99
            ok = same_tokens and top1 and top5 >= 4 and d1 <= TOL_TOP1 and sum(deltas) / max(len(deltas), 1) <= TOL_MEAN
            failed += not ok
            print(f"{'OK  ' if ok else 'FAIL'} tokens_equal={same_tokens} top1={top1} top5_overlap={top5}/5 shared={len(both)}/20 mean|d|={sum(deltas)/max(len(deltas),1):.3f} max|d|={max(deltas, default=0):.3f}  {p[:36]!r}")
        print("failed:", failed)
        return 1 if failed else 0
    finally:
        srv.send_signal(signal.SIGTERM)

if __name__ == "__main__":
    sys.exit(main())
