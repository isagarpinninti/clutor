#!/usr/bin/env python3
"""The batched prompt path must give the same logits as stepping token by token, on every architecture.

Runs `clutor-engine prefillcheck` on each tiny test model with the batch cut at 64, 7 and 1 tokens (a short prompt then crosses many batch boundaries), with the weights
mapped and streamed from disk, with and without SMART-ROUTE, and fails if any combination disagrees or the top token differs.

usage: prefill_check.py <engine> <dir with tiny_<arch>.gguf> <prompt>
"""
import json
import os
import subprocess
import sys

ARCHS = ["olmoe", "qwen3moe", "granitemoe", "qwen2", "llama", "qwen35"]
BATCHES = ["64", "7", "1"]
LONG_BATCHES = ["256", "100"]   # a long prompt: a streamed run takes batches up to 256 tokens, so these cross the 64-token size the short prompt cannot reach
LONG = " ".join(["the cat is in the room and a rat saw the hat on a mat"] * 14)
MODES = [[], ["--cache-mb", "1"], ["--cache-mb", "1", "--smart", "2,12,0.015"]]


def main() -> int:
    engine, folder, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    bad, n = [], 0
    for arch in ARCHS:
        model = os.path.join(folder, f"tiny_{arch}.gguf")
        if not os.path.exists(model):
            print(f"missing {model}")
            return 2
        for text, batch in [(prompt, b) for b in BATCHES] + [(LONG, b) for b in LONG_BATCHES]:
            for mode in MODES:
                env = dict(os.environ, CLUTOR_PS_B=batch)
                r = subprocess.run([engine, "prefillcheck", model, text, "--repeat", "1" if text is LONG else "2", *mode], capture_output=True, text=True, env=env)
                if "this model is dense" in r.stderr:   # a dense model has no experts to stream or substitute: the engine refuses these flags, by design
                    continue
                n += 1
                last = (r.stdout.strip().splitlines() or [""])[-1]
                try:
                    ok = json.loads(last).get("ok") is True
                except json.JSONDecodeError:
                    ok = False
                # A sanitizer finding (a data race or a memory error) shows as a non-zero exit even when the printed result is fine.
                if r.returncode != 0:
                    ok = False
                    last = f"{last} [exit code {r.returncode}] {r.stderr.strip().splitlines()[0] if r.stderr.strip() else ''}"
                if not ok:
                    bad.append(f"{arch} batch={batch} {' '.join(mode)}: {last or r.stderr.strip()}")
    if bad:
        print("\n".join("FAIL " + b for b in bad))
        return 1
    print(f"prefill-test: {n} combinations agree with token-by-token stepping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
