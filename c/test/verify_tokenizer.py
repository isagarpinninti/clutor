#!/usr/bin/env python3
"""Compares the engine's tokenizer with llama.cpp's on a fixed set of strings, using a model file (or a vocabulary-only copy of one).
Usage: verify_tokenizer.py MODEL.gguf [--engine build/clutor-engine]    (needs llama-tokenize on PATH)"""
import json, os, subprocess, sys

model = sys.argv[1]
engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else os.path.join(os.path.dirname(__file__), "..", "build", "clutor-engine")
CASES = [
    "Hello, world! It's 2024.", "The quick brown fox jumps over the lazy dog.", "  leading spaces and    multiple   spaces   ",
    "line one\nline two\n\nline four\ttab", "def f(x):\n    return x**2 + 3.14159  # comment", "don't stop, can't stop; won't stop... \"quoted\" (parens) [brackets] {braces}",
    "12345 6789012 3,000,000 $19.99 100% a@b.com http://x.org/a?b=c&d=e", "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "café naïve Zürich 日本語 привет 🙂", "I'M SHOUTING, YOU'RE LISTENING, WE'LL SEE; THEY'VE GONE.", "trailing newlines\n\n\n", "\n\n  indented after blank lines",
    "x = [1, 2, 3]\nfor i in range(10):\n\tprint(i)\n", "tabs\t\tand   spaces \n and more", "end with space ", "a\r\nb\r\n\r\nc", "1+1=2; 2*2=4; 10/5=2",
    "snake_case and camelCase and kebab-case and PascalCase", "<think>\nlet me see\n</think>\n", "!!! ??? ... --- ___ ===", "price: €5, £6, ¥7", "ﬁne ﬂow ½ ² ³", "   ", "word",
]
bad = 0
for text in CASES:
    mine = [int(x) for x in subprocess.run([engine, "tokenize", model, text], capture_output=True, text=True).stdout.split()]
    out = subprocess.run(["llama-tokenize", "-m", model, "-p", text, "--ids", "--no-bos"], capture_output=True, text=True).stdout.strip().splitlines()
    ref = json.loads(out[-1]) if out else []
    ok = mine == ref
    bad += not ok
    print("OK  " if ok else "FAIL", repr(text[:44]))
    if not ok:
        print("   engine", mine[:24]); print("   llama ", ref[:24])
print("mismatches:", bad, "of", len(CASES))
sys.exit(1 if bad else 0)
