#!/usr/bin/env python3
"""Mutation fuzzing of the file readers: corrupt a valid GGUF in many ways and run the engine (built with ASan + UBSan) on it. Any sanitizer report,
crash by signal, or hang is a failure; an ordinary error message and a non-zero exit is the correct answer to a bad file.
Usage: fuzz_mutate.py ENGINE GGUF [--iterations N] [--seed N]"""
import os, random, subprocess, sys, tempfile

engine, src = sys.argv[1], sys.argv[2]
iters = int(sys.argv[sys.argv.index("--iterations") + 1]) if "--iterations" in sys.argv else 300
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
rng = random.Random(seed)
data = open(src, "rb").read()
head = min(len(data), 1 << 16)   # the header and tokenizer tables are where the parsers live; most mutations go there
env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:abort_on_error=0", UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1")

def mutate(b):
    b = bytearray(b)
    kind = rng.randrange(6)
    if kind == 0:   # flip a few bits
        for _ in range(rng.randint(1, 8)): i = rng.randrange(head); b[i] ^= 1 << rng.randrange(8)
    elif kind == 1:   # overwrite a run with random bytes
        i = rng.randrange(head); n = rng.randint(1, 64); b[i:i + n] = bytes(rng.randrange(256) for _ in range(min(n, len(b) - i)))
    elif kind == 2:   # truncate
        b = b[:rng.randrange(len(b))]
    elif kind == 3:   # huge counts / lengths in the first 32 bytes
        i = rng.randrange(8, 32); b[i:i + 8] = (rng.choice([0, 1, 2 ** 31, 2 ** 32, 2 ** 63 - 1, 2 ** 64 - 1])).to_bytes(8, "little")
    elif kind == 4:   # splice a copy of one region over another
        i, j, n = rng.randrange(head), rng.randrange(head), rng.randint(1, 256); b[j:j + n] = b[i:i + n]
    else:   # zero a run
        i = rng.randrange(head); n = rng.randint(1, 128); b[i:i + n] = bytes(min(n, len(b) - i))
    return bytes(b)

cmds = [["info", None], ["tokenize", None, "the cat, 日本語 <|endoftext|> 123"], ["logits", None, "the cat", "--top", "3"]]
bad = 0
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "m.gguf")
    for it in range(iters):
        open(path, "wb").write(mutate(data))
        cmd = cmds[0] if it % 3 else rng.choice(cmds)
        args = [engine] + [path if a is None else a for a in cmd]
        try:
            r = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=60, env=env)
        except subprocess.TimeoutExpired:
            print("HANG   iteration %d: %s" % (it, cmd[0])); bad += 1; open("fuzz_fail_%d.gguf" % it, "wb").write(open(path, "rb").read()); continue
        out = r.stderr + r.stdout
        if r.returncode < 0 or "Sanitizer" in out or "runtime error" in out:
            print("CRASH  iteration %d: %s (exit %d)\n%s" % (it, cmd[0], r.returncode, out[:600])); bad += 1
            open("fuzz_fail_%d.gguf" % it, "wb").write(open(path, "rb").read())
print("%d iterations, %d failures (seed %d)" % (iters, bad, seed))
sys.exit(1 if bad else 0)
