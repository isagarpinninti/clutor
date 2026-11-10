#!/usr/bin/env python3
"""A mixture-of-experts model that does not fit in memory must load by itself (streaming its experts), and say the same thing as when it is mapped; a dense one must refuse, clearly.

Pretends the machine is tiny with CLUTOR_FAKE_RAM_BYTES, runs each tiny test model, and checks:
  * MoE models: the run succeeds, reports streaming, and the greedy text is identical to the mapped run;
  * dense models: the run fails and the message says why;
  * the `chat` protocol's ready event says autoStream is true (and false on a normal machine).

usage: auto_stream.py <engine> <dir with tiny_<arch>.gguf>
"""
import json, os, subprocess, sys

ENGINE, FOLDER = sys.argv[1], sys.argv[2]
MOE = ["olmoe", "qwen3moe", "qwen3vlmoe", "granitemoe"]
DENSE = ["qwen2", "llama"]
failed = 0

def check(ok, what, detail=""):
    global failed
    print(("ok   " if ok else "FAIL ") + what + ("" if ok else f"  {detail}"))
    failed += 0 if ok else 1

def run(args, fake):
    env = dict(os.environ)
    if fake: env["CLUTOR_FAKE_RAM_BYTES"] = str(fake)
    else: env.pop("CLUTOR_FAKE_RAM_BYTES", None)
    return subprocess.run([ENGINE, *args], capture_output=True, env=env, timeout=120)

for arch in MOE:
    model = os.path.join(FOLDER, f"tiny_{arch}.gguf")
    normal = run(["generate", model, "the cat is in the room", "--n", "12"], 0)
    small = run(["generate", model, "the cat is in the room", "--n", "12"], 200_000)
    check(normal.returncode == 0 and small.returncode == 0, f"{arch}: runs mapped and on a tiny machine", small.stderr.decode(errors="replace")[:200])
    check(small.stdout == normal.stdout, f"{arch}: the streamed text equals the mapped text")
    check(b"[streaming:" in small.stderr and b"[streaming:" not in normal.stderr, f"{arch}: streams only when memory is short")
    for fake, want in ((0, "false"), (200_000, "true")):
        env = dict(os.environ)
        if fake: env["CLUTOR_FAKE_RAM_BYTES"] = str(fake)
        else: env.pop("CLUTOR_FAKE_RAM_BYTES", None)
        p = subprocess.Popen([ENGINE, "chat", model, "--ctx", "64"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True)
        ready = json.loads(p.stdout.readline() or "{}")
        p.stdin.write(json.dumps({"op": "quit"}) + "\n"); p.stdin.flush(); p.wait(timeout=30)
        check(str(ready.get("autoStream")).lower() == want, f"{arch}: chat reports autoStream={want}", str(ready)[:200])

for arch in DENSE:
    model = os.path.join(FOLDER, f"tiny_{arch}.gguf")
    r = run(["generate", model, "the cat is in the room", "--n", "4"], 200_000)
    check(r.returncode != 0 and b"dense" in r.stderr, f"{arch}: a dense model that does not fit refuses and says why", r.stderr.decode(errors="replace")[:200])

print("auto-stream-test:", "all passed" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
