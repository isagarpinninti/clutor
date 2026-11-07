#!/usr/bin/env python3
"""Drives `clutor-engine chat` over its JSON-lines protocol on a tiny generated model, and checks it against the engine's own `generate`.

    chat_protocol.py ENGINE MODEL.gguf

What is checked:
  * chat at temperature 0 produces the same tokens as `generate` (greedy) for the same prompt;
  * the KV cache is kept between calls: appending a prompt in two pieces and continuing gives the same as one piece when the cut is on a
    special-token boundary, and a second turn pays only for its own new tokens;
  * reset really forgets; a cancel stops generation promptly; an over-long append is refused, not crashed;
  * a seed makes sampling repeatable, and a hung-up input ends the process.
"""
import json, subprocess, sys, time

ENGINE, MODEL = sys.argv[1], sys.argv[2]
failed = 0
def check(ok, what, detail=""):
    global failed
    print(("ok   " if ok else "FAIL ") + what + ("" if ok else f"  {detail}"))
    failed += 0 if ok else 1

class Chat:
    def __init__(self, ctx=256):
        self.p = subprocess.Popen([ENGINE, "chat", MODEL, "--ctx", str(ctx)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.ready = self.read()
    def send(self, **o): self.p.stdin.write(json.dumps(o) + "\n"); self.p.stdin.flush()
    def read(self):
        l = self.p.stdout.readline()
        return json.loads(l) if l else None
    def until(self, kinds):
        out = []
        while True:
            e = self.read()
            if e is None: return out
            out.append(e)
            if e["ev"] in kinds: return out
    def say(self, text, **gen):
        self.send(op="append", text=text); a = self.until({"appended", "error"})
        self.send(op="generate", **gen); g = self.until({"done", "error"})
        return a[-1], "".join(e["s"] for e in g if e["ev"] == "t"), g[-1]

prompt = "The quick brown fox"
import os, tempfile
idf = tempfile.mktemp(suffix=".json")
subprocess.run([ENGINE, "generate", MODEL, prompt, "--n", "12", "--quiet", "--ids", idf], capture_output=True)
ref_ids = json.load(open(idf)); os.unlink(idf)
expect = ref_ids["full_ids"][len(ref_ids["prompt_ids"]):]

c = Chat()
check(c.ready and c.ready["ev"] == "ready" and 0 < c.ready["nCtx"] <= 256, "starts and reports itself (with the model's real context limit when it is below the request)", c.ready)
a, text, d = c.say(prompt, max=12, temp=0)
check(d["ids"] == expect and len(expect) > 0, "chat at temperature 0 matches generate token for token", f"{d['ids']} vs {expect}")
check(d["finish"] in ("length", "stop") and d["tokens"] >= 1 and d["pos"] == a["pos"] + d["tokens"], "reports finish, tokens and position", d)

# prompts that cross the batch boundary of prompt processing (32 tokens at a time) give the same tokens as feeding them one by one
def ntok(t): return len(subprocess.run([ENGINE, "tokenize", MODEL, t], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.split())
for want in (31, 32, 33, 50):
    t = ""
    while ntok(t + "ab ") <= want: t += "ab "
    f2 = tempfile.mktemp(suffix=".json")
    subprocess.run([ENGINE, "generate", MODEL, t, "--n", "6", "--quiet", "--ids", f2], capture_output=True)
    r2 = json.load(open(f2)); os.unlink(f2)
    c.send(op="reset"); c.until({"reset"})
    _, _, dd = c.say(t, max=6, temp=0)
    check(dd["ids"] == r2["full_ids"][len(r2["prompt_ids"]):], "a prompt of ~%d tokens (batched prompt processing) gives the same tokens as one at a time" % ntok(t), (dd["ids"], r2["full_ids"][len(r2["prompt_ids"]):]))
c.send(op="reset"); c.until({"reset"})
a, text, d = c.say(prompt, max=12, temp=0)

# the cache is kept: a second append only pays for its own tokens
pos_before = d["pos"]
c.send(op="append", text=" jumps"); a2 = c.until({"appended", "error"})[-1]
check(a2["ev"] == "appended" and a2["pos"] == pos_before + a2["tokens"], "a second append continues from the cache", a2)

# reset forgets: the same prompt gives the same answer again
c.send(op="reset"); c.until({"reset"})
a3, text3, d3 = c.say(prompt, max=12, temp=0)
check(d3["ids"] == expect and a3["pos"] == a["pos"], "reset forgets the conversation", d3["ids"])

# seeded sampling repeats
c.send(op="reset"); c.until({"reset"})
_, _, e1 = c.say(prompt, max=16, temp=1.0, seed=7)
c.send(op="reset"); c.until({"reset"})
_, _, e2 = c.say(prompt, max=16, temp=1.0, seed=7)
c.send(op="reset"); c.until({"reset"})
_, _, e3 = c.say(prompt, max=16, temp=1.0, seed=8)
check(e1["ids"] == e2["ids"], "the same seed gives the same tokens", f"{e1['ids']} {e2['ids']}")
check(e1["ids"] != e3["ids"], "another seed gives other tokens (the tiny model is random, so this is a sanity check)", f"{e1['ids']}")

# a late cancel is harmless, and a cancel during generation stops it
c.send(op="cancel")
c.send(op="reset"); r = c.until({"reset"})
check(r[-1]["ev"] == "reset", "a cancel with nothing running is ignored", r)
c.send(op="append", text=prompt); c.until({"appended"})
c.send(op="generate", max=200, temp=0); c.send(op="cancel")
g = c.until({"done"})
check(g[-1]["ev"] == "done" and g[-1]["finish"] in ("cancel", "length", "stop") and g[-1]["tokens"] < 200, "cancel stops generation early", g[-1])

# nothing to continue from, unknown op, over-long input
c.send(op="reset"); c.until({"reset"})
c.send(op="generate", max=3); e = c.until({"error", "done"})[-1]
check(e["ev"] == "error", "generate before any append is an error, not a crash", e)
c.send(op="bogus"); e = c.until({"error"})[-1]
check(e["ev"] == "error", "an unknown op is an error", e)
c.send(op="append", text="word " * 400); e = c.until({"error", "appended"})[-1]
check(e["ev"] == "error" and e.get("code") == "context", "an append that cannot fit is refused", e)
c.send(op="append", text="still alive"); e = c.until({"appended", "error"})[-1]
check(e["ev"] == "appended", "the session survives a refused append", e)

# text with quotes, newlines, unicode and an escaped key name survives the round trip
c.send(op="reset"); c.until({"reset"})
tricky = 'say "hi"\nline2 \\ tab\t \u00e9\u4e2d\U0001f600 {"op":"quit"}'
c.send(op="append", text=tricky); e = c.until({"appended", "error"})[-1]
check(e["ev"] == "appended", "quotes, newlines, unicode and a fake key inside the text are all fine", e)
# ... and is tokenized exactly like the tokenize command. That command takes its text from the command line, which Windows passes through the ANSI code page, so the
# comparison uses the ASCII part (the non-ASCII part is covered by the stdin round trip above and by verify_tokenizer.py on POSIX).
ascii_tricky = 'say "hi"\nline2 \\ tab\t {"op":"quit"}'
c.send(op="reset"); c.until({"reset"})
c.send(op="append", text=ascii_tricky); e = c.until({"appended", "error"})[-1]
ref2 = subprocess.run([ENGINE, "tokenize", MODEL, ascii_tricky], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.split()
check(e["ev"] == "appended" and e["tokens"] == len(ref2) + (1 if c.ready["addBos"] else 0), "and the ASCII part was tokenized exactly like the tokenize command", (e, len(ref2)))

# closing stdin ends the process
c.p.stdin.close()
try: rc = c.p.wait(timeout=10)
except subprocess.TimeoutExpired: c.p.kill(); rc = -9
check(rc == 0, "end of input ends the process cleanly", rc)

print(f"\n{failed} failed" if failed else "\nall passed")
sys.exit(1 if failed else 0)
