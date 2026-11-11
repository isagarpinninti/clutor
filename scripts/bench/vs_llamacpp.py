#!/usr/bin/env python3
"""Clutor's engine against llama.cpp on the same GGUF file, same machine, runs one after another.

  python3 scripts/bench/vs_llamacpp.py MODEL.gguf [--prompt-tokens 128] [--gen 64] [--runs 3] [--ctx 1024] [--json out.json]

What it measures, for each engine and mode:
  start_s     seconds from launching the process until it can take a prompt (Clutor: the `ready` event; llama.cpp: not reported by llama-bench, so left empty)
  prefill_tps prompt tokens per second
  decode_tps  generated tokens per second (greedy)
  first_token_s  prompt of the given length until the first generated piece (Clutor only: llama-bench has no such number)
  peak_rss_mib   peak resident memory of the process (ru_maxrss of the child)

Modes: `clutor` (the engine, CPU only), `llama.cpp cpu` (-ngl 0, same kind of hardware use) and `llama.cpp gpu` (Metal, the default of llama-bench).
Compare Clutor with the CPU row; the GPU row is what llama.cpp does when it is allowed to use the graphics chip, which the Clutor engine does not.
Each number is the mean of --runs runs, with the lowest and highest next to it. Nothing here is a general claim: it is this machine, this file, this moment.
Needs: the engine (c/build/clutor-engine) and llama-bench on PATH (or LLAMA_BENCH=/path)."""
import argparse, json, os, platform, resource, shutil, subprocess, sys, time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENGINE = os.environ.get("CLUTOR_ENGINE", os.path.join(REPO, "c", "build", "clutor-engine"))
LLAMA_BENCH = os.environ.get("LLAMA_BENCH") or shutil.which("llama-bench")

# a fixed paragraph, repeated until it is about the requested number of tokens
PARA = ("The harbour town woke slowly. Fishermen carried nets down to the boats while gulls argued over scraps on the quay, "
        "and the baker propped her door open so the smell of warm bread drifted along the lane. ")


def mean_range(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(sum(xs) / len(xs), 2), "min": round(min(xs), 2), "max": round(max(xs), 2), "runs": len(xs)}


def rss_mib():
    r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def clutor_once(model, prompt, gen, ctx):
    """One fresh process: start, feed the prompt, decode `gen` tokens greedily."""
    t0 = time.time()
    p = subprocess.Popen([ENGINE, "chat", model, "--ctx", str(ctx)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out = {"start_s": None, "prefill_tps": None, "decode_tps": None, "first_token_s": None, "prompt_tokens": None}
    def read_until(kinds):
        while True:
            line = p.stdout.readline()
            if not line:
                return None
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("ev") in kinds:
                return ev
    if read_until({"ready"}) is None:
        return out
    out["start_s"] = time.time() - t0
    p.stdin.write(json.dumps({"op": "append", "text": prompt, "raw": True}) + "\n"); p.stdin.flush()
    t1 = time.time()
    ap = read_until({"appended"})
    if ap is None:
        return out
    out["prompt_tokens"] = ap.get("tokens")
    out["prefill_tps"] = ap["tokens"] / ap["seconds"] if ap.get("seconds") else None
    p.stdin.write(json.dumps({"op": "generate", "max": gen, "temp": 0}) + "\n"); p.stdin.flush()
    tg = time.time()
    first = None
    while True:
        line = p.stdout.readline()
        if not line:
            break
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("ev") == "t" and first is None:
            first = time.time()
        if ev.get("ev") == "done":
            out["decode_tps"] = ev.get("tokensPerSecond")
            break
    if first is not None:
        out["first_token_s"] = (first - tg) + (ap.get("seconds") or 0)
    p.stdin.close(); p.wait(timeout=30)
    return out


def llama_bench(model, ngl, ptok, gen, runs):
    cmd = [LLAMA_BENCH, "-m", model, "-p", str(ptok), "-n", str(gen), "-r", str(runs), "-ngl", str(ngl), "-o", "json"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        rows = json.loads(r.stdout[r.stdout.index("["):])
    except ValueError:
        return {"error": (r.stderr or r.stdout)[-300:]}
    res = {"threads": rows[0].get("n_threads")}
    for row in rows:
        if row.get("n_gen", 0) == 0:
            res["prefill_tps"] = {"mean": round(row["avg_ts"], 2), "runs": runs, "stddev": round(row.get("stddev_ts", 0), 2)}
        elif row.get("n_prompt", 0) == 0:
            res["decode_tps"] = {"mean": round(row["avg_ts"], 2), "runs": runs, "stddev": round(row.get("stddev_ts", 0), 2)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--prompt-tokens", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3); ap.add_argument("--ctx", type=int, default=1024); ap.add_argument("--json")
    a = ap.parse_args()
    if not os.path.exists(ENGINE):
        sys.exit(f"engine not found: {ENGINE} (build it: make -C c)")
    prompt = (PARA * 40)
    # find a prompt length close to the request: ask the engine how many tokens a paragraph is, then cut
    runs = [clutor_once(a.model, prompt[: a.prompt_tokens * 4], a.gen, a.ctx) for _ in range(a.runs)]
    ptok = runs[0]["prompt_tokens"] or a.prompt_tokens
    result = {
        "machine": {"platform": platform.platform(), "cpu": platform.processor() or platform.machine()},
        "model": os.path.basename(a.model), "model_bytes": os.path.getsize(a.model), "prompt_tokens": ptok, "gen_tokens": a.gen,
        "clutor": {k: mean_range([r[k] for r in runs]) for k in ("start_s", "prefill_tps", "decode_tps", "first_token_s")},
        "clutor_peak_rss_mib": round(rss_mib(), 0),
    }
    if LLAMA_BENCH:
        result["llama.cpp cpu"] = llama_bench(a.model, 0, ptok, a.gen, a.runs)
        result["llama.cpp gpu"] = llama_bench(a.model, 99, ptok, a.gen, a.runs)
    else:
        result["llama.cpp"] = "llama-bench not found; install llama.cpp or set LLAMA_BENCH"
    print(json.dumps(result, indent=2))
    if a.json:
        open(a.json, "w").write(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
