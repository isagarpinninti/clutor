# Clutor

**A terminal tool for running big mixture-of-experts models from disk on small machines, with a coding agent that cannot touch your project until you say so.**

Clutor has three parts, and the line between them is the point of this README:

1. **A native engine** (`c/`, plain C, no dependencies) that reads a GGUF file directly and streams its routed experts from disk through an adaptive cache with next-layer prefetch and probability-bounded cache-aware routing. It **measures** disk reads, waits, hit rates and prefetch behaviour. It does **not** yet run a transformer, so it does not generate text.
2. **An honest planner and CLI** (`packages/core`, `packages/cli`, TypeScript): will this model fit, where does every byte go, how fast could the disk possibly let it go. It drives `llama.cpp` to actually generate text, and serves OpenAI- and Ollama-compatible APIs.
3. **A sandboxed coding agent**: it works in a held-aside copy, asks before running project code, shows an exact diff, and signs a receipt of what it did.

[Quickstart](#quickstart) · [Commands](#commands) · [What is measured, what is not](#what-is-measured-and-what-is-not) · [Docs](#documentation)

## Quickstart

Needs Node.js 20+ and a C compiler (for the engine; `xcode-select --install` on macOS). llama.cpp (`brew install llama.cpp`) is needed to chat or run the agent.

```bash
git clone https://github.com/isagarpinninti/clutor.git && cd clutor
npm install && npm run build
npm run engine                              # compiles c/ into c/build/clutor-engine
alias clutor="node $PWD/packages/cli/dist/bin.js"     # `clt` is the short name once installed with npm

clutor init
clutor plan deepseek-v3 --ram 16            # will it fit; the fastest the disk allows
clutor pull olmoe-1b-7b-q4_k_m              # a 3.9 GiB real MoE model, asks first
clutor stream models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf --tokens 300 --cache-mb 512
clutor bench --file models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf     # compares cache strategies
```

Installed from npm instead (`npm install -g @clutor/cli`)? The package carries the engine's C source: run `clutor engine build` once (needs a C compiler) and it is compiled into `~/.clutor/bin/clutor-engine`, where `stream`, `bench` and `/cache` find it. `plan`, `chat`, the agent and the server need no compiler.

To chat or use the agent, set `backend.kind` to `llama-server` and `backend.llamaServer.model` to a `.gguf` in `clutor.config.json` (see `examples/clutor.config.json`), then:

```bash
clutor                                       # interactive session: type a task, or /help (/plan /status /verify /undo /tools /cost ...)
clutor chat                                  # plain chat with the model: /help /layers /cache /reset /exit
clutor decide --prompt "PR #42 changes auth with no tests" --choices "approve,request_changes,block"
clutor code "add a token-bucket rate limiter with tests" --dir ./myproject --verify "npm test"
clutor serve                                 # OpenAI + Ollama API on 127.0.0.1:4320
```

## Commands

| Command | What it does |
| --- | --- |
| `plan <model>` | Memory plan from a catalog id or a real `.gguf` header: fits or not, where the memory goes, disk-bound speed ceiling at several hit rates |
| `simulate <model>` | Replays a synthetic routing trace through LRU and Clutor's cache (simulated, labelled) |
| `stream <file.gguf>` | **Native engine.** Streams the file's experts from disk for N tokens of synthetic routing; reports measured I/O, waits, cache, prefetch and peak RSS. `--shift AT,LEN` makes those tokens uniform scan noise; `--window N` prints hit rate and peak RSS per N tokens |
| `bench --file <f>` | Measures random expert reads, then compares LRU, adaptive cache + prefetch, and + smart-route on that file. `--receipt` signs the result into the receipt log |
| `engine build\|where` | Compile the engine, or show which binary is used |
| `chat` | REPL against the configured model. `/layers` shows where the file's bytes are; `/cache` runs a streaming pass over the file |
| `decide` | One forward pass: probability of each choice's first token (not a calibrated confidence) |
| `code "<task>"` | The agent. Review the diff, then `--apply` or answer the prompt |
| `serve` | Local OpenAI/Ollama/MCP server; supervises `llama-server` |
| `receipts`, `ledger verify`, `receipt verify <clt_…\|file>`, `receipt key`, `rollback` | Signed receipts, offline verification (prints the decoded payload), undo |
| `pull`, `models`, `inspect`, `backend`, `doctor`, `status` | Housekeeping |

## What is measured, and what is not

| Thing | Status |
| --- | --- |
| Bytes read from disk, time in `pread`, time the compute thread waited, overlap | **Measured** by the engine on your machine |
| Cache hit rate, admission rejects, prefetch issued / late / unused, substitutions, largest probability moved | **Measured** by the engine, **on synthetic routing** (a skewed random router with drifting topics). A real model's routing will differ |
| "Passes per second" | A streaming pass over synthetic routing, with a real Q4_0 SIMD kernel run over the loaded bytes (not the model's own maths). **Not model tokens per second** |
| Memory plan and ceilings | **Computed** from the file's tensor table or published architecture; disk speed is **assumed** until you run `bench` |
| `chat`, `decide`, the agent | Generated by **llama.cpp**, not by Clutor's engine |
| Hit rates in `simulate` | **Simulated** |

Results on the generated test model (12 layers, 64 experts, 3.2 MiB each; this machine, an M2 with 16 GiB; synthetic routing, 300 tokens, 510 MiB cache):

| | LRU, no prefetch | adaptive + prefetch | + smart-route (2,12,0.015) |
| --- | --- | --- | --- |
| hit rate | 57.5% | 80.2% | 91.0% |
| wall time | 44.3 s | 36.0 s | 30.5 s |
| GB read per token | 0.170 | 0.232 | 0.196 |

Prefetch buys speed by reading more (some prefetches are never used). The window-size search matters: a fixed-window W-TinyLFU did **worse** than LRU on this trace at most cache sizes, and the adaptive one beat LRU at 128 MB (15.7% vs 0%), tied at 1 GiB, and lost 2.9 points at 512 MB. See [docs/NATIVE_ENGINE.md](docs/NATIVE_ENGINE.md).

## A 744B-class stress test (scaled topology)

`glm-5-744b` is a **stress-test profile**, not GLM-5's published configuration: 78 layers x 256 routed experts, top-8, with sizes (9.79 GiB dense including embeddings, 419 GiB of routed experts at Q4_K_M) taken from a brief and the other hyperparameters chosen to reach them. To run the real engine on that shape without 419 GiB of disk, the tests use a **generated 2.34 GiB Q4_0 file with the same topology** (78 x 256 = 19,968 expert slices, 0.12 MiB each, random weights). Hit rates, misses and timings are measured; the routing is synthetic.

**Memory plan (computed; disk speed assumed at 3 GiB/s).** `clutor plan glm-5-744b --ram N`, GiB:

| | 8 | 16 | 25 |
| --- | --- | --- | --- |
| Embeddings, head, routers | 1.53 | 1.53 | 1.53 |
| Dense weights | 0.21 (2-layer window) | 0.21 (2-layer window) | 8.26 (resident) |
| KV cache (8,192 tokens) | 2.44 | 2.44 | 2.44 |
| Double-Buffer DMA (runtime buffers) | 0.38 | 0.38 | 0.38 |
| Expert cache | 1.64 (78 experts) | 7.93 (378 experts) | 8.87 (423 experts) |
| Left for the OS | 1.80 | 3.50 | 3.50 |
| `CL-101` | ok (6.19 of 6.20) | ok (12.5 of 12.5) | ok (21.5 of 21.5) |
| Disk-bound ceiling at 0% / 95% hits | 0.14 / 0.34 tok/s | 0.14 / 0.34 tok/s | 0.23 / 4.58 tok/s |

One token needs 624 experts, so `CL-102` warns at every size: the cache holds 78 to 423. At 8 and 16 GiB the 8.26 GiB of dense weights are re-read every token, which caps generation at 0.36 tok/s even with a perfect cache.

**Cache strategies (measured, synthetic routing).** `clutor bench --file <that file> --cache-mb 256 --tokens 200`: 2,114 of 19,968 experts cached.

| | LRU, no prefetch | adaptive + prefetch | + smart-route (2,12,0.015) |
| --- | --- | --- | --- |
| hit rate | 51.9% | 76.6% | 92.4% |
| misses per token (of 624) | 300.2 | 145.8 | 47.3 |
| admission rejects | 0 | 50,292 | 37,651 |
| substitutions per token (largest probability moved) | 0 | 0 | 96.7 (0.0150) |
| passes per second | 14.6 | 17.3 | 22.6 |

Smart-route is not free accuracy: it swaps about 15% of lookups for cached experts, which is why its hit rate is highest. Passes per second depend on whether the OS had the freshly written file in memory; two runs gave 1.18x and 1.35x for prefetch, and 1.55x and 1.77x for smart-route. Hit rates and misses were stable to within a few points.

**Scan pollution (`stream --shift 80,10 --window 10`, 200 tokens, 256 MB cache).** Tokens 81 to 90 are uniform noise over all 256 experts; the rest use one tight topic. The lookahead predictor gets an independent random draw during the noise, so it cannot predict it. Hit rate per 10-token window:

| | 71-80 | 81-90 (noise) | 91-100 | 191-200 | admission rejects |
| --- | --- | --- | --- | --- | --- |
| LRU, no prefetch | 52.0% | 9.3% | 44.3% | 52.2% | 0 |
| W-TinyLFU, no prefetch | 68.0% | 10.9% | 68.0% | 68.9% | 35,056 |
| adaptive + prefetch | 80.7% | 12.2% | 79.4% | 80.4% | 38,303 |

LRU is flushed by the noise and needs another window to recover; the frequency filter keeps the hot experts. (There is no "AUTOPIN" feature; the protection is W-TinyLFU admission.)

**Memory ceiling.** A 1,000-token run at the same cache size had a peak RSS (`ru_maxrss`) of 269.98 MiB in each of ten 100-token windows: no growth. The engine also computes what it expects to use from its own allocations; it predicted 271.73 MiB against 270.02 MiB measured. The cache fits inside `--cache-mb` (2,114 slots, 256.0 MiB); the whole process sits about 14 MiB above that budget, because the staging buffers, the resident dense weights and the process baseline are not part of it. `ru_maxrss` can only rise, so a flat line means the peak was reached in the first window.

**Receipt.** `clutor bench ... --receipt` signs one entry into `.clutor/receipts.jsonl` with the start and finish times of the run, and `clutor receipt verify <clt_...> --key-id <id>` prints the decoded payload. The receipt proves the text was not changed after signing; it does not prove the numbers are right.

## What Clutor does not do

- It does not make a huge model fast. 744 billion parameters in a few gigabytes at double-digit tokens per second is not possible; the planner prints the real ceiling (for example DeepSeek-V3 Q4 on 16 GiB: about 0.3 tok/s with no cache hits, about 5 at 95%).
- The native engine cannot generate text yet: no attention, no quantized matmuls, no tokenizer. That is the next milestone, and until then text comes from llama.cpp.
- The command sandbox is not a security boundary against malicious code you approve.
- Receipts prove a log was not altered; they do not prove which model ran.
- The AVX2 path of the Q4_0 kernel compiles but has never been run (the author has no x86 machine); the NEON and scalar paths are tested.
- The coding agent is only as reliable as the model behind it: with Qwen2.5-Coder-7B a small test-writing task passed verification in 3 of 6 runs, and when verification failed nothing was written.
- Windows and Linux are untested by the author (macOS arm64 only). CI builds the engine on Linux and macOS.

See [docs/LIMITS.md](docs/LIMITS.md) and [docs/VISION_AUDIT.md](docs/VISION_AUDIT.md).

## Documentation

| | |
| --- | --- |
| [Guide](docs/GUIDE.md) | Install, configure, connect an editor, plan a model, use the agent |
| [Native engine](docs/NATIVE_ENGINE.md) | The C engine: design, cache policy, prefetch, smart-route, results, honest limits |
| [How the planner works](docs/ENGINE.md) | The arithmetic, the cache policies, what is measured and what is modelled |
| [The agent](docs/AGENT.md) | Tools, jail, command policy, approvals, constrained decoding, KV snapshots, receipts |
| [Security model](docs/SECURITY_MODEL.md) | Who can talk to the server and how |
| [Limits](docs/LIMITS.md) | What Clutor will not do, and the claims it will not make |
| [Vision audit](docs/VISION_AUDIT.md) | The original brief, item by item |

## Development

```bash
npm install
npm run gates      # typecheck, lint, build engine, C tests, unit tests, build, bench, smoke
make -C c sanitize SMALL=/path/to/moe.gguf   # engine under ASan, UBSan and TSan
```

MIT licensed. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
