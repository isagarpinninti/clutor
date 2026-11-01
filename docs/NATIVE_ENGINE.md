# The native engine (`c/`)

A C11 program with no dependencies beyond libc, libm and pthreads. Build: `npm run engine` (or `make -C c`). It reads GGUF files directly, so there is no conversion step and no second copy of the model.

**What it is:** an out-of-core *streaming* engine. It parses the GGUF header, finds where every layer's dense weights and every routed expert live, and for each token of a routing trace it loads what that token needs from disk through a cache, while prefetching the next layer. It records what happened.

**What it is not (yet):** a general inference engine. Streaming (`stream`) still computes on bytes that are not the model's weights: its "compute" step is a real **Q4_0 x Q8_0 dot-product kernel** (`c/src/kernel.c`: ARM NEON, with the dot-product instruction when the CPU target has it; AVX2; portable C11) run over every byte the streamer loads, read as Q4_0 whatever their true type (the OLMoE file is Q4_K/Q6_K). `--compute checksum` swaps it for a cheap hash, which makes a run measure I/O alone. Without `--trace` the router is **synthetic**. Every report carries `provenance: {routing: "synthetic" | "real", disk: "measured", compute: ...}`.

**What it can do now: a real forward pass for `olmoe`, `qwen3moe` and `granitemoe` models** (`generate`, `logits`, `tokenize`). `c/src/forward.c` runs embedding, RMSNorm, attention with QK-norm and NeoX rotary embeddings, the softmax router over all experts, the top-8 SwiGLU experts, and the output projection, in float32, reading Q4_K/Q6_K/Q4_0/Q5_0/Q8_0/F16/F32 weights straight from the memory-mapped GGUF (`c/src/quant.c`). `c/src/tokenizer.c` is a byte-level BPE built from the file's own tables. It runs at about 0.03 s per token with the file in RAM on an M2 with 8 threads (see Compute speed below; it was 0.3 s before the integer kernels). By default it maps the whole file; with `--cache-mb N` the experts are instead **streamed from disk through the cache inside the forward pass** (below). `generate --trace FILE` records the router's raw scores for every token and layer; `stream --trace FILE` replays that real routing through the cache, prefetch and smart-route code.

**How it was checked against llama.cpp** (`make verify-llama MODEL=...`, `c/test/verify_llama.py`; OLMoE-1B-7B-0924-Instruct Q4_K_M, this Mac, CPU only):
- Tokenizer: identical token ids on 10 test strings (ASCII, code, numbers, whitespace runs, accents, CJK, Cyrillic, emoji, special tokens). Letter/number/punctuation classes above U+007F are approximated, so unusual scripts could still differ.
- Next-token distribution on 6 prompts: same top-1 token every time, top-5 overlap 4-5 of 5, mean |difference in logprob| 0.04-0.17 over the top 20, worst single difference 0.61. They are not bit-identical: llama.cpp quantizes activations to Q8_K and keeps an f16 KV cache; this engine uses float32 throughout.
- Router: for "The capital of France is", the experts llama.cpp's `llama-eval-callback` shows (first and last three of each token's eight, 456 in all) are inside this engine's top-8 in 96.5% of cases (98.0% when llama.cpp is run with an f32 KV cache). Layer 0 matches exactly. The rest are borderline experts that float rounding flips; this is a measured agreement, not a proof of bit-equality. `c/test/compare_routing.py` repeats the check.
- AddressSanitizer + UBSan: clean on `logits` and `tokenize` with the real model.

**The kernel paths, and what has been tested.** `make test` runs the unit tests (41,795 checks: hand-computed blocks, nibble order, an independent dequantize-and-multiply reference, vector path against portable path) on the path this machine selects. `make test-paths` also builds and runs the portable path and, on arm64, NEON without the dot-product instruction. The AVX2 path compiles for x86_64 (`make test-avx2` cross-builds it) but **has not been executed**: it needs an x86_64 machine with AVX2, and this development machine is an Apple M2 without Rosetta.

## Design

```
            ┌───────────── compute thread ─────────────┐        ┌── io threads (pread) ──┐
 layer L:   │ wait dense L ─▶ route ─▶ acquire experts │ ─jobs▶ │ demand queue  (first)  │
            │ issue prefetch for L+1 ─▶ wait ─▶ touch  │        │ prefetch queue         │
            └───────────────────────────────────────────┘        └────────────────────────┘
 dense ring:  (ring mode) two buffers; layer L+1's dense weights load while L computes
 expert pool: cache slots (policy decides who stays) + staging slots ("Buffer B", prefetched, not yet wanted)
```

- **Dense weights, two modes** (`--dense auto|resident|ring`). *Resident*: every layer's dense weights (attention, norms, shared experts) are read once before the timed run and kept in RAM; steady-state dense traffic is zero. *Ring*: two buffers sized for the largest layer, with layer L+1 read while L is used, re-reading every token; this is the mode for machines where the dense set does not fit. *Auto* (default) picks resident when the dense weights fit in a quarter of physical RAM. Every report says which mode ran, because the choice changes bytes per token a great deal.
- **Expert streamer.** Routed experts are fused per layer in GGUF (one tensor each for gate, up and down holds every expert, experts slowest-varying), so one expert is a few contiguous byte ranges. They are read with `pread` by a small pool of I/O threads; demand reads jump ahead of prefetches.
- **Cache policy** (`cache.c`, no locks, no I/O): LRU; W-TinyLFU (small admission window, then a segmented LRU whose admission is decided by a 4-bit count-min sketch of access frequency, aged by periodic halving); and **adaptive** (default), which hill-climbs the window size by comparing hit rates between periods.
- **Prefetch staging.** A prefetched expert goes to a separate staging area, not into the policy's window. Otherwise a 1% window would evict it before it is used, and a prefetch could displace something the policy chose to keep. A staged expert that is then wanted is adopted into the cache without copying; one that never is wanted is counted as wasted.
- **Prefetch prediction.** Layer L+1's experts are predicted while layer L computes. With synthetic routing the predictor is the synthetic router with noise, correlated with the true router by `--predict-corr` (default 0.8, **an assumption**). With `--trace` the predictor is causal and honest: it guesses layer L+1 from what that layer chose for the previous token (nothing for the first token), and its accuracy is whatever the trace gives.
- **SMART-ROUTE (J, M, τ).** Keep the true top-J experts. For each remaining slot whose expert is *not* in memory, use an expert that *is* in memory if it ranks in the top-M and its router probability is within τ of the one it replaces. With τ = ∞ this is unbounded cache-aware routing; with τ = 0.015 no single substitution can move more than 0.015 of probability, and the engine reports the largest it actually moved and the total. Whether that bound is small enough to leave output quality unchanged is **not tested** here, because there is no model to test it on.
- **Pinning.** Experts in use are pinned and cannot be evicted. If everything evictable is pinned the access bypasses the cache and is counted.

## Reading the output

```
clutor stream models/x.gguf --tokens 300 --cache-mb 512 --smart 2,12,0.015 --no-os-cache
```

| Line | Meaning |
| --- | --- |
| `GB read; waited X s of Y s` | Bytes read from the file; time the compute thread sat idle waiting for them, out of wall time |
| `overlap` | `1 − wait / total read time`: how much reading happened while something else was going on |
| `cache N% hits (M served by prefetch)` | Demand accesses found in the cache or in staging |
| `prefetch issued / late / never used` | Late: the expert had not finished loading when it was wanted. Never used: staged, then dropped |
| `smart-route … largest probability moved` | Never exceeds τ (tested) |

`--no-os-cache` asks the OS not to keep what is read (`F_NOCACHE` on macOS, `posix_fadvise(DONTNEED)` on Linux). Without it, a model that fits in free RAM is served from the OS page cache after the first pass and the "disk" numbers are RAM numbers.

## Results so far

Machine: Apple M2, 16 GiB, internal SSD. Model: generated MoE file, 12 layers × 64 experts (8 used), 3.2 MiB per expert, 2.4 GiB, zero-filled (the bytes are real reads; the contents are not weights). Synthetic routing, locality 1.0, seed 1.

**Policy, prefetch off, 400 tokens after 100 warm-up** (hit rate):

| Cache | LRU | W-TinyLFU, window 1% | window 10% | window 40% | Adaptive |
| --- | --- | --- | --- | --- | --- |
| 128 MB (40 experts) | 0.0% | 0.0% | 0.0% | 14.3% | **15.7%** |
| 512 MB (160) | **57.5%** | 40.5% | 55.6% | 58.0% | 54.6% |
| 1 GiB (321) | 82.9% | 78.9% | 79.5% | 83.5% | 82.8% |

Reading it honestly: a fixed small window (the textbook W-TinyLFU) loses to LRU here; LRU loses completely when the cache is too small for a token's working set (a cyclic scan); the adaptive version finds a good window by itself at the small size, ties at the large size, and is 2.9 points behind LRU at 512 MB. It is not a free win. On a different workload the ordering could differ, which is why it adapts.

**Strategies, 510 MiB cache, 300 tokens** (`clutor bench --file … --cache-mb 512`):

| | LRU, no prefetch | adaptive + prefetch | + smart-route (2,12,0.015) |
| --- | --- | --- | --- |
| hit rate | 57.5% | 80.2% | 91.0% |
| misses per token | 40.8 | 19.0 | 8.7 |
| GB read per token | 0.170 | 0.232 | 0.196 |
| time waiting for disk | 39.5 s | 31.1 s | 25.8 s |
| wall time | 44.3 s | 36.0 s | 30.5 s |

Prefetch reads *more* bytes (some prefetches are wrong or late) to wait less. Smart-route then removes misses by substituting resident experts, at the stated probability cost.

**What would change these numbers:** a real model's routing (more or less reuse, different cross-layer correlation), the real predictor accuracy, a real SSD under a cold cache, and compute that is this model's real maths instead of a kernel over bytes it was not written for. None of those were available, so none are claimed.

## Safety of the code

- 40,212 C unit checks (`make -C c test`): cache invariants under random traffic with pins (never over capacity, map and slots agree), scan resistance, staging, SMART-ROUTE bounds and distinctness, the GGUF reader on a real llama.cpp file, a missing file and junk input.
- `make -C c sanitize SMALL=<moe.gguf>`: the unit tests under AddressSanitizer + UBSan, and a threaded streaming run under ASan/UBSan and under ThreadSanitizer. All clean on the author's machine.
- The C GGUF parser is cross-checked against the independent TypeScript parser on the same files (totals, resident bytes, per-layer dense / expert / router bytes) in `packages/cli/test/engine.test.ts`.
- The parser bounds-checks every read, caps string and array sizes, and refuses a tensor that points outside the file.

## Not done

Forward pass (attention, RoPE, norms, quantized matmuls for Q4_K / Q8_0 / …), a tokenizer, a KV cache, validation of logits against llama.cpp, Metal / NEON-tuned kernels, the learned pinned-hot-expert set, multi-SSD mirroring, and the real next-layer predictor. The first of those is the milestone that turns this from a measuring instrument into an engine.


## Real routing against synthetic routing (measured 2026-10-31)

One trace: OLMoE-1B-7B Q4_K_M generated 109 tokens after "Once upon a time, in a small village by the sea, there lived an old fisherman who" (about 130 tokens of real router scores). Replay: 100 measured tokens after 20 warm-up, 512 MiB expert cache (about 131 of 1024 experts), 3 I/O threads, OS cache bypass requested, this Mac's internal SSD. One prompt, one machine, one run each: treat the gaps as indicative.

| routing and policy | hit rate | misses / token | GB read / token |
|---|---|---|---|
| real, LRU, no prefetch | 34.7% | 83.6 | 0.32 |
| real, adaptive, no prefetch | 31.3% | 88.0 | 0.34 |
| real, adaptive + prefetch from the previous token | 46.8% | 68.1 | 0.55 |
| real, adaptive + prefetch + smart-route 2,12,0.015 | 56.9% | 55.1 | 0.50 |
| **synthetic** (locality 1.0), adaptive + prefetch | 77.1% | 29.3 | 0.38 |

What this says, and does not: the synthetic router's default skew was **far too optimistic** for this model (77% against 47% for the same cache and policy), so earlier hit-rate figures from synthetic runs should not be read as what a real model gets. On this trace plain LRU beat the adaptive policy without prefetch; the admission-aware cache has not shown an advantage on real OLMoE routing yet. Prefetch from the previous token helps hit rate but reads 60% more bytes, most of it never used (5,504 of 7,536). Smart-route raises the hit rate further; its cost in output quality is now measured inside the real forward pass (next sections): small settings cost about 2-3% perplexity and change the top-1 token at 6-10% of positions.


## Compute speed (measured 2026-10-31)

The first forward pass dequantized every weight block to float32 and multiplied in float: about **0.30 s per token**. Profiling put 95% of the time in those loops and the rest in creating threads for each matrix product. It now quantizes each activation vector to 8-bit blocks of 256 (ggml's Q8_K layout) and multiplies K-quant weights with integer arithmetic (`c/src/quant.c`: NEON dot-product instruction, with a portable scalar path), runs on a persistent thread pool whose workers spin briefly before sleeping, and batches a layer's expert products into two parallel jobs (all gate/up, then all down). Result: **0.03 s per token** with the file in RAM, M2 MacBook Air, 8 threads, short prompts, one machine. The output text is unchanged, and agreement with llama.cpp improved because the activation rounding now matches: the router agrees on 99.8% of the visible choices against an f32 KV cache (97.4% against llama.cpp's default f16 cache; it was 98.0% and 96.5%). `make test` and `make test-paths` check the integer kernels against dequantize-and-multiply on both paths. No x86 vector path exists yet (it falls back to the scalar one, which is correct but slow).

## Streaming inside the real forward pass (measured 2026-10-31)

`generate --cache-mb N` serves each layer's experts from `c/src/xs.c`: an I/O thread pool reads the eight chosen experts at once, misses go through the same LRU / W-TinyLFU / adaptive policy and staging area as `stream`, and the dense weights (attention, norms, routers, embeddings) stay mapped. Optionally (`--prefetch on [--prefetch-n N]`) the experts the next layer chose for the **previous token** are read early; this is off by default because it lost on every disk-bound measurement below.

**A correction.** An earlier version of this section reported "no visible slowdown from streaming". That came from a slow float compute path (0.3 s/token) that hid the disk, and from a benchmark that ran the mapped mode first, which loads the whole file into the OS cache so the later "streamed" runs read RAM (an impossible 11 GB/s showed it). The script now runs the mapped mode last, and the numbers below are from a freshly copied file that nothing had mapped.

Setup: OLMoE-1B-7B Q4_K_M (3.92 GiB) copied with `fastcopy` (written without caching), reads with `F_NOCACHE`, 80-token prompt-plus-generation as in `c/test/bench_stream.sh`, 3 I/O threads, M2 MacBook Air internal SSD. A fresh single-thread read of the file runs at about 1.5 GB/s; with three or more threads the engine's effective read rate saturates at about 1.4-2.7 GB/s (6 and 12 threads gave nothing more). One run per row; **the same setting varied by about 30% between runs** (256 MiB without prefetch: 0.19, 0.24, 0.25 s/token), so read the trends, not the decimals.

| mode | s / token | peak RSS | hit rate | GB read / token |
|---|---|---|---|---|
| stream, 128 MiB, adaptive, prefetch 8 | 0.35 | 544 MiB | 33.6% | 0.76 |
| stream, 256 MiB, adaptive, prefetch 8 | 0.29 | 672 MiB | 38.4% | 0.68 |
| stream, 512 MiB, adaptive, prefetch 8 | 0.20 | 927 MiB | 45.9% | 0.55 |
| stream, 1024 MiB, adaptive, prefetch 8 | 0.16 | 1,435 MiB | 59.2% | 0.39 |
| stream, 256 MiB, LRU, prefetch 8 | 0.27 | 672 MiB | 33.7% | 0.76 |
| stream, 256 MiB, adaptive, no prefetch | 0.19-0.25 | 610 MiB | 16.6% | 0.41 |
| whole file mapped (run last) | 0.07 | 3,711 MiB | n/a | n/a |

Prefetching only the previous token's top-N experts per layer (256 MiB, adaptive, second cold copy): N=0: 0.25 s/token, 16.6% hits, 0.41 GB/token; N=1: 0.25, 22.1%, 0.43; N=2: 0.28, 23.7%, 0.47; N=4: 0.29, 28.1%, 0.56; N=8: 0.34, 38.1%, 0.68. I/O threads (no prefetch): 1 thread 0.34 s/token (0.96 GB/s effective), 3 threads 0.24 (1.40), 6 threads 0.24, 12 threads 0.23.

What this shows:
- **Same output everywhere.** The generated text is byte-identical in every mode (`c/test/verify_stream.py` checks six more settings).
- **Bounded memory, at a speed cost.** Peak resident memory is 544 MiB to 1.4 GiB against 3.7 GiB mapped, but streaming is 2.3x to 5x slower than the mapped file on this machine: with compute at 0.03 s/token, the run is **disk-bound** (the compute thread waits for the disk about 85-90% of the time).
- **Fewer bytes is the only lever left.** The disk tops out near 1.4 GB/s here, so tokens per second is set by bytes read per token: a 1 GiB cache cut that from 0.76 to 0.39 GB/token and the time from 0.35 to 0.16 s. Faster compute or more threads cannot help; a better cache policy, a smarter predictor or router substitution (not yet applied in the real pass) could.
- **Previous-token prefetch does not pay.** It lifts the hit rate (16.6% to 38.1% at 256 MiB) but reads 66% more bytes, about two thirds of them never used, and every step up in N got slower.
- **Not demonstrated:** a model larger than RAM (the only real MoE here is 3.9 GiB on a 16 GiB machine), a slower or faster disk, and any comparison with AirLLM or Colibri. The address, undefined and thread sanitizers are clean on the streamed path.


## SMART-ROUTE inside the real forward pass (measured 2026-10-31)

`generate` and `quality` accept `--smart J,M,TAU` together with `--cache-mb N`. At each layer the engine asks the cache which experts are in memory right now (`cl_xs_resident`), `cl_route` keeps the true top-J, lets a slot that would need a cold expert take a resident expert from the top-M if its router probability is within TAU, and the substituted expert is weighted by **its own** router probability. Resident experts are requested before cold ones, so a miss cannot evict an expert the same layer is about to use. `quality` runs the same text through exact routing (whole file mapped) and the configured run in lock-step, teacher-forced, and reports perplexity of each, the KL divergence of the run from exact, and how often the highest-probability token differs.

Setup: OLMoE-1B-7B Q4_K_M, 799 tokens scored of this project's own README, this document and the changelog (English technical prose; one text, one machine, one run per row), 256 MiB expert cache, adaptive policy, no prefetch.

| smart-route J,M,TAU | perplexity (exact 10.943) | mean KL | top-1 token agrees | slots substituted | cache hit rate | GB read / token | largest probability moved |
|---|---|---|---|---|---|---|---|
| off | 10.943 (x1.000) | 0 | 100% | 0% | 19.7% | 0.392 | 0 |
| 6,16,0.02 | 11.150 (x1.019) | 0.034 | 93.5% | 6.0% | 29.4% | 0.345 | 0.020 |
| 4,16,0.02 | 11.198 (x1.023) | 0.048 | 92.1% | 6.3% | 29.5% | 0.344 | 0.020 |
| 4,16,0.05 | 11.311 (x1.034) | 0.075 | 89.5% | 8.0% | 30.9% | 0.338 | 0.050 |
| 2,24,0.05 | 11.814 (x1.080) | 0.143 | 84.7% | 11.0% | 34.8% | 0.319 | 0.050 |
| 2,32,0.2 | 12.354 (x1.129) | 0.191 | 83.1% | 13.5% | 37.5% | 0.305 | 0.166 |
| 0,64,1.0 (unbounded cache-aware routing) | 29.718 (x2.716) | 1.386 | 61.3% | 18.9% | 40.0% | 0.294 | 0.691 |

What this shows, and does not:
- **The saving is real but modest.** Mild settings read 12-14% fewer bytes per token (0.392 to 0.338-0.345 GB) for 2-3% higher perplexity. The largest saving, 25%, comes with perplexity 2.7x higher: unbounded cache-aware routing damages this model badly, which is what the probability bound is for.
- **A bounded probability move is not free.** Even TAU = 0.02 changes the highest-probability token at 6.5% of positions here. Whether that matters depends on the task; perplexity on prose does not measure code or tool-use quality.
- **The byte saving is measured; a time saving is not.** Reads are 12-14% lower, so a disk-bound run should speed up by about that much at most; the quality runs map the whole file for the exact reference, which warms the OS cache, so their timings say nothing about disk speed.
- **Not tested:** other models, other text, longer contexts, and any downstream task.
- A bug check that matters: with SMART-ROUTE off, the streamed run matches exact routing token for token (KL 0, perplexity identical), and `J = 8` (nothing may be substituted) is byte-identical to exact in `c/test/verify_stream.py`.


## Router-ahead prefetch (measured 2026-10-31)

`--prefetch ahead` applies the NEXT layer's router to the current hidden state (right after this layer's attention) and starts reading the experts it picks, so the reads overlap this layer's experts. `--prefetch prev` is the earlier guess from the previous token's choices; `off` is the default. The engine reports what share of the predicted experts the router then chose.

Setup: OLMoE-1B-7B Q4_K_M, a fresh cold copy of the file, `F_NOCACHE` reads, 256 MiB cache, 80 generated tokens, two runs per row (the Qwen3 download was running in the background, which adds a little noise).

| mode | job seconds | hit rate | GB read / token | predicted experts chosen |
|---|---|---|---|---|
| off | 26.5, 24.8 | 15.5% | 0.41 | n/a |
| prev | 35.7, 35.9 | 38% | 0.68 | 37% |
| ahead | 26.1, 25.9 | 84% | 0.48 | 82% |

- **The predictor is good:** 82% of the experts it picks are chosen, against 37% for the previous-token guess, and the hit rate rises from 15% to 84%.
- **It does not make this run faster.** The run is bandwidth-bound (the compute thread waits on the disk about 88% of the time at roughly 1.5 GB/s), so time is bytes read divided by bandwidth. Starting reads earlier cannot beat that, and the 18% of wrong guesses add about 17% more bytes. Time with `ahead` equals time with no prefetch; `prev` is slower because most of what it fetches is unused.
- **So prefetch stays off by default.** Prediction should help where the run is latency-bound or compute-bound (a faster disk or a heavier model); that is untested here. The only levers that cut time on this machine are the ones that cut bytes: a larger cache, a better eviction policy, smart-route, and lower-bit experts.
