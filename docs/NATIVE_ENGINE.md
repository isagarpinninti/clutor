# The native engine (`c/`)

A C11 program with no dependencies beyond libc, libm and pthreads. Build: `npm run engine` (or `make -C c`). It reads GGUF files directly, so there is no conversion step and no second copy of the model.

**What it is:** an out-of-core *streaming* engine. It parses the GGUF header, finds where every layer's dense weights and every routed expert live, and for each token of a routing trace it loads what that token needs from disk through a cache, while prefetching the next layer. It records what happened.

**What it is not (yet):** a general inference engine. Streaming (`stream`) still computes on bytes that are not the model's weights: its "compute" step is a real **Q4_0 x Q8_0 dot-product kernel** (`c/src/kernel.c`: ARM NEON, with the dot-product instruction when the CPU target has it; AVX2; portable C11) run over every byte the streamer loads, read as Q4_0 whatever their true type (the OLMoE file is Q4_K/Q6_K). `--compute checksum` swaps it for a cheap hash, which makes a run measure I/O alone. Without `--trace` the router is **synthetic**. Every report carries `provenance: {routing: "synthetic" | "real", disk: "measured", compute: ...}`.

**What it can do now: a real forward pass for `olmoe` models** (`generate`, `logits`, `tokenize`). `c/src/forward.c` runs embedding, RMSNorm, attention with QK-norm and NeoX rotary embeddings, the softmax router over all experts, the top-8 SwiGLU experts, and the output projection, in float32, reading Q4_K/Q6_K/Q4_0/Q5_0/Q8_0/F16/F32 weights straight from the memory-mapped GGUF (`c/src/quant.c`). `c/src/tokenizer.c` is a byte-level BPE built from the file's own tables. It is slow (about 0.3 s per token on an M2 with 8 threads, weights dequantized on the fly, no batching). By default it maps the whole file; with `--cache-mb N` the experts are instead **streamed from disk through the cache inside the forward pass** (below). `generate --trace FILE` records the router's raw scores for every token and layer; `stream --trace FILE` replays that real routing through the cache, prefetch and smart-route code.

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

What this says, and does not: the synthetic router's default skew was **far too optimistic** for this model (77% against 47% for the same cache and policy), so earlier hit-rate figures from synthetic runs should not be read as what a real model gets. On this trace plain LRU beat the adaptive policy without prefetch; the admission-aware cache has not shown an advantage on real OLMoE routing yet. Prefetch from the previous token helps hit rate but reads 60% more bytes, most of it never used (5,504 of 7,536). Smart-route raises the hit rate further, but this engine does not yet apply its substitutions inside the forward pass, so the cost in output quality is **unmeasured**.


## Streaming inside the real forward pass (measured 2026-10-31)

`generate --cache-mb N` serves each layer's experts from `c/src/xs.c`: an I/O thread pool reads the eight chosen experts at once, misses go through the same LRU / W-TinyLFU / adaptive policy and staging area as `stream`, and while layer L computes, the experts that layer L+1 chose for the **previous token** are read early (a causal guess, not an oracle). The dense weights (attention, norms, routers, embeddings) stay mapped.

Setup: OLMoE-1B-7B Q4_K_M (3.92 GiB), the 80-token prompt "Once upon a time, in a small village by the sea, there lived an old fisherman who" plus 40 generated tokens, the model file copied with `fastcopy` so it was not sitting in the OS cache, reads with `F_NOCACHE`, 3 I/O threads, 8 compute threads, M2 MacBook Air with an internal SSD (a single thread reads this file at about 2.0 GB/s). One run per row.

| mode | s / token | peak RSS | hit rate | GB read / token |
|---|---|---|---|---|
| whole file mapped | 0.31-0.36 | 3,548 MiB | n/a | n/a |
| stream, 128 MiB cache, adaptive + prefetch | 0.32 | 533 MiB | 34.6% | 0.80 |
| stream, 256 MiB, adaptive + prefetch | 0.46 | 648 MiB | 36.9% | 0.72 |
| stream, 512 MiB, adaptive + prefetch | 0.37 | 917 MiB | 43.6% | 0.53 |
| stream, 1024 MiB, adaptive + prefetch | 0.35 | 1,420 MiB | 56.8% | 0.39 |
| stream, 256 MiB, LRU + prefetch | 0.49 | 657 MiB | 34.5% | 0.80 |
| stream, 256 MiB, adaptive, no prefetch | 0.52 | 600 MiB | 15.1% | 0.41 |

What this shows:
- **Same output.** The generated text is byte-identical in every mode (`c/test/verify_stream.py` checks five more cache, policy, thread and prefetch settings). Streaming changes where the bytes come from, not the maths.
- **Bounded memory.** Peak resident memory is 533 MiB to 1.4 GiB depending on the cache, against 3.5 GiB when the file is mapped: 6.7x lower at the smallest cache.
- **No visible slowdown on this machine**, because the compute thread waited for the disk only 0.5-3 s of 19-30 s. That is a property of a fast internal SSD and a slow float32 compute path (0.3 s/token); the run-to-run spread (0.31 to 0.52 s/token across rows) is larger than most gaps between rows, so do not read a ranking into the timings.
- **Prefetch trades bandwidth for hits.** It lifts the 256 MiB hit rate from 15% to 37%, but reads 75% more bytes and about 60% of the prefetched experts are never used. On a slower disk that extra traffic would compete with demand reads; this machine could not show that.
- **Not demonstrated:** a model larger than RAM (the only real MoE here is 3.9 GiB on a 16 GiB machine), a slow disk, and any comparison with AirLLM or Colibri. The sanitizers (address, undefined, thread) are clean on the streamed path.
