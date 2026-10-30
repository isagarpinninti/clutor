# The native engine (`c/`)

A C11 program with no dependencies beyond libc, libm and pthreads. Build: `npm run engine` (or `make -C c`). It reads GGUF files directly, so there is no conversion step and no second copy of the model.

**What it is:** an out-of-core *streaming* engine. It parses the GGUF header, finds where every layer's dense weights and every routed expert live, and for each token of a routing trace it loads what that token needs from disk through a cache, while prefetching the next layer. It records what happened.

**What it is not (yet):** a transformer. There is no attention, no tokenizer and no forward pass, so it cannot generate text. The "compute" step is a real **Q4_0 x Q8_0 dot-product kernel** (`c/src/kernel.c`: ARM NEON, with the dot-product instruction when the CPU target has it; AVX2; portable C11) run over every byte the streamer loads. The streamed bytes are read as Q4_0 whatever their true type (the OLMoE file is Q4_K/Q6_K), so the kernel is genuine CPU work at the right data rate but is **not that model's maths**. `--compute checksum` swaps it for a cheap hash, which makes a run measure I/O alone. The router that decides which experts a token wants is **synthetic**. Every report carries `provenance: {routing: "synthetic", disk: "measured", compute: "Q4_0 x Q8_0 dot-product kernel (neon+dotprod) ..."}` and `config.computeKernel` names the path that ran.

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
- **Prefetch prediction.** Layer L+1's experts are predicted while layer L computes. In a real model that would be layer L+1's router applied to layer L's hidden state. Here the predictor is the synthetic router with noise, correlated with the true router by `--predict-corr` (default 0.8, **an assumption**, not a measurement).
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
