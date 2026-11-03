# Test report

Run on 2026-10-31 on one MacBook Air (Apple M2, 16 GiB, macOS), repository state at commit `8aa3c36` plus the test tooling committed with this report. "Re-run now" means run again while writing this report; "earlier" means run earlier the same day and not repeated (llama.cpp needs about 30 minutes for a full pass on the 17 GiB model). Every number is measured. The re-run commands are in the last column.

## 1. Automated suites (re-run now)

| Suite | Result | What it covers | Re-run |
|---|---|---|---|
| C unit tests, three kernel builds | 41,811 checks, 0 failed, on NEON+dotprod, NEON, and portable C | cache policies and invariants, router and SMART-ROUTE bounds, GGUF and metadata readers, dequantizers, integer dot products against dequantize-and-multiply, the expert store | `make -C c test test-paths` |
| Sanitizer build of the unit tests | 41,811 checks, 0 failed under AddressSanitizer + UBSan | memory safety of the same code | `make -C c sanitize` |
| Golden test | 90 checks OK, 0 FAIL, in 15 blocks (olmoe, qwen3moe, granitemoe, dense qwen2 and dense llama, each on 3 builds) | the engine against an independent pure-Python forward pass on a tiny generated model with random Q4_K/Q6_K weights: exact float math agrees to 8e-5 in logprob, and the 8-bit integer path agrees to 5e-5 against a reference that applies the same 8-bit activation rounding to every K-quant product (it used to be checked only against a loose tolerance) | `make -C c golden-paths` |
| Fuzzing | 600 corrupted model files this run (3,600 earlier), 0 crashes or hangs under ASan + UBSan | the GGUF, metadata and tokenizer readers, and the forward pass on bad shapes | `make -C c fuzz` |
| TypeScript | typecheck clean, lint clean, 168 core tests and 98 CLI tests pass | planner, agent, receipts, server, CLI, installer (checksum match, tamper, missing line, insecure URL), resumable verified downloads, `generate` and `quality` | `npm run typecheck && npm run lint && npm test` |
| CI on GitHub | all 6 jobs green on the last pushed commit (Linux and macOS, Node 20 and 22, sanitizers + fuzz, dependency audit) | includes the AVX2 kernels on a real x86 Linux runner: `kernel path: avx2`, 41,811 checks, 0 failed | `gh run list` |

## 2. Agreement with llama.cpp, OLMoE-1B-7B Q4_K_M (re-run now)

| Check | Result |
|---|---|
| Tokenizer, 24 strings (ASCII, code, digits, newlines, accents, CJK, Cyrillic, emoji, special tokens) | identical, 0 mismatches |
| Next-token distribution, 6 prompts | token ids identical 6/6; same top token 6/6; top-5 overlap 4-5 of 5; mean abs logprob difference 0.04-0.22 (worst single 0.49) |
| Router choices (visible 6 of 8 experts, all 16 layers) | 455/456 = **99.8%** inside this engine's top-8 (llama.cpp with an f32 KV cache) |
| Greedy continuation, 3 prompts, 24 tokens (first measured 2026-11-01) | identical on 2 of 3; the story prompt diverges at generated token 7, a near-tie 0.15 wide. llama.cpp evaluating the same text in one batch ranks the candidates exactly as this engine does (this engine's exact-float path: 43519 at -0.862 and 1500 at -1.015, llama.cpp's batch evaluation: -0.862 and -1.013) while llama.cpp's own incremental greedy run chose the other one: llama.cpp differs from itself between its batch and token-by-token paths here, and the engine agrees with its batch path |
| Streamed from disk equals mapped, 8 cache/policy/prefetch/smart settings | byte-identical text in all 8 |
| Perplexity against Colibri's engine on the same 799 tokens (earlier) | 10.943 (this engine) against 10.87 (Colibri) |

## 3. Agreement with llama.cpp, Qwen3-30B-A3B Q4_K_M (17.3 GiB, larger than this machine's RAM)

The file was downloaded and verified byte-exact: 18,556,685,824 bytes, SHA-256 `0d003f66...a8f5`, equal to the publisher's hash. Results from earlier today:

| Check | Result |
|---|---|
| Tokenizer, 24 strings | identical, 0 mismatches |
| Runs end to end on a 16 GiB machine | yes: "The capital of France is Paris. The capital of Italy is Rome." at 0.52 s/token, peak RSS 4.0 GiB, 4 GiB expert cache, 65% hit rate. Without `--cache-mb` the engine refuses to map the file, by design |
| Router choices, 3 prompts, all 48 layers | 98.0%, 96.7%, 97.0% of visible experts inside this engine's top-8 |
| **Greedy continuation, 3 prompts, 24 tokens each** | **identical in all three** (llama.cpp against this engine) |
| Next-token logprobs, 6 prompts | token ids identical 6/6, top token 6/6, top-5 overlap 5/5 on all; but mean abs logprob difference 0.16-2.23 (worst single 6.0). 3 of 6 exceed the tolerances set on OLMoE and are marked FAIL by the script |

How to read the last row. The script's tolerances were tuned on OLMoE and Qwen3 is far more sensitive: it is 48 layers deep with 8-of-128 routing, so tiny numerical differences (8-bit activation rounding, accumulation order, llama.cpp's f16 KV cache against this engine's f32) are amplified. Evidence that it is noise and not a bug: the routing and the greedy text match, the tiny Qwen3-style model agrees with its independent reference to 6e-5, and for one prompt the engine's exact-float logits differ from its own integer-path logits by up to 1.3, larger than the gap to llama.cpp. The worst prompt (mean 2.23) is OLMoE's chat-tag text, which Qwen was never trained on, so its distribution is flat and fragile. This is judged noise on the evidence, not proven; a comparison on longer continuations or a perplexity run against llama.cpp would tighten it.

## 3b. Agreement with llama.cpp, Granite 3.0 3B-A800M Q4_K_M (the third architecture; run 2026-11-01)

The file (2,059,356,000 bytes, 1.92 GiB) was downloaded across several dropped connections and verified byte-exact against the publisher's SHA-256 before use. It fits in RAM, so it ran mapped and streamed.

| Check | Result |
|---|---|
| Tokenizer (`refact` pre-tokenizer), 24 strings, on the real file | identical, 0 mismatches |
| Next-token logprobs, 6 prompts | token ids identical 6/6; same top token 6/6; top-5 overlap 5/5 on all; mean abs logprob difference 0.11-0.40, worst single 1.03; 2 prompts (0.395 and 0.402) just exceed the 0.35 tolerance tuned on OLMoE and are marked FAIL by the script |
| Router choices, 3 prompts, all 32 layers | 99.0%, 98.4%, 98.5% of visible experts inside this engine's top-8 |
| Greedy continuation, 3 prompts, 24 tokens | identical in all three |
| Streamed from disk equals mapped, 8 settings (including SMART-ROUTE J=8) | byte-identical text in all 8 |
| AddressSanitizer + UBSan, real model, streamed with SMART-ROUTE | no reports |
| Regression: OLMoE logits, tokenizer and unit tests after the architecture refactor | unchanged |

Judged the same way as Qwen3: the routing and the generated text match, the exact-float path agrees with an independent reference to 5e-5 on a tiny Granite-style model, and the remaining logprob gaps look like numerical noise, which is not proven. Qwen3 and Granite both show larger gaps than OLMoE; both are deeper or sharper models than OLMoE's 16 layers.

## 3c. Agreement with llama.cpp, Qwen2.5-1.5B-Instruct Q4_K_M (the fourth architecture, dense; run 2026-11-01)

The file (1,117,320,736 bytes, 1.04 GiB) was verified byte-exact against the publisher's SHA-256 before use.

| Check | Result |
|---|---|
| Tokenizer, 24 strings, on the real file | identical, 0 mismatches |
| Next-token logprobs, 6 prompts | token ids identical 6/6; same top token 6/6; top-5 overlap 4-5 of 5; mean abs logprob difference 0.09-0.22 (worst single 1.03); all 6 inside the tolerances |
| Greedy continuation, 3 prompts, 24 tokens | identical on 1 of 3; the other two diverge at generated tokens 14 and 11 |
| AddressSanitizer + UBSan on the real model; 200 fuzzed dense files | no reports; 0 failures |
| Regression: OLMoE logits (6/6) and tokenizer, Granite (same 4 of 6 as before), unit tests | unchanged |
| The engine refuses `--cache-mb` and `--smart` on this dense model | yes, with an explanation |

The two greedy divergences, investigated. At each, the top two candidates are close in llama.cpp itself (0.36 apart). On the code prompt (token 14), exact float math ranks the candidates the way llama.cpp does (75698 first, 1887 second, 0.21 apart); the 8-bit activation rounding of the integer path lifts 1887 by 0.2 and flips the near-tie, and llama.cpp's own rounding lands differently from mine. On the story prompt (token 11), llama.cpp evaluated on the identical text prefix ranks 264 first, as this engine does, while its own incremental decode chose 566: llama.cpp disagrees with itself between batch and token-by-token evaluation at this position, and the engine agrees with its batch evaluation to within 0.02-0.05. So the evidence is near-tie flips from numerical noise, not a structural error; but this is a weaker result than the three earlier models (identical continuations on all 3 prompts for OLMoE, Qwen3 and Granite), and a long-generation comparison or a perplexity comparison with llama.cpp is the check that would settle it.

## 3d. Agreement with llama.cpp, Llama 3.2 1B Instruct Q4_K_M (the fifth architecture, dense; run 2026-11-01)

The file (807,694,464 bytes, 770 MiB) was verified byte-exact against the publisher's SHA-256 before use.

| Check | Result |
|---|---|
| Tokenizer (`llama-bpe`: digits in groups of three, whole-vocabulary-entry shortcut), 24 strings, on the real file | identical, 0 mismatches |
| Next-token logprobs, 6 prompts (with the BOS token) | token ids identical 6/6; same top token 6/6; top-5 overlap 4-5 of 5; mean abs logprob difference 0.05-0.21 (worst single 0.52); all 6 inside the tolerances |
| Greedy continuation, 3 prompts, 24 tokens | identical on 1 of 3; the other two diverge at generated tokens 3 and 12 |
| AddressSanitizer + UBSan on the real model; 200 fuzzed Llama files | no reports; 0 failures |
| Regression after the tokenizer and rope changes: tokenizer on OLMoE, Qwen2.5, Granite; logits OLMoE 6/6, Qwen2.5 6/6, Granite 4/6 (as before) | unchanged |

**A bug the real file found, and a test it motivated.** The first comparison failed on all 6 prompts with large gaps (up to 6.8): this Llama GGUF has no `tokenizer.ggml.add_bos_token` key, `llama-tokenize` still prepends BOS, and the engine defaulted to no BOS, so it fed the model a different context. llama.cpp's loader turns BOS on by default for Llama 3 vocabularies (it also sets `ignore_merges`); the engine now does both. The golden test's Llama variant now omits the key, as the real file does, so the default is covered.

The two greedy divergences, investigated the same way as for Qwen2.5. At token 3 (prompt "The capital of France is"), llama.cpp evaluating the prefix as one batch ranks 469 over 6864 by 0.17, as does this engine (gap 0.05 integer, 0.13 exact float), while llama.cpp's own incremental run chose 6864: llama.cpp disagrees with its own batch evaluation. At token 12 (the code prompt), llama.cpp ranks 76798 over 1925 by 0.10 and this engine's exact float math agrees (gap 0.014); the 8-bit activation rounding of the integer path flips it to 1925 by 0.03. Near-ties of at most 0.17: consistent with numerical noise, not proof. Two dense models (Qwen2.5 and Llama) now show 1 of 3 identical greedy continuations, against 3 of 3 for Granite and Qwen3 and 2 of 3 for OLMoE; a perplexity or long-generation comparison with llama.cpp is the check that would settle whether dense models carry a small systematic difference.

## 3e. Agreement with llama.cpp, Qwen3.5-9B Q4_K_M (the sixth architecture, hybrid; run 2026-11-03)

llama.cpp `llama-server -ngl 0` (CPU) against `clutor-engine`, same file (5.23 GiB, 32 layers: 24 Gated DeltaNet, 8 gated attention).

| Check | Result |
|---|---|
| Tokenization, 24 strings (`verify_tokenizer.py`) | identical 24/24, including `<think>`, `</think>` and the ChatML markers |
| Next-token logprobs, 6 prompts (`verify_llama.py`) | token ids identical 6/6; same top token 6/6; top-5 overlap 4-5 of 5; mean abs logprob difference 0.085-0.146 on the five normal prompts (worst single 0.36) and 0.327 on the off-format sixth (worst 0.86); all 6 inside the tolerances |
| Greedy continuation, 3 prompts, 24 tokens (`verify_greedy.py`) | identical on 3 of 3 |
| Layer 0, tensor by tensor against `llama-eval-callback` (sums over the 5 prompt tokens) | attn_norm 3e-6, qkv projection 4e-8, z 2e-6, softplus 4e-8, gate 8e-8, beta 2e-7, convolution+SiLU 6e-7, both L2 norms 1e-6, delta-rule output 1e-6, gated norm 1e-6, output projection 2e-6; the feed-forward sums agree to 1e-5 to 5e-5 and the layer output to 1.4e-4 |
| Golden test (tiny generated `qwen35` with Q4_K, Q5_K and Q6_K weights, partial rotary, 3 recurrent layers and 1 attention layer; independent pure-Python reference) | float math agrees to 5e-5 in logprob on every seed tried; the 8-bit path agrees to 5e-5 on the default seed on NEON, NEON without dot product and portable C. On other seeds and on a differently compiled build (the sanitizer build) it drifts on some prompts by up to 0.13 (mean up to 0.035): a rounding flip at one quantization point, carried by the recurrent state, which the double-precision reference does not make identically. So for this architecture the 8-bit path is checked with wider limits (mean 0.08, max 0.3) and the float path, strict at 1e-3, is the logic check |
| Chat protocol (`make chat-test`) | passes, including `reset` (which must zero the recurrent state) and cancel |

What this does not show: the logprob gaps (about 0.1) are not zero, and the exact cause is not isolated. After adding a `Q5_K` integer kernel, which made the first layer's projection match llama.cpp to 4e-8, the end-to-end gaps did not shrink, so they are not (only) from that. Candidates not ruled out: llama.cpp's f16 KV cache against this engine's f32, its chunked delta-rule algorithm for batched prompts, and accumulated 8-bit rounding differences over 32 layers. Per-layer hidden states drift away from llama.cpp's by a few percent of the largest shown value by layer 3 and stay there; judged as numerical noise, not proven. Speed is in section 4b.

## 4. Speed and memory (earlier; one run per setting, about 30% run-to-run spread)

OLMoE, 100-token job: this engine 9.6-9.9 s with the file in RAM against Colibri 15.4-16.9 s; with experts streamed from a cold disk under a memory budget, 481 MiB-1.4 GiB peak RSS and 15.5-26.6 s depending on the cache (docs/BENCHMARKS.md). Streaming is disk-bound at about 1.4 GB/s on this SSD.

**Qwen3-30B-A3B (17.3 GiB) on this 16 GiB machine, this engine only (partial).** A fresh cold copy of the model, `F_NOCACHE` reads, a one-token prompt and 48 greedy tokens, three runs per setting (`scripts/bench/qwen3_outofcore.py`):

| expert cache | tokens / second | s / token | cache hit rate | GB read / token | peak RSS |
|---|---|---|---|---|---|
| 2 GiB | 2.04-2.17 | 0.46-0.49 | 45.7% | 0.60 | 2.9 GiB |
| 4 GiB | 2.17-2.70 | 0.37-0.46 | 62.4% | 0.41 | 4.7-4.8 GiB |
| 8 GiB | 2.56-3.03 | 0.33-0.39 | 81.7% | 0.20 | 5.7-8.1 GiB |

So a model 1.08 times the size of RAM generates 2-3 tokens per second with a 3-8 GiB resident footprint. The runs improved a little from the first to the third at every setting (for example 0.46 to 0.37 s/token at 4 GiB), which I did not explain; it may be thermal or dense-weight page-cache warming, so treat the spread as the noise. **The llama.cpp comparison was not completed**: the benchmark was stopped at the user's request while llama.cpp was loading the file, so there is no measured speed ratio against llama.cpp memory-mapping, and no claim of one is made here.

## 4b. Speed of the 9B model on a 16 GiB M2 Air, and what moved it (measured 2026-11-03; one run per setting, browsers and other apps open throughout)

The machine was under memory pressure the whole time (the system compressor held 8-11 GiB of other programs' memory), which is a normal state for a laptop and matters more than any kernel.

| Change | Decode | Prompt processing | Note |
|---|---|---|---|
| Starting point (memory-mapped, one token per pass) | 0.13 tok/s (7.35 s/token) | the same per token | the 5.2 GiB file was evicted and re-read from the SSD on every token; the profile showed 159 of 206 s in the weight products and 35 s in the output layer, 0.7 s in the DeltaNet recurrence |
| Weights pinned in memory (`--lock`, `mlock`) | 6.3 tok/s (0.16 s/token) | 8.2 tok/s | the 40-fold step: the operating system now compresses or swaps other programs instead of the model. This is what a GPU runtime gets by wiring its buffers |
| + vectorized 5- and 6-bit weight unpacking, one horizontal add per block instead of per 16 elements | 5.5 -> 8.0 tok/s | 8.6 -> 18.3 tok/s | integer arithmetic is exact in any order: golden tests, 41,811 unit checks and greedy text unchanged |
| + prompts read in batches of 32 tokens, each weight block unpacked once for the whole batch | | (included above) | the same tokens as feeding one at a time, checked on prompts of 31 to 50 tokens |
| + scale/min unpacking with word operations | 8.3 tok/s | 19.0 tok/s | checked equal to the old unpacking on 400,000 random values |

Decode now streams about 43 GB/s of weights, which is close to what this chip's memory delivers, so further CPU gains on decode are bounded. llama.cpp with Metal on the same machine measured 13.4 tok/s on this model in the earlier test (a different quantization pass, not re-run now), so the CPU engine is at roughly 60% of that. Prompt processing at 19 tok/s is compute-bound and could still gain from 2x2 int8 matrix instructions (i8mm), which are not used. In the terminal chat: first reply 4 s including a 63-token prompt, second turn 1 s.

## 5. Bugs the testing found (all fixed)

1. A corrupted tensor shape in a model file made the forward pass write past a buffer (found by fuzzing).
2. `logits` JSON could contain invalid UTF-8 for half-character tokens (found by the golden test).
3. An earlier "no slowdown from streaming" result was wrong: the mapped run warmed the OS cache, so the streamed runs read RAM (found by an impossible 11 GB/s). Docs corrected.
4. The routing-comparison script mis-aligned tokens on prompts longer than six (llama.cpp elides middle rows), briefly showing a false 61.5% (fixed; true value 96.7%).

## 6. Not tested or not proven

- The llama.cpp half of the Qwen3 out-of-core benchmark (stopped), and any comparison with Colibri on Qwen3-30B-A3B (Colibri lists only the sibling Qwen3-Coder-30B-A3B in its own int4 format, so a same-model comparison is not possible).
- The AVX2 kernels beyond the CI unit test; Windows (the native engine does not support it); other architectures (only olmoe and qwen3moe have a forward pass); non-ASCII letter/number classes in the tokenizers are approximations (the 24-string sets include accents, CJK, Cyrillic and emoji and pass).
- The last 5 local commits have not been through CI (not pushed).
- Perplexity against llama.cpp on Qwen3; behaviour past a few hundred tokens of context.
