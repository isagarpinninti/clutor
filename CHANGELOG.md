# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

### Added
- **A real forward pass for `olmoe` models in the native engine**: byte-level BPE tokenizer, quantized weight readers (Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, F16, BF16, F32), attention, router and experts. New engine commands `tokenize`, `logits`, `generate`. Checked against llama.cpp on OLMoE-1B-7B Q4_K_M: identical token ids on 10 test strings, same top-1 token on 6 prompts, 96.5-98% of visible router choices inside this engine's top-8 (see docs/NATIVE_ENGINE.md for the numbers and limits). Maps the whole file unless `--cache-mb` is given.
- **Experts streamed inside the real forward pass**: `generate --cache-mb N` reads each layer's experts from disk through the cache so resident memory is bounded by the cache. On OLMoE-1B-7B Q4_K_M: output byte-identical to the mapped run, peak RSS 544 MiB to 1.4 GiB against 3.7 GiB, but 2.3x to 5x slower than mapped because the run is disk-bound (about 1.4 GB/s on this SSD). Previous-token prefetch is optional and off by default: it raised hit rates but lost on time in every measurement. One run per setting, about 30% run-to-run spread; see docs/NATIVE_ENGINE.md.
- **SMART-ROUTE inside the real forward pass** (`--smart J,M,TAU` with `--cache-mb`) and a `quality` command (perplexity, KL divergence and top-1 agreement against exact routing). On OLMoE-1B-7B over 799 tokens of project text: mild settings read 12-14% fewer bytes per token for 2-3% higher perplexity and a different top-1 token at 6-10% of positions; unbounded cache-aware routing reads 25% fewer bytes but raises perplexity 2.7x.
- **Verification that needs no model**: `make -C c golden` / `golden-paths` generate a tiny runnable OLMoE (random Q4_K / Q6_K weights) and check the engine against an independent pure-Python forward pass; in exact float mode the two agree to 1e-4 on every kernel path, and the 8-bit integer path stays within a stated tolerance. `make -C c fuzz` corrupts model files thousands of ways under AddressSanitizer + UBSan. Both run in CI.
- **Qwen3-MoE** (Qwen3-30B-A3B family) in the forward pass: grouped-query attention, per-head QK-norm, renormalised expert weights, Qwen's pre-tokenizer (identical to llama.cpp on 24 strings). Checked against an independent reference on a tiny generated model and against llama.cpp on the real 17.3 GiB model: identical tokenization on 24 strings, 96.7-98.0% router agreement on three prompts, and identical 24-token greedy continuations on three prompts (token logprobs differ more than on OLMoE, which looks like numerical noise amplified by 48 layers; see docs/TEST_REPORT.md). Runs a model larger than RAM at 2-3 tokens/s with a 3-8 GiB footprint; the speed comparison against llama.cpp memory-mapping was not completed. The engine now refuses to map a file larger than 80% of RAM and asks for `--cache-mb`.
- `--prefetch ahead` (router-ahead predictor: 82% of its picks are chosen against 37% for `prev`) — accurate, but it does not cut time on a bandwidth-bound run, so prefetch stays off by default.
- **Fixed by that testing**: a corrupted tensor shape in a model file could make the forward pass write past a buffer (every tensor's shape and the hyper-parameters are now validated when the model opens); `logits` JSON could contain invalid UTF-8 for tokens that are half a character.
- **`clutor generate` and `clutor quality`** expose the real forward pass and the quality comparison from the main CLI, with live output; AVX2 integer kernels for Q4_K / Q6_K (compiled for x86_64, not yet executed by the author; CI will run them); `CLUTOR_FLOAT_MATH=1` runs the exact float path for diagnosis.
- **Prebuilt engine binaries**: `.github/workflows/release-binaries.yml` builds the engine on macOS arm64 / x86_64 and Linux x86_64 / arm64 on a version tag, runs the golden test on each binary, and attaches tarballs plus SHA256SUMS to the release; `clutor engine install` downloads the one for this machine, refuses it unless the SHA-256 matches, and installs it to `~/.clutor/bin` (tested against a local server; no real release exists yet).
- **`clutor pull`**: downloads now resume after an interruption (Range request, the `.part` file is kept) and are verified against the SHA-256 Hugging Face publishes for the file, deleting it on a mismatch; the OLMoE entry points at the exact file the engine was validated on, and Qwen3-30B-A3B (17.3 GiB, larger than 16 GiB of RAM) is listed.
- **Release workflow** skips a package whose version is already on npm instead of failing the run.
- docs/SCORECARD.md, docs/ROADMAP.md, docs/MODELS.md.
- **Faster compute**: integer dot products for Q4_K/Q6_K weights against 8-bit activations (NEON dot-product instruction), a persistent spinning thread pool and batched expert products take the forward pass from 0.30 to 0.03 s per token with the file in RAM, and bring the router closer to llama.cpp (99.8% of visible choices against an f32 KV cache).
- `generate --trace FILE` records the model's real router scores; `clutor stream --trace FILE` replays them through the cache, prefetch and smart-route code, and the report says `routing: real`. On one 130-token trace the old synthetic default overstated the hit rate (77% against 47%).

### Changed
- The interactive session opens with a large CLUTOR wordmark, the version, folder and model (colour only on a terminal, never with `NO_COLOR`; a one-line title on narrow terminals).
- README restructured around a logo (`assets/clutor-logo.svg`), badges and a sample session; no claims changed.

## 0.1.1

### Changed
- **`@clutor/cli` now ships the native engine's C source.** After `npm install -g @clutor/cli`, `clutor engine build` compiles it (needs a C compiler) into `~/.clutor/bin/clutor-engine`, where `stream`, `bench` and `/cache` find it. Nothing is compiled during `npm install`. A source checkout still builds into `c/build`.
- `@clutor/core` is republished at 0.1.1 with no code changes so both packages share a version.

## 0.1.0

First release. Terminal only: there is no desktop app.

### Added
- **Interactive session**: `clutor` with no command in a terminal opens a prompt like Claude Code. Plain text is a task for the sandboxed agent (diff, approval, receipt as before); `/` commands cover `/help /status /doctor /model /models /plan /simulate /inspect /bench /stream /pull /init /backend /ledger /decide /verify /yes /undo /tools /cost /history /clear /layers /cache /exit`. The model server starts once and stays up between tasks; earlier tasks are summarised to the next one; history is kept in `~/.clutor/history`; Ctrl-D leaves. `/cost` shows tokens reported by the model server, nothing estimated.
- **Native engine (`c/`)**: C11, no dependencies. GGUF v2/v3 reader; two-slot dense ring; expert streamer (`pread`, I/O thread pool, demand-before-prefetch); LRU, W-TinyLFU (4-bit count-min sketch, SLRU) and adaptive-window cache; prefetch staging pool; SMART-ROUTE (J, M, tau). Measures I/O, waits, hit rates and prefetch behaviour on a synthetic router. Does not run a transformer yet.
- **`@clutor/core`**: GGUF reader, memory planner and catalog (now including OLMoE-1B-7B and Qwen2.5-Coder-7B), cache simulator, drafting, the agent (strict tools, path jail, no-shell command policy, shadow workspace, diffs, apply/rollback), safety gates `CL-101..108`, ECDSA P-256 signed receipts.
- **`@clutor/cli`**: `clutor` and `clt` with `init`, `status`, `models`, `pull`, `inspect`, `plan`, `simulate`, `stream`, `bench`, `engine`, `chat`, `decide`, `code`, `serve`, `backend`, `receipts`, `ledger`, `receipt`, `rollback`, `doctor`; local OpenAI/Ollama/MCP server on :4320 (`POST /v1/decide` returns first-token choice probabilities and a signed receipt); llama-server supervision; JSON-schema-constrained tool calls and KV-prefix snapshots against llama-server.
- Tests: 158 core, 64 cli (including the C engine cross-checked against the TypeScript parser), 40,212 C unit checks, sanitizer runs, a smoke script.

### Changed
- `clutor code --verify` retries: a failing check is fed back (exit code and exact output, plus the task and how to read an assertion) for up to 3 self-healing turns; the final state must pass before `--apply` writes anything. The command typed after `--verify` runs without a prompt (it is the user's own command); one that only comes from the config file still asks. A verify command that was not approved is reported as "NOT RUN", not as a failure of the code.
- The agent's system prompt carries the project's module type and the `node:test` imports when no test framework is installed; its conversation limit follows the configured context size instead of a fixed 60,000 characters.
- `clutor plan --ram N` plans for an N GiB machine; the OS headroom is 22% of RAM clamped to 1.8-3.5 GiB. The plan labels the runtime line "Double-Buffer DMA (runtime buffers)".
- The engine's compute step is a Q4_0 x Q8_0 SIMD kernel (`--compute kernel|checksum`).
- `clutor bench` adds rows for scan-pollution rejects, prefetch hit/late, lookahead overlap and disk wait vs compute time.
- Receipts moved from HMAC to ECDSA P-256 with the public key inside the signed body, verifiable offline.
- The agent's first message is now the project map then the task, so tasks on a project share a cacheable prefix.

### Fixed
- **Engine memory**: expert buffers are packed in one arena at a 4 KiB stride instead of one page-rounded allocation each (16 KiB pages on Apple silicon made every slot about 5% too big), and the cache size is now budget / stride, so the cache never exceeds `--cache-mb`. Measured peak RSS = budget + staging + dense weights + about 3 MiB, flat across tokens.
- `clutor chat`, `decide`, `code` and `serve` failed to start llama-server on a machine without `~/.clutor/slots`; the folder is now created (mode 0700) first.

### Removed
- The Electron desktop app, the React components and the Playwright suite.

### Known limits
See [docs/LIMITS.md](docs/LIMITS.md) and [docs/VISION_AUDIT.md](docs/VISION_AUDIT.md).
