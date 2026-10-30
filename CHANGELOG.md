# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow [Semantic Versioning](https://semver.org/).

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
