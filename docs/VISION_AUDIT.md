# Vision audit: what Clutor matches, and what it does not claim

The original brief asked for a 100% terminal tool that beats Colibri and Ollama with a 744B model in 3.8 GB at 16 tok/s. This page says, item by item, what is built, what changed, and what is not claimed and why. Every number Clutor prints is labelled measured, computed, assumed or simulated.

## Matches the vision

| Item | Status |
| --- | --- |
| Terminal only (no Electron, React, GUI) | Done. The desktop app was removed (archived outside the repo). |
| `clutor` and `clt` binaries | Done. |
| `init`, `status`, `models`, `inspect`, `plan`, `simulate`, `bench`, `chat`, `code`, `decide`, `serve`, `ledger`, `receipt verify`, `doctor`, `rollback` | Done. |
| Server on :4320: OpenAI `/v1/chat/completions`, `POST /v1/decide` (token-protected: it signs a receipt), Ollama `/api/chat`, `/api/generate`, `/api/tags`, MCP, agent API | Done (loopback only, Host/Origin checks, bearer token for file and process routes). |
| GGUF v2/v3 parser, expert byte-offset layout, real byte-range reads in `bench` | Done, tested on a real llama.cpp file. |
| Memory planner for 8/16/32 GiB hosts, dense-ring and streamed-expert strategies | Done, computed from the model's real tensor sizes. |
| W-TinyLFU (count-min sketch, window + SLRU) and an adaptive variant | Done, simulated on synthetic traces. |
| SMART-ROUTE: keep true top-J, substitute only resident experts within a probability bound | Done in the simulator. |
| Prompt-lookup drafting and expert coalescing | Done in the simulator. |
| Agent: repo map, read/write/edit, no-shell allowlisted commands, shadow workspace, unified diffs, apply with conflict checks, checkpoints, rollback | Done. |
| `CL-101..108` gates; `clt_` hash-chained receipts in `.clutor/receipts.jsonl` | Done. |
| `decide`: one forward pass, probability per choice | Done. First-token probability from the backend; not calibrated, and says so. |
| Smoke test and CI | `scripts/smoke.mjs`; CI workflow exists but has not run on GitHub yet. |

## Does not match, deliberately

| Brief | Why it is not built as written |
| --- | --- |
| 744B model in 3.8 GB at 16.4 tok/s | Not achievable: each token needs gigabytes read from disk. The planner reports the real ceiling (for example DeepSeek-V3 Q4 on 16 GiB: about 0.3 tok/s cold, about 5 at a 95% hit rate). Clutor will not print the 16.4 figure. |
| Pure TypeScript kernels, zero native code | Replaced by the decision to write a C engine (`c/`). Q4 matmuls in JavaScript would be 50 to 100 times slower than C. |
| Side-by-side "Colibri vs Clutor" table with Colibri's numbers | Colibri's numbers cannot be produced without running Colibri on the same machine. `simulate` compares baseline LRU against Clutor's policy inside one simulator and labels it simulated. |
| ECDSA P-256 receipts | **Done.** Public key and thumbprint are inside each signed receipt; `clutor receipt verify` works offline; `--key-id` / `--pubkey` pin a key. |
| `zod` as a peer dependency of core | Core does not use zod (it has its own validators), so declaring it would be false. The CLI uses it. |
| Agent tool named `verify_build` | Verification is a `--verify` option and a final check, not a tool the model calls. |
| "744B", "scan rejects 0", "96.8% hit" in the sample screen | Those were illustrative. Hit rates here come from simulation until the engine can measure real routing. |
| Lookahead L to L+1 prefetch on a real model | Built in the C engine and measured, but on a synthetic router: without a forward pass there is no hidden state to predict from. |
| `buildSyntheticMoeShardFile` | Exists as `writeSyntheticGguf`. |

## Done since the first audit

| Item | State |
| --- | --- |
| C engine: GGUF v2/v3 parser, dense weights resident or in a 2-slot ring (`--dense`), expert streamer with `pread` and an I/O thread pool | Done, cross-checked against the TypeScript parser, sanitizer-clean |
| W-TinyLFU (4-bit count-min sketch, SLRU) plus an adaptive window | Done; a fixed window lost to LRU on the test trace, so the adaptive one is the default |
| SMART-ROUTE (J, M, tau) in C, with the bound tested | Done |
| Lookahead prefetch into a staging pool ("Buffer B") on a background thread | Done, on a synthetic predictor |
| `clutor stream`, `bench` strategy comparison, `/layers`, `/cache` | Done, labelled measured / synthetic |
| ECDSA P-256 receipts, offline `receipt verify` | Done |
| JSON-schema-constrained tool calls and KV-snapshot save/restore through llama-server | Done and checked against a live llama-server (constrained: 8 of 8 replies valid; snapshot: saved, then restored, including for a different task on the same project). A speed-up from restoring has **not** been measured |
| Real downloadable test models under 4.8 GB | OLMoE-1B-7B Q4_K_M (3.92 GiB, real MoE) and Qwen2.5-Coder-7B Q4_K_M (4.36 GiB, dense); sizes checked on Hugging Face. Neither has been downloaded or run by the author yet |

## Still to build

The forward pass: attention, RoPE, norms and quantized matmuls (Q4_K, Q8_0 at least), a tokenizer and a KV cache, so the engine can generate text and be validated against llama.cpp's logits. Until then `chat`, `decide` and `code` need a llama-server or any OpenAI-compatible endpoint, and the engine's hit rates are for synthetic routing only.
