# Architecture

How the pieces fit, and where each one lives. This page describes what exists; the status of each part and where it is tested is in [CAPABILITIES.md](CAPABILITIES.md), and what is only an idea is in [DIRECTIONS.md](DIRECTIONS.md).

```
                          the person
              ┌─────────────┴───────────────┐
        desktop app                    command line / local server
   (React UI + Tauri/Rust host)          (TypeScript: packages/cli)
              │                                   │
              │  JSON lines over a pipe           │  JSON lines, or llama-server over HTTP
              ▼                                   ▼
        ┌───────────────────── clutor-engine (C, c/) ─────────────────────┐
        │ GGUF reader · tokenizer · forward pass · expert streaming       │
        │ cache + prefetch · K-quant / IQ kernels · backend seam (CPU)    │
        └──────────────┬─────────────────────────────┬────────────────────┘
                       │ memory-mapped weights        │ pread of experts
                       ▼                              ▼
                      RAM                       the model file on SSD
```

## The runtime: `c/`

A C11 program with no dependencies beyond libc, libm and pthreads. It reads GGUF files directly (no conversion, no second copy). Main parts:

| Part | Files | What it does |
|---|---|---|
| Model file | `gguf.c`, `meta.c` | Parses the header, finds every tensor and expert |
| Tokenizer | `tokenizer.c` | Tokenizes and detokenizes; checked against llama.cpp |
| Forward pass | `forward.c`, `route.c`, `vision.c` | Attention (or Gated DeltaNet), the router, the experts, the picture encoder; seven text architectures |
| Kernels | `kernel.c`, `quant.c`, `iq.c` | Integer dot products for K-quant and IQ weights: NEON, AVX2, AVX-512 VNNI where the processor has it, chosen when the program starts (`CLUTOR_ISA` forces a lower one) |
| Expert streaming | `xs.c`, `stream.c`, `cache.c` | Reads the experts a token needs from disk with `pread` through an I/O thread pool and a cache (LRU, W-TinyLFU or adaptive); prefetches the next layer's experts while this one computes |
| Hardware seam | `backend.h` | Every matrix product goes through `cl_matvec_batch`; a backend can take the products it supports (`CLUTOR_BACKEND`). Only the CPU backend exists |
| Chat protocol | `chat.c` | A long-lived process: one JSON object per line in, events out (`ready`, `appended`, `t`, `done`, `progress`, `error`). Both the app and the terminal chat use it |
| Benchmarks and checks | `bench.c`, `main.c` | `stream`, `diskbench`, `generate`, `logits`, the golden tests |

How memory is used: a model that fits is memory-mapped (and optionally pinned with `--lock`). A mixture-of-experts model that does not fit keeps the always-used weights resident and streams the experts through the cache. A dense model bigger than memory is not helped by streaming: every token reads every weight (the engine pins as many whole layers as it can, which is described but not measured under real pressure; see [LIMITS.md](LIMITS.md)). The design, with the measurements, is in [NATIVE_ENGINE.md](NATIVE_ENGINE.md).

## Planner and the rest of the TypeScript: `packages/core`, `packages/cli`

- **`packages/core`**: a dependency-free GGUF reader, the memory planner (`planner/plan.ts`), the plan report with its labelled numbers (`planner/report.ts`), the benchmark record format (`planner/record.ts`), the cache-policy lab, the coding agent (loop, tools, shadow workspace, diffs, checkpoints, signed receipts), the command policy, and the Node-only parts (`node/`: scratch runner, OS sandbox, policy file, host capabilities).
- **`packages/cli`**: the `clutor` command (`plan`, `inspect`, `bench`, `doctor`, `code`, `serve`, `receipts`, `rollback`, ...), the terminal UI, the engine driver and the local HTTP server.

The planner is arithmetic from the model's own file and the numbers you give it (RAM, disk speed). Every figure in its report is labelled measured, estimated or theoretical ([ENGINE.md](ENGINE.md)).

## The desktop app: `app/`

- **`app/src`**: the React interface, the agent controller (approvals, "ask first" / "auto-run checks" / "plan only"), the job queue for background questions, the MCP client, the model library and the home screen.
- **`app/src-tauri`**: the Rust host. It starts the engine as a child process and relays its JSON lines, enforces the project-folder jail for file reads and writes, runs agent commands in a scratch copy (inside `sandbox-exec` on macOS), reads the policy file, makes web requests, and runs the image engine (stable-diffusion.cpp, built from `app/src-tauri/image`).
- The app and the command line share the planner's speed estimate (`estimateTps` in core) and read the same `clutor.policy.json`.

## What crosses a boundary

| From | To | What | Guarded by |
|---|---|---|---|
| UI | Rust host | Tauri commands (read, write, run, scan models) | The folder jail; the host refuses paths outside the project |
| Rust host | engine | JSON lines on a pipe | Local process only |
| Agent | project files | Changes held in a shadow copy until you apply them | Diff review, checkpoints, receipts |
| Agent | commands | A short allow-list, run in a scratch copy | Your approval for anything that runs project code or uses the network; the OS sandbox where it exists ([SECURITY_MODEL.md](SECURITY_MODEL.md)) |
| Chat with browse on | the internet | The search text and the pages fetched | An explicit switch in the composer |
| Cloud models (optional) | a provider you add | The conversation | Only when you pick a cloud model |

Nothing leaves the machine unless you turn on browsing or pick a cloud model.
