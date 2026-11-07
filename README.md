<p align="center">
  <img src="assets/clutor-logo.svg" width="520" alt="Clutor">
</p>

<h3 align="center">Run large language models on your own machine, with honest numbers.</h3>

<p align="center">
  <a href="https://github.com/isagarpinninti/clutor/actions/workflows/ci.yml"><img src="https://github.com/isagarpinninti/clutor/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.npmjs.com/package/@clutor/cli"><img src="https://img.shields.io/npm/v/@clutor/cli?color=2ea043" alt="npm version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-1f6feb" alt="MIT licence"></a>
</p>

<p align="center">
  <a href="#the-app"><b>App</b></a> ·
  <a href="#the-idea"><b>The idea</b></a> ·
  <a href="#install-and-run"><b>Install</b></a> ·
  <a href="#supported-models"><b>Models</b></a> ·
  <a href="#command-line"><b>CLI</b></a> ·
  <a href="#honest-status"><b>Status</b></a> ·
  <a href="#documentation"><b>Docs</b></a>
</p>

---

Clutor is a local AI workbench built around its own inference engine. It has three parts:

| Part | What it is |
| --- | --- |
| **Engine** (`c/`) | Plain C with no dependencies. Reads GGUF files directly, runs the forward pass, and for mixture-of-experts models streams only the experts a token needs from disk through an adaptive cache with prefetch. |
| **App** (`app/`) | A desktop app (Tauri 2: Rust shell, system web view, React) with **Chat**, a **Code** agent, background jobs and MCP tools. Dark only, no Electron. |
| **Planner, CLI and agent** (`packages/`) | TypeScript. Answers "will this model fit, and how fast could the disk possibly let it go", and provides the terminal UI and the sandboxed coding agent. |

## The app

<table>
<tr><td width="50%">

**Chat**
- Streams replies from the model you load, on Clutor's own engine
- **Background jobs**: hand a slow question to a big model and come back to the answer; jobs survive a restart and resume where they stopped
- Think mode for reasoning models, with the thought process collapsible
- **Browse** (on by default, can be turned off): questions are looked up on the web first; the answer comes with numbered sources
- Model picker under the message box, context ring in the top bar

</td><td width="50%">

**Code**
- Pick a project folder and describe a change
- The agent edits a **private copy**, can run your tests there (asking first), then shows an exact diff
- **Apply** is refused if a file changed in the meantime; every apply keeps a checkpoint, so **Undo** works
- Side panel with **Terminal**, **Browser**, **Changes** and **Files**
- Permission modes: *Ask first*, *Auto-run checks*, *Plan only*
- **Web tools** for the agent (search and read pages), asked once per run
- **MCP client**: add stdio MCP servers for extra tools; the agent asks before calling any tool the server does not mark read-only

</td></tr>
<tr><td>

**Usage, not billing**
Context fill, tokens written, measured speed, model size against your RAM, whether the model is pinned in memory. All local: no quota, no cost.

</td><td>

**Designed to stay out of the way**
Nothing loads at launch (pinning several GB at once stalls a computer). Models larger than 40% of RAM are not pinned. Everything stays on the device.

</td></tr>
</table>

The app runs on macOS today. The same code is set up for Windows, Linux and Android; see [Honest status](#honest-status) for what has and has not been tested.

## The idea

A mixture-of-experts (MoE) model is huge on disk and small per token: only a few experts are active for each token. The always-needed part stays in RAM and the rest is read from disk when needed. That is how a model larger than memory can run at all, and Clutor exists to answer, with measurements, the question that "run a huge model on a laptop" claims skip: **how fast can the disk possibly let this go, and what does the cache actually buy?**

Rough physics, computed, not measured (disk assumed at 3 GB/s):

| Model | Read per token | Disk-bound ceiling |
| --- | --- | --- |
| Dense 100B at Q4 | about 56 GB | about 19 s per token |
| MoE with about 5B active parameters | about 3 GB | about 1 s per token |
| 671B MoE with 37B active | about 20 GB | about 7 s per token |

So large MoE models are usable for slow, background work, and dense models of that size are not. The app's background jobs exist for exactly that case. Run `clutor plan <model> --ram N` for the arithmetic on your machine, and see [docs/ENGINE.md](docs/ENGINE.md).

## Install and run

### The desktop app (from source)

Requires Node.js 20+, a Rust toolchain and a C compiler.

```bash
git clone https://github.com/isagarpinninti/clutor.git && cd clutor
npm install
cd app && npm install
npm run tauri dev                  # run it
npx tauri build --bundles app      # macOS: builds src-tauri/target/release/bundle/macos/Clutor.app
```

Put `.gguf` files in the app's models folder (shown in the app; override with `CLUTOR_MODELS=/path`), or use **Get a model** in the picker. The build is unsigned, so macOS may ask you to right-click the app and choose Open the first time.

### The command line

```bash
npm install -g @clutor/cli
clutor engine install     # prebuilt engine for macOS or Linux, SHA-256 checked (or: clutor engine build)
clutor                    # interactive terminal session
```

From a checkout: `npm install && npm run build && npm run engine`.

## Supported models

The native engine runs a real forward pass, checked against llama.cpp, for these architectures (Q4_K_M and similar quantizations):

| Family | Type | Notes |
| --- | --- | --- |
| OLMoE (`olmoe`) | MoE | experts streamed from disk |
| Qwen3-MoE (`qwen3moe`) | MoE | experts streamed from disk |
| Granite 3.0 MoE (`granitemoe`) | MoE | experts streamed from disk |
| Llama 3.x (`llama`) | dense | whole model read per token |
| Qwen2.5 (`qwen2`) | dense | |
| Qwen3.5 (`qwen35`) | hybrid dense (Gated DeltaNet + attention) | thinking models supported |

DeepSeek, GLM, Kimi and Mixtral are known to the planner by their published shapes, but the engine has never opened one. Details and the exact agreement figures per model are in [docs/MODELS.md](docs/MODELS.md).

## Command line

| Command | What it does |
| --- | --- |
| `clutor` | Interactive terminal chat and agent (`/help` lists commands, `@file` attaches a file) |
| `plan <model>` | Memory plan from a catalog id or a real `.gguf` header: does it fit, where the memory goes, the disk-bound speed ceiling |
| `generate <file.gguf> "<prompt>"` | Native engine, real forward pass; `--cache-mb N` streams the experts from disk |
| `quality <file.gguf> <text>` | Perplexity and agreement of streamed or smart-routed runs against exact routing |
| `stream`, `bench` | Measure expert streaming and compare cache strategies on a file |
| `code "<task>"` | The coding agent in the terminal; review the diff, then `--apply` |
| `serve` | Local OpenAI / Ollama / MCP-compatible server |
| `pull`, `models`, `inspect`, `doctor`, `receipts`, `rollback` | Housekeeping, signed receipts, undo |

## How the engine streams experts

For every layer of every token the engine routes (top-k experts), looks the experts up in an adaptive W-TinyLFU cache, prefetches the next layer's likely experts with a pool of `pread` workers, overlaps that I/O with compute, and optionally routes to a cached expert when one is nearly as probable (bounded so no probability moves by more than a set amount). Pinning the model in memory (`--lock`) is a large speed-up for models that fit. Measured cache and I/O results, with what is measured, computed, simulated and assumed labelled on every number, are in [docs/STREAMING_RESULTS.md](docs/STREAMING_RESULTS.md).

## Honest status

What was verified, and what was not:

- **Verified:** the engine matches llama.cpp on tokenization and top tokens for the architectures above; the engine builds and passes its chat-protocol tests on Linux, macOS and Windows (MinGW) in CI; the app's agent, apply/undo, MCP client, job queue and usage logic are covered by unit tests, and the Rust file jail and command runner by Rust tests.
- **Only on macOS arm64:** all performance numbers and the desktop app itself.
- **Not tested by the author:** the app on Windows, Linux and Android (no Android build yet); MCP HTTP transport (stdio only); the AVX2 kernels beyond a CI unit test.
- **Limits of the agent:** it is only as reliable as the model behind it. Small models often make malformed tool calls; those are rejected, and every change is shown to you before anything is written. The command sandbox isolates the project and environment, **not the machine**: a command you approve can still read files your user can read.
- **Not claimed:** that a huge model runs fast. The planner prints the real ceiling for your disk.

See [docs/LIMITS.md](docs/LIMITS.md) and [docs/VISION_AUDIT.md](docs/VISION_AUDIT.md).

## Repository layout

```
c/                 native engine: GGUF parser, forward pass, I/O pool, cache, prefetch, kernels, chat library
app/               desktop app: src/ (React UI, agent, MCP, jobs), src-tauri/ (Rust: engine host, file jail, command runner)
packages/core/     planner, catalog, gates, agent core, receipts, server
packages/cli/      the `clutor` command, terminal UI, engine driver
docs/              guides, engine design, models, limits, results
examples/          sample clutor.config.json
```

## Documentation

| | |
| --- | --- |
| [Guide](docs/GUIDE.md) | Install, configure, connect an editor, plan a model, use the agent |
| [Native engine](docs/NATIVE_ENGINE.md) | The C engine: design, chat mode, cache policy, limits |
| [Models](docs/MODELS.md) | Which models run, and how they were checked |
| [Streaming results](docs/STREAMING_RESULTS.md) | Measured cache, I/O and stress-test results |
| [The agent](docs/AGENT.md) | Tools, jail, command policy, approvals, receipts |
| [Planner](docs/ENGINE.md) | The arithmetic and what is measured versus modelled |
| [Security model](docs/SECURITY_MODEL.md) | Who can talk to the server and how |
| [Limits](docs/LIMITS.md) | What Clutor will not do |
| [Changelog](CHANGELOG.md) | What changed |

## Development

```bash
npm install
npm run gates                      # typecheck, lint, engine build and tests, unit tests, build, bench, smoke
cd app && npm test && cargo test --manifest-path src-tauri/Cargo.toml --lib
```

MIT licensed. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
