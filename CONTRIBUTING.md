# Contributing

Bug reports with a way to reproduce them, and small focused pull requests, are the most useful.

## Setup

Node.js 20 or newer and a C compiler (`cc`; Xcode command line tools on macOS) for the native engine.

```bash
git clone https://github.com/isagarpinninti/clutor.git && cd clutor
npm install
npm run build
```

## The checks

```bash
npm run typecheck   # tsc for core and cli
npm run lint
npm run engine      # compile c/
npm run engine:test # C unit tests
npm run engine:golden # the engine against an independent pure-Python forward pass (float and integer paths, every kernel this machine compiles)
npm run engine:fuzz   # corrupted model files under ASan + UBSan: errors, never crashes or hangs
npm test            # core, cli (includes the engine integration tests)
npm run bench       # hot-path budgets
npm run smoke       # runs the built CLI end to end
make -C c sanitize SMALL=/path/to/moe.gguf   # ASan, UBSan, TSan
npm run gates       # everything above, in order
```

## Layout

| Path | What |
| --- | --- |
| `packages/core` | The engine. The main entry has no `node:*` imports (it runs in a browser); `@clutor/core/node` adds files, processes and hardware probing |
| `packages/cli` | The `clutor` command, the local server, the llama-server supervisor |
| `c/` | The native engine (C11): GGUF reader, cache policy, router, streamer. `c/test` has its unit tests |
| `scripts/` | Bench and smoke scripts |

## Rules of the house

- **Numbers on screen say what they are.** Measured, read from a file, computed from published figures, assumed, or simulated: label it, and do not present one as another. A new claim about speed or quality needs a measurement behind it, or a clear label that it is a ceiling or a model.
- **A security behaviour comes with a test that tries to break it**, at the lowest level that can show it: the pure function, the HTTP surface, then the built CLI.
- **Untrusted by default.** The model's replies, file contents and command output are data. Nothing a file says may widen what the agent is allowed to do.
- **Refusals say why.** Return a reason the user can act on.
- **No `any`, no `eval`, no TODOs** left in the tree.
- **A change to the forward pass or a kernel keeps `npm run engine:golden` passing**, and a change to `c/src/gguf.c`, `meta.c` or `tokenizer.c` keeps `npm run engine:fuzz` clean. Neither needs a model download.
- **C code is sanitizer-clean.** Run `make -C c sanitize` before sending a change to `c/`; a new concurrency path needs a ThreadSanitizer run.
- **Do not add a dependency to `@clutor/core`.** It has none, deliberately.

## Pull requests

Say what changed and why, and mention anything you could not test (a Linux or Windows change you could not run is worth saying).
