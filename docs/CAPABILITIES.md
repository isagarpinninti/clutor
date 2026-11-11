# What works where

Each cell says how it is known. **CI** means a GitHub Actions job runs it on every change (`.github/workflows/ci.yml`). **Author** means the author ran it by hand, on one Apple M2 with 16 GiB. **Built** means it compiles there and nothing more. **No** means not supported. **Not run** means not tested by anyone we know of. A blank claim is better than a guess, so a cell that says "not run" is a gap, not a hidden pass.

| | macOS (Apple silicon) | Linux (x86-64) | Windows (x86-64) |
|---|---|---|---|
| Engine builds and its unit checks pass | CI, author | CI | CI (MinGW-w64) |
| Engine output equals an independent reference (golden test, kernel paths for that CPU: NEON / AVX2) | CI | CI | CI |
| Chat protocol, batched prompt path, automatic streaming of a model larger than memory (on small generated or fixture models) | CI | CI | chat protocol: CI; the other two: not run |
| Engine under ASan, UBSan, TSan | not run | CI | not run |
| Real model, real text (Llama 3.2 1B, Qwen3 30B-A3B, Qwen3-VL 30B) | author | not run | not run |
| Speed numbers in this repository | author (one machine) | none | none |
| Expert streaming from SSD, timed on a model larger than memory | author | not run | not run |
| CPU kernels | NEON + dot product, author and CI | AVX2: CI correctness only, never timed | not timed |
| GPU, NPU | **No.** The hardware is detected (`clutor doctor`) and not used; the backend seam exists (`c/src/backend.h`) with no GPU backend behind it | **No** | **No** |
| Planner (`clutor plan`), inspect, simulate | CI (Node 20, 22) | CI (Node 20, 22) | not run (the Node suite does not run on Windows) |
| `clutor doctor`, `clutor bench`, `clutor models` (command-line tests; `bench` also runs as a CI step) | CI, author | CI | not run |
| Coding agent: file jail, diffs, checkpoints, rollback, receipts (unit tests) | CI, author | CI | not run |
| Agent commands in an OS sandbox | seatbelt: CI and author | bubblewrap: CI (needs `bwrap` installed; found and fixed a real bug on its first CI run) | **No.** Policy, scratch copy and your approval only |
| Desktop app | built in CI; used by the author | built in CI; never run | built in CI; never run |
| Picture input (Qwen3-VL) | author | not run | not run |
| Image generation | author (one model: Qwen-Image-2.1 Turbo) | built, never run | built, never run |
| MCP | stdio: unit tests (CI) | stdio: unit tests (CI) | not run; HTTP transport not implemented |
| Local server (`clutor serve`) | CI (smoke) | CI (smoke) | not run |

## How to read the gaps

- The engine's **correctness** is checked on all three systems by comparing it against an independent implementation. Its **speed** is known on one machine only.
- Anything under a platform that says "not run" may well work. The point is that nobody has shown it, and Clutor will not say it does.
- To add a row, run `clutor bench --file MODEL.gguf --record benchmarks/<machine>/<model>.json` and open a pull request with the file (see `benchmarks/README.md`).

## Maturity of the main parts

| Part | Status |
|---|---|
| Out-of-core mixture-of-experts engine (cache, prefetch, forward pass for seven architectures) | Implemented; speed measured on one machine |
| Planner and its report (`clutor plan`) | Implemented; speed estimate rests on one fitted measurement (see the report's confidence) |
| Benchmark record and the llama.cpp comparison script | Implemented; one machine |
| Coding agent, receipts, rollback | Implemented |
| OS sandbox for agent commands | Implemented on macOS and Linux (bubblewrap); not on Windows |
| Policy file (`clutor.policy.json`) | Implemented in the command line and the desktop app (the app's agent commands; `requireSandbox` can only be satisfied on macOS, because the app has no Linux or Windows sandbox) |
| Hardware backend seam | Implemented (registry and test); no accelerator backend |
| Multi-machine inference | Design only (`docs/DIRECTIONS.md`) |
