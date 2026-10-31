# Scorecard

How Clutor rates against its own bar and against Colibri and AirLLM (docs/BENCHMARKS.md), kept in the repo so the ratings can be argued with. A score is only raised when the evidence column can be re-run or inspected; anything that needs other people (users, contributors) or a measurement not yet taken is marked as such and is not scored.

Scale: 10 is "nothing material left to ask for". 9.5 is the project's target on every row; it is not reached yet, and the last column says what it would take.

| Criterion | Now | Evidence (re-runnable) | What 9.5 needs |
|---|---|---|---|
| Model coverage | 2 | One architecture runs end to end (`olmoe`). `docs/MODELS.md` | At least five architectures with a real forward pass, each checked against llama.cpp: Qwen3-MoE next, then DeepSeek-style, Mixtral-style and a dense model |
| Speed on the one model | 8 | 9.6-9.9 s for the 100-token job against 15.4-16.9 s for Colibri (docs/BENCHMARKS.md) | Same comparison on 3+ models and a second machine; router-ahead prefetch measured |
| Memory efficiency | 9 | Runs from 481 MiB to 1.4 GiB RSS on a 3.9 GiB model | Held on a model larger than RAM |
| Larger-than-RAM models | not scored | Not run. `docs/ROADMAP.md` milestone 2 | A 17 GiB class MoE run end to end with peak RSS and tokens/second reported against llama.cpp mmap on the same machine, 3 runs each |
| Correctness checking | 9 | `make -C c golden-paths` (independent pure-Python forward pass; float math agrees to 1e-4 on every kernel path), `make -C c verify-llama MODEL=...` (tokens, logprobs, routing against llama.cpp), 3,600 fuzz runs clean | The same two checks run in CI on Linux x86 (the AVX2 path has only been cross-compiled, never executed, until CI runs) and on every added architecture |
| Honest reporting | 9 | Every figure is labelled measured / computed / assumed / simulated; earlier wrong numbers were corrected in the docs, with the reason | Keep it: a third party reproducing a table from `scripts/bench/` |
| Install and ease of use | 5 | `npm i -g @clutor/cli`, `clutor engine install` (prebuilt, SHA-256 checked; exercised only against a local server until a release exists) or `clutor engine build`, `clutor generate` / `clutor quality`; a model must be downloaded and placed by hand | Prebuilt binaries for macOS and Linux with checksums, `clutor pull` for the supported models, a documented one-command install |
| Maturity | 5 | CI on Linux and macOS (unit tests, golden test, sanitizers, fuzzing, dependency audit), a release workflow that skips already-published versions, SECURITY / CONTRIBUTING / issue templates, 260 TypeScript and 41,811 C checks | Releases cut and installable (0.2.0 not published yet), prebuilt binaries, CI green on the new jobs on a real push, a stated compatibility and deprecation policy, a second maintainer able to release |
| Community | not scored | Needs people other than the author: outside users, issues, contributors, independent reproductions of the benchmarks | Outside contributors merged, benchmark results submitted from other hardware, issues answered within a stated time. `docs/ROADMAP.md` lists what is built to invite them; none of it counts until others act |

Unweighted mean of the rows that are scored: **about 6.4/10** (2, 8, 9, 9, 9, 5, 5). Two rows are deliberately not scored. Reaching 9.5 means every scored row at 9.5 and the two unscored rows measured, which today is not true and cannot be made true by the author alone.
