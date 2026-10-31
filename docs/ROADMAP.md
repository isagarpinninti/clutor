# Roadmap

Each milestone ends with something checkable. Dates are not promised; order is.

1. **Verified foundations (done).** Real OLMoE forward pass; golden test against an independent implementation; fuzzed file readers; CI jobs for both; `clutor generate` and `clutor quality`; benchmark write-up against Colibri and AirLLM.
2. **A model larger than RAM.** Qwen3-30B-A3B Q4_K_M (about 17 GiB on a 16 GiB machine): Qwen3-MoE forward pass, its tokenizer, parity with llama.cpp (top token on at least 5 of 6 prompts, at least 95% routing agreement), then an out-of-core run with a 4-8 GiB expert cache. Pre-registered success: peak RSS under 8 GiB; throughput reported whatever it is; a win over llama.cpp memory-mapped only if the median of three runs is at least 10% faster with non-overlapping ranges. Needs a 17 GiB download.
3. **Install story.** Built: a workflow that compiles the engine on macOS arm64 / x86_64 and Linux x86_64 / arm64, runs the golden test on each binary and attaches tarballs with SHA-256 to the release, and `clutor engine install`, which downloads and verifies the right one (tested against a local server; not yet run against a real release). To do: cut 0.2.0 so a release exists; `clutor pull` for supported models.
4. **Fewer bytes per token.** The router-ahead predictor is built and measured (82% of its picks are chosen, but it does not cut time on a bandwidth-bound run: docs/NATIVE_ENGINE.md); next, a cache policy compared on real traces; lower-bit expert variants only with their quality cost measured.
5. **A second architecture family** (DeepSeek-style or Mixtral-style), then a dense model for the non-MoE case.
6. **Invite others in.** A `clutor report` JSON format and a results table for benchmark submissions from other hardware; good-first-issue labels; GitHub Discussions. None of this raises the community score until others use it.

Not planned: GPU backends, training, a desktop app (it was removed on purpose).
