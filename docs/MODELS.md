# Model compatibility

What the native engine can do with which model. "Real forward pass" means `clutor generate`, `clutor quality` and routing recorded from the model itself; "stream only" means the cache can be exercised on the file's layout with synthetic routing (or a trace from elsewhere), but the engine cannot compute the model.

| Model family | Real forward pass | Stream only | Checked against llama.cpp | Notes |
|---|---|---|---|---|
| OLMoE-1B-7B (`olmoe`) Q4_K_M | yes | yes | yes: identical token ids on 10 strings; same top token on 6 prompts; 97-99.8% of visible router choices | The only one. Tested file: OLMoE-1B-7B-0924-Instruct Q4_K_M. Needs Q4_K / Q6_K tensors; other quantizations dequantize in float (slower) |
| Granite 3.0 3B-A800M (`granitemoe`) Q4_K_M | yes | yes | yes: identical tokenization on 24 strings; token ids and top token identical on 6 prompts; 98.4-99.0% router agreement on 3 prompts; identical 24-token greedy continuations on 3 prompts; streamed text identical to mapped in 8 settings. Token logprobs differ more than on OLMoE (mean 0.11-0.40, worst 1.03; 2 of 6 prompts over the OLMoE-tuned tolerance), judged numerical noise, not proven | 40 experts (8 used), 32 layers, GQA 24/8, no QK-norm, adjacent-pair rotary embeddings, four scaling multipliers, the `refact` pre-tokenizer (every digit alone). The 1.9 GiB file fits in RAM |
| Qwen3-MoE (Qwen3-30B-A3B) Q4_K_M | yes | yes | yes: identical tokenization on 24 strings; 96.7-98.0% router agreement on 3 prompts; identical 24-token greedy continuations on 3 prompts. Token logprobs differ more than on OLMoE (mean 0.16-2.23), judged numerical noise amplified by 48 layers, not proven | The 17.3 GiB file is larger than 16 GiB of RAM: needs `--cache-mb` (the engine refuses to map it). 2.0-3.0 tokens/s at 2-8 GiB caches (docs/TEST_REPORT.md) |
| Qwen2-MoE style | no | yes | no | Used for generated test files |
| DeepSeek, GLM, Kimi, Mixtral | no | catalog / planner only | no | The memory planner knows their published shapes; the engine has never opened one |
| Dense models (Llama, Qwen dense, ...) | no | no | no | No experts to cache: each token reads the whole model |

Quantization support in `c/src/quant.c`: F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K (dequantization); integer dot products for Q4_K and Q6_K (NEON with the dot-product instruction, AVX2 [unit test passed on a CI x86 Linux runner; not run by the author locally], portable C). The tokenizer is byte-level BPE with the `olmo`, `qwen2` or `refact` pre-tokenizer; letter and number classes are exact for ASCII and approximate above U+007F.

Platforms: macOS arm64 is where everything was measured. Linux builds the same code (CI); the AVX2 kernels passed their unit test on a CI x86 runner but the author has not run them locally. Windows is not supported by the native engine.
