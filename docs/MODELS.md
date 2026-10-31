# Model compatibility

What the native engine can do with which model. "Real forward pass" means `clutor generate`, `clutor quality` and routing recorded from the model itself; "stream only" means the cache can be exercised on the file's layout with synthetic routing (or a trace from elsewhere), but the engine cannot compute the model.

| Model family | Real forward pass | Stream only | Checked against llama.cpp | Notes |
|---|---|---|---|---|
| OLMoE-1B-7B (`olmoe`) Q4_K_M | yes | yes | yes: identical token ids on 10 strings; same top token on 6 prompts; 97-99.8% of visible router choices | The only one. Tested file: OLMoE-1B-7B-0924-Instruct Q4_K_M. Needs Q4_K / Q6_K tensors; other quantizations dequantize in float (slower) |
| Qwen3-MoE (Qwen3-30B-A3B) | planned (roadmap milestone 2) | yes (layout) | no | Needs grouped-query attention, per-head QK-norm and top-k weight renormalization |
| Qwen2-MoE style | no | yes | no | Used for generated test files |
| DeepSeek, GLM, Kimi, Mixtral | no | catalog / planner only | no | The memory planner knows their published shapes; the engine has never opened one |
| Dense models (Llama, Qwen dense, ...) | no | no | no | No experts to cache: each token reads the whole model |

Quantization support in `c/src/quant.c`: F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K (dequantization); integer dot products for Q4_K and Q6_K (NEON with the dot-product instruction, AVX2 [cross-compiled only], portable C). The tokenizer is byte-level BPE with the `olmo` pre-tokenizer; letter and number classes are exact for ASCII and approximate above U+007F.

Platforms: macOS arm64 is where everything was measured. Linux builds the same code (CI); the AVX2 kernels have never been executed by the author. Windows is not supported by the native engine.
