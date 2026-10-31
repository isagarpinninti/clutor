# Benchmarks against Colibri and AirLLM

Measured 2026-10-31 on one machine: MacBook Air, Apple M2, 16 GiB, internal SSD (a fresh single-thread read runs at about 1.5 GB/s), macOS. Every number below is **measured**, one run per row unless a range is given, and the same setting varied by up to about 30% between runs, so read orders of magnitude and large gaps, not decimals. The scripts are in `scripts/bench/`.

## What could and could not be compared

| | Clutor | Colibri (v2.0.0 checkout e0db083) | AirLLM 4.0.0 |
|---|---|---|---|
| Runs OLMoE-1B-7B | yes | yes (its `olmoe` engine) | **no on macOS**: `AutoModel` always selects an MLX path that only supports Llama-architecture models |
| Weights used | OLMoE-1B-7B-0924-Instruct Q4_K_M GGUF | the same GGUF, converted by Colibri's own `convert_gguf_to_olmoe.py` (experts dequantized and re-quantized to int8, dense weights kept in float32) | TinyLlama-1.1B-Chat (bf16 safetensors) |

So Clutor and Colibri are compared on the same model, prompt and token count. AirLLM could only be run on a different, dense model, and cannot be compared with Clutor directly: its result describes AirLLM, not a ranking.

## Clutor against Colibri, OLMoE-1B-7B

Job: the 20-token prompt "Once upon a time, in a small village by the sea, there lived an old fisherman who" and 80 greedy tokens (Clutor's own continuation). Job time includes the prompt: Colibri prefills the 20 tokens as one batch, Clutor feeds them one at a time, which favours Colibri a little. Colibri ran with its own defaults (4 physical-core threads; 8 threads gave the same speed), Clutor with 8.

**Correctness, same text, same weights:** teacher-forced perplexity over 799 tokens of this project's docs is **10.943 (Clutor, exact routing) and 10.87 (Colibri)**. The engines' greedy continuations agree on 24 of 80 tokens, then diverge: expected for two engines with different weight precision, and not evidence that either is wrong.

**Everything in RAM** (file already in the OS cache), two rounds each:

| engine | job seconds | peak RSS |
|---|---|---|
| Clutor, file mapped | 9.6-9.9 | about 3.7 GiB |
| Colibri, cache 64 experts per layer (everything) | 15.4-16.9 | about 4.1 GiB |

**Memory-bounded, cold files** (fresh copies written without caching, read with the OS cache bypassed, so experts really come from the SSD):

| engine and setting | job seconds | peak RSS | expert hit rate |
|---|---|---|---|
| Clutor, 128 MiB expert cache | 26.6 | 481 MiB | 7.3% |
| Clutor, 256 MiB | 23.6 | 610 MiB | 15.5% |
| Clutor, 512 MiB | 20.0 | 866 MiB | 30.9% |
| Clutor, 1024 MiB | 15.5 | 1,374 MiB | 49.9% |
| Colibri, 4 experts per layer | 18.4 | 2,119 MiB | 17.2% |
| Colibri, 8 per layer | 15.7 | 2,805 MiB | 33.0% |
| Colibri, 16 per layer | 14.5 | 3,491 MiB | 53.1% |
| Colibri, 32 per layer | 14.3 | 4,864 MiB | 77.2% |
| Colibri, 16 per layer + its prefetch (`PILOT=1`) | 12.7 | 3,594 MiB | 56.3% |

What this does and does not say:
- **Same model, in RAM:** Clutor is about 1.6x faster.
- **Memory floor:** Colibri keeps the dense weights in float32 (RSS 2.0 GB right after load), so it cannot go below about 2.1 GiB here; Clutor ran down to 481 MiB (slowly, 26.6 s).
- **At a similar memory budget Clutor was as fast or faster** (1,374 MiB: 15.5 s against Colibri's 2,119 MiB: 18.4 s), but that gap is inside the run-to-run spread, so call it comparable, not a win.
- **Not tested:** any model larger than RAM, which is Colibri's real target (it runs GLM, DeepSeek, Kimi and Qwen engines Clutor does not have; Clutor has one architecture). Hit rates differ in meaning because the engines' caches and expert sizes differ (Colibri's int8 experts are about 6.3 MB, Clutor's Q4_K experts 3.9 MiB). Colibri was not tuned beyond defaults and `PILOT`; Clutor's prefetch was off.

## AirLLM, TinyLlama-1.1B (not comparable with the above)

40 greedy tokens from the same prompt on the same weights:

| mode | s / token | peak RSS |
|---|---|---|
| PyTorch (transformers), whole model in RAM, bf16 | 0.06 | 2,306 MiB |
| AirLLM (MLX path), one layer in memory at a time | 1.74 | 891 MiB |

AirLLM used 39% of the memory and was about 29x slower, and the text it produced matched PyTorch's. That is the layer-streaming trade: every token re-reads every layer, so speed is bounded by disk bandwidth over the whole model (here 2.2 GB per token; 1.74 s is about 1.3 GB/s). A mixture-of-experts engine reads only the chosen experts: Clutor reads 0.24-0.45 GB of OLMoE's 3.9 GB per token (6-11%).

## What this does not show

Nothing here shows Clutor is "more powerful" than either project in general. It shows that on one small MoE model, one machine, Clutor is faster in RAM, runs in less memory, and is comparable under a memory budget. Colibri supports far larger models and many more architectures; AirLLM targets dense models far larger than this. A fair test of the larger claim needs a model that does not fit in RAM, which this Mac has not run.
