# Clutor against llama.cpp, same file, same machine

Run it yourself and send the JSON it writes (see the end). Everything below is **measured** on one computer, and each row says how many runs it is the mean of.

```bash
make -C c
python3 scripts/bench/vs_llamacpp.py MODEL.gguf --runs 3 --json my-result.json
```

It needs `llama-bench` from llama.cpp on the PATH (or `LLAMA_BENCH=/path`). It starts the Clutor engine fresh for every run, feeds a fixed paragraph of about 110 tokens, decodes greedily, and runs `llama-bench` twice: `llama.cpp cpu` (`-ngl 0`) and `llama.cpp gpu` (all layers on the graphics chip, its default).

**Compare Clutor with the CPU row.** The Clutor engine does not use the GPU, so the GPU row shows what llama.cpp gains by using it, not a like-for-like race. On this Mac `llama.cpp cpu` still uses Apple's Accelerate library for part of prompt processing, so its prefill is not pure CPU code either.

## Apple M2, 16 GB, macOS (the only machine measured)

Tokens per second; for Clutor the mean with the lowest and highest run in brackets, for llama.cpp the mean ± standard deviation.

| Model | | Clutor | llama.cpp CPU | llama.cpp GPU (Metal) |
|---|---|---|---|---|
| Llama 3.2 1B Q4_K_M (0.8 GB, dense), 3 runs | decode | 50.5 | 56.5 ± 0.9 | 89.1 ± 2.7 |
| | prompt (109 tokens) | 151 to 196 (varies between sessions) | 129.5 ± 3.1 | 812.9 ± 8.8 |
| | first token | 0.6 to 0.9 s | not reported | not reported |
| Qwen3-VL 30B-A3B Q2_K (11 GB, MoE, fits in RAM), 3 runs | decode | 13.7 (10.4 to 17.4) | 13.4 ± 2.2 | 17.4 ± 13.6 |
| | prompt (110 tokens) | 18.5 (7.9 to 24.7) | 16.4 ± 1.7 | 97.8 ± 4.4 |
| | first token | 7.8 s (4.5 to 14.0) | not reported | not reported |

What this says, and what it does not:

- On the small dense model Clutor decodes about **10% slower** than llama.cpp's CPU path (50.5 against 56.5). Prompt processing is similar or somewhat faster. I profiled a decode run with macOS `sample`: about 75% of samples are in the matrix-product kernels (`run_chunks`), about 18% are worker threads waiting between jobs, and attention, the softmax and sampling are each under 2%. So the gap is in the kernels or in how the work is split across threads, not in anything around them. Changing the thread count (4 to 8 on this 4+4-core chip, three sweeps) moved decode by about 5% at most and the runs overlapped, so it is not the explanation either. The cause is still unknown.
- On the 30B mixture-of-experts model the two CPU paths are **level within the noise** (13.7 against 13.4, and the spread of Clutor's own runs is wider than the gap). The model fits in memory, so this does not test the engine's streaming from disk.
- llama.cpp on the GPU is far ahead at processing a prompt (about 6 times on the 1B model, about 5 times on the 30B) and ahead at decoding (about 1.6 times on the 1B). Its 30B GPU decode figure varies so much between runs (± 13.6) that I would not quote it.
- The machine had about 9 GB of swap in use from earlier work, so the 30B rows are noisier than a fresh machine would give. A first run of the 30B file on a cold page cache took 55 s to the first token and about 1.4 tok/s; those runs were discarded because other work was running, and the table is from the rerun.
- Not measured: memory use of llama.cpp, power, a context longer than about 1000 tokens, models larger than memory against llama.cpp's memory mapping, any machine other than this one.

## Other hardware

Nobody has run this yet on anything else. If you do, please send the JSON file (open an issue with it attached; the `machine` field records the platform) and I will add a row. A useful submission is the same two models above, or any GGUF you have, with `--runs 3`.

| Hardware | RAM | Model | Clutor decode | llama.cpp CPU decode | Source |
|---|---|---|---|---|---|
| (none yet) | | | | | |
