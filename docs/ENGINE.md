# How the engine works

This page covers the TypeScript planner and simulator. The native streaming engine is in [NATIVE_ENGINE.md](NATIVE_ENGINE.md). Clutor contains a planner, a GGUF reader, a cache simulator, a native streaming engine and an agent. It does **not** yet contain a transformer: generating text is done by llama.cpp (or any OpenAI-compatible server) behind it. This page explains what the parts compute, and what each number on screen is based on.

## What is measured, computed, assumed or simulated

| Figure | Where it comes from |
| --- | --- |
| Sizes from a `.gguf` file | **Read** from the file's own tensor table. Header plus the sum of tensor sizes equals the file size exactly (tested on a real llama.cpp-made file) |
| Sizes for a catalog model | **Computed** from the model's published architecture (layers, hidden size, experts, attention shape) at a chosen quantisation. The computation reproduces each model's published parameter count to within 2% (tested) |
| Disk speed | **Assumed** (3 GiB/s) until you run `clutor bench` , then **measured**. Node cannot bypass the operating system's page cache, so a small test file may be served from memory; the result says whether the file was big enough (over 1.5x your RAM) to mean "from the SSD" |
| Tokens-per-second ceilings | **Computed**: bytes the disk must supply per token, divided by disk speed. An upper bound, not a prediction |
| Cache hit rates in the lab | **Simulated** on a synthetic routing trace |
| Speed in the lab | **Modelled** from the disk speed and compute time you enter |

Nothing on screen presents a simulation or an assumption as a measurement.

## The GGUF reader

`parseGguf` reads the header, metadata and tensor table through a `ByteReader`, so a 400 GB file is inspected by reading its first few megabytes. It handles versions 2 and 3, all metadata types, and the ggml tensor types it has sizes for (unknown types are reported, and the totals say they are too small). It refuses: bad magic, other versions, counts or string lengths that would exhaust memory, tensors that start outside the file or are not aligned. It only ever throws its own `GgufError` (fuzzed with random bytes, truncations and flipped bytes).

`buildModelLayout` classifies tensors:

- **Always resident:** token embeddings, output head, norms, and the routers (`ffn_gate_inp`).
- **Dense layer weights:** attention, norms, shared experts, and the feed-forward of dense layers. Read for every token.
- **Routed experts:** for fused tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), experts are the slowest-varying dimension, so each expert is one contiguous slab per tensor; the layout records the exact byte range of every expert. Older per-expert tensors are handled too.

## The memory planner

Given a layout and a RAM budget (machine RAM minus headroom; by default the operating system keeps 22% of RAM, but never under 1.8 GiB or over 3.5 GiB, reserved before the dense window and the expert cache are sized). `clutor plan --ram N` plans for a machine with N GiB of RAM, so the headroom comes out of N:

```
resident  = embeddings + output head + routers
kv        = 2 x layers x kvHeads x headDim x contextTokens x (kvBits / 8)
runtime   = 384 MiB (compute buffers, the CLI)
```

For multi-head latent attention models (DeepSeek) the KV cache holds one compressed latent per token, which is modelled as a single "head".

The strategy is the first of these that fits:

1. **Fully resident:** everything fits. No disk reads while generating.
2. **Dense resident, experts streamed:** the dense weights fit in memory with room for at least two tokens' worth of experts. The rest of the budget is the expert cache.
3. **Dense ring, experts streamed:** even the dense weights do not fit; a window of two layers is kept and the rest re-read for every token.
4. **Does not fit:** the always-resident parts alone exceed the budget. The shortfall is reported.

The speed ceiling at cache hit rate *h* is

```
tokens/s <= diskBytesPerSec / ( denseStreamBytes + activeExpertBytes x (1 - h) )
```

where `activeExpertBytes = routedLayers x expertsUsed x expertBytes`. For a dense model there are no experts, so no cache can help: the ceiling is the disk speed divided by the model size. That is why a 70B dense model cannot be interactive from an SSD.

The planner also warns when the cache cannot hold one token's experts (every token then re-reads most of them).

## The cache policies

Experts are cached by `(layer, expert)`. Three policies share one interface, so any difference in hit rate is the policy's.

- **LRU:** evict the least recently used. The baseline.
- **W-TinyLFU:** a small LRU window takes every new entry; when it overflows, its oldest entry competes with the main space's eviction victim and is admitted only if a count-min sketch says it has been used more often. The main space is segmented (probation and protected). Frequently used experts survive a long run of rarely used ones.
- **Adaptive W-TinyLFU** (default): the right window size depends on the workload. When consecutive tokens reuse experts, a big window (LRU-like) wins; with weak reuse among many one-offs, a small window (LFU-like) wins. Hill climbing on the window got stuck, so instead several key-only "ghost" caches with different windows replay the same accesses, and the real cache takes the window of whichever ghost has the best recent hit rate.

### What the experiments showed

Qwen3-30B-A3B at Q4, 700 tokens of a synthetic trace, hit rate by policy (`scripts` and the lab reproduce this):

| Locality | Cache | LRU | W-TinyLFU, fixed 1% window | Adaptive |
| --- | --- | --- | --- | --- |
| 0.9 | 1 GiB | 0.0% | 21.8% | 21.8% |
| 0.9 | 2 GiB | 54.1% | 36.5% | 54.2% |
| 0.9 | 4 GiB | 75.5% | 59.2% | 75.5% |
| 0.5 | 1 GiB | 0.0% | 23.3% | 23.3% |
| 0.5 | 2 GiB | 36.3% | 38.2% | 35.5% |
| 0.5 | 4 GiB | 55.0% | 59.5% | 61.0% |

Three things to take from it:

1. **No single policy wins.** Fixed W-TinyLFU loses badly to LRU when neighbouring tokens reuse experts (0.9 locality). Plain LRU collapses to nothing when the cache is smaller than one token's working set (1 GiB here: a token touches about 1.4 GB, so LRU cycles and never hits).
2. **The adaptive policy tracks the better one** in five of six rows, beats both in one, and loses 2.7 points in one (0.5 locality, 2 GiB). It is not free.
3. **When the cache is large and reuse is high, there is little to gain over LRU.** The lab shows the two tied.

These are results on a synthetic trace with the stated locality. A real model's routing may differ, and the output labels the simulation accordingly.

### Read-ahead

The prefetcher learns, online, how often expert *b* follows expert *a* at the next layer, and predicts the next layer's experts from the current layer's choices. It reads ahead only when the predictor is confident (default 0.4) **and** the disk would otherwise be idle during this layer's compute. An early version that prefetched freely made the model slower: with the disk already busy 94% of the time, speculative reads competed with the reads the model was waiting for. For a model that does not fit in memory the disk is almost always the bottleneck, so read-ahead usually does nothing, and the lab says so.

### Stand-in experts ("SMART-ROUTE")

Optionally, a missing expert can be replaced by a cached one among the router's next-best choices, only when the router's probability gap is at most a threshold (default 0.015), at most a few per layer. This can change the model's output. Clutor does not know by how much: it reports the share of router probability that went to exactly the chosen experts, as a proxy, and a check (CL-103) fails if the threshold is set above 0.05. It is off by default.

## Prompt-lookup drafting

`PromptLookupIndex` guesses upcoming tokens by finding the current suffix earlier in the text and copying what followed. `evaluateDrafting` measures, on a token stream, how many tokens one verification pass would accept if the model's output were exactly that stream: an upper bound on a real model, measured on real text (with a rough code tokenizer, so indicative only). It is a library component; the server does not yet use it to speed up generation, because that needs support in the model server.

## The checks (CL-101 to CL-108)

Each looks at something concrete and reports pass, warning, failure, or "not run".

| | Looks at |
| --- | --- |
| CL-101 | The plan fits within the RAM ceiling, and the ceiling leaves the OS at least 1 GiB |
| CL-102 | The streaming window can hold a token's worth of experts |
| CL-103 | Stand-in experts are off or bounded (warning above 0.015, failure above 0.05) |
| CL-104 | The cache policy resists scans (plain LRU gets a warning) |
| CL-105 | The share of the model's replies that were not valid tool calls |
| CL-106 | Blocked attempts to leave the folder, read secrets, run disallowed commands, or exceed limits |
| CL-107 | Whether commands that run project code needed your approval |
| CL-108 | Every tensor lies inside the model file; the receipt chain is intact |

They report. They do not decide for you.
