# Benchmark records

One JSON file per run, committed under `benchmarks/<machine>/`, for example `benchmarks/mac-m2-16gb/qwen3-30b-a3b-q4km.json`.

```bash
clutor bench --file MODEL.gguf --record benchmarks/mac-m2-16gb/MODEL.json
```

The format is `BenchRecord` in `packages/core/src/planner/record.ts` (`schema: 1`). It records the machine (OS, CPU, cores, RAM, CPU features, accelerators present), the model, the runtime settings, and the results. Rules:

- Every result is `{ value, basis, runs }`, and `basis` is `measured`, `estimated` or `theoretical`. A result that was not measured is `null`, never `0`.
- `clutor bench --record` records, **measured**: with an expert model, the disk read speed (random expert-sized reads); with a real `.gguf` and the native engine, one fresh engine process's load time, prompt speed, first-token time and greedy generation speed (a single run, so `runs` is 1; the note says the file cache was not dropped). Not recorded yet, so `null`: cache hit rate and peak memory. A dense model has no expert reads, so its disk figure is `null`. For several runs against llama.cpp use `scripts/bench/vs_llamacpp.py` (see `docs/bench/`).
- `accelerators` lists hardware that is present; `usedByClutor` is `false` for all of it because the engine computes on the CPU only.
- A file smaller than about 1.5 times your RAM may be served from the operating system's cache, so the disk figure is then an upper bound; the record says so in `notes`.

Machine folders are empty until someone runs this. Nothing in this directory is a claim about performance yet.
