# Documentation map

Every topic below points at the page that covers it. **Status** is about the thing described: *implemented* (works and is tested somewhere, see [CAPABILITIES.md](CAPABILITIES.md) for where), *experimental* (works in a narrow case or has little testing), *planned* (design only, nothing built). A topic with no page of its own says so.

| Topic | Read | Status |
|---|---|---|
| Architecture: what lives where | [README](../README.md) ("Repository layout"), [NATIVE_ENGINE](NATIVE_ENGINE.md) ("Design") | implemented. No single architecture page yet |
| Inference engine | [NATIVE_ENGINE](NATIVE_ENGINE.md) | implemented for seven architectures |
| MoE expert streaming and the memory model | [NATIVE_ENGINE](NATIVE_ENGINE.md), [ENGINE](ENGINE.md) ("The memory planner"), [STREAMING_RESULTS](STREAMING_RESULTS.md) | implemented; measured on one machine |
| Caching and prefetching | [ENGINE](ENGINE.md) ("The cache policies"), [NATIVE_ENGINE](NATIVE_ENGINE.md) ("Router-ahead prefetch") | implemented |
| Performance planner and report (`clutor plan`) | [ENGINE](ENGINE.md), [GUIDE](GUIDE.md) §1 | implemented; the speed estimate rests on one fitted measurement |
| Benchmarking and comparison with llama.cpp | [bench/README](bench/README.md), [BENCHMARKS](BENCHMARKS.md), [../benchmarks/README](../benchmarks/README.md) | implemented; one machine |
| Model support | [MODELS](MODELS.md), README ("Supported models") | implemented; list is by architecture |
| Security: server, command sandbox, receipts | [SECURITY_MODEL](SECURITY_MODEL.md), [LIMITS](LIMITS.md) | implemented; OS sandbox on macOS and Linux only |
| Coding agent | [AGENT](AGENT.md), [GUIDE](GUIDE.md) §5 | implemented; only as reliable as the model |
| Command line | [GUIDE](GUIDE.md), `clutor --help` | implemented |
| Hardware support and what runs where | [CAPABILITIES](CAPABILITIES.md) | implemented table; GPU/NPU: none |
| GPU and other accelerators | [DIRECTIONS](DIRECTIONS.md) §1 | planned (the seam exists, no backend) |
| Mesh of machines | [DIRECTIONS](DIRECTIONS.md) §2 | planned, design only |
| Troubleshooting | `clutor doctor`; [GUIDE](GUIDE.md) | no dedicated page yet |
| What Clutor will not claim | [LIMITS](LIMITS.md), [SCORECARD](SCORECARD.md) | n/a |
| Roadmap | [ROADMAP](ROADMAP.md) | n/a |
