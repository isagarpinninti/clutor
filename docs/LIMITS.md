# Limits, and claims Clutor will not make

## The engine does not generate text yet

`c/` is a native streaming engine: it parses GGUF, reads experts from disk with `pread`, manages the expert cache, prefetches the next layer and applies cache-aware routing. For seven architectures (`olmoe`, `qwen3moe`, `granitemoe`, `qwen3vlmoe`, the dense `qwen2` and `llama`, and the hybrid `qwen35`) it also has a real forward pass (a tokenizer, attention or Gated DeltaNet, K-quant integer kernels, the model's own router) and can generate text with `clutor generate`; the interactive terminal chat and its coding agent run on it for those models. No other architecture has one yet, so for them the chat and the agent, and always `clutor decide` and the HTTP server, use llama.cpp (`llama-server`) or any OpenAI-compatible server. When a model is larger than memory, the operating system pages it in through llama.cpp's memory mapping; Clutor cannot control that cache.

What the engine reports is real where it says **measured** (bytes, time, waits, cache behaviour on the machine it ran on) and synthetic where it says so (`stream` without `--trace` uses a made-up router). On one OLMoE trace the synthetic default overstated the hit rate (77% against 47%), so do not read synthetic hit rates as a prediction for a real model; `generate`, `quality` and `stream --trace` use the model's real routing.

The engine bypasses the OS page cache only when asked (`--no-os-cache`: `F_NOCACHE` on macOS, `posix_fadvise` on Linux). It does not use `O_DIRECT`.

## Speed claims it will not make

The planner's ceilings come from arithmetic. Reading 12 GB of active experts per token from a 3 GiB/s disk takes about four seconds; no cache policy changes that unless most of the experts are already in memory. A 744-billion-parameter model in a few gigabytes of RAM at double-digit tokens per second is not physically possible, and Clutor will tell you so rather than show such a number. Dense models (Llama 70B and similar) have no experts to cache: each token reads the whole model.

## The cache lab is synthetic

Routing traces are generated with a stated shape (skewed expert popularity, neighbouring tokens that pick similar experts, layers that see correlated inputs). A policy that does well here may do differently on a real model. If you have a real routing trace, `parseRoutingTrace` reads one (one JSON object per layer, tokens separated by a blank line), but no tool in this release records one from llama.cpp.

## The sandbox is a seatbelt, not a vault

Commands run without a shell, from a short list of programs, in a scratch copy of the project, with a clean environment and a timeout, and only after you approve anything that runs project code. That stops a confused or prompt-injected model from running `rm -rf ~` or `curl | sh`.

On top of that, on **macOS** (`sandbox-exec`) and on **Linux with `bwrap` installed** the command and everything it starts also run in an operating-system sandbox: writes are allowed only in the scratch copy and the temporary folders, the usual credential places (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.config`, `~/.npmrc`, the keychain and browser profile folders) cannot be read, and the network is off except for a package install. `clutor doctor` says which one is active. Tested: on macOS the tests run real commands and confirm that a write outside the scratch folder fails, that `~/.ssh` cannot be listed and that a download fails. **Not tested:** the Linux (`bwrap`) path has never been run by the author. **Not covered:**

- Windows has no OS sandbox here; only the policy, the scratch copy and your approval protect you.
- The sandbox does not limit CPU, memory, disk use or how long a command runs apart from the timeout.
- The list of secret places is a list, not a guarantee: a token kept anywhere else, or in an environment variable you pass through, is readable.
- `sandbox-exec` is deprecated by Apple and could stop working in a future macOS; a flaw in the sandbox itself would also break this.
- A package install still has network access and runs the package's own scripts.
- The app also has a command box that runs what you type with your own permissions and no sandbox (the agent never uses it).

Only approve commands you recognise, in projects you trust. `CLUTOR_SANDBOX=off` turns the OS sandbox off for a tool that cannot run inside it.

## Receipts prove less than they sound like

Receipts are a chain of ECDSA P-256 signatures. Anyone can verify a log with no secret, and `--key-id` / `--pubkey` pin it to the key you published. A valid chain shows the log was not altered by someone without the private key. It does not prove which model ran, that its output is correct, or that the key is not held by someone else: that last one is on you to check by comparing the key id out of band.

## Platform status

- **macOS (Apple silicon):** developed and tested here (Apple M2, 16 GiB): TypeScript tests, C unit tests, ASan/UBSan/TSan runs, and the engine on a 2.4 GiB generated model.
- **Linux:** the engine and CLI are expected to work and CI builds and tests them; the author has not run them on Linux hardware.
- **Windows:** not supported by the engine (it uses POSIX `pread` and pthreads). The TypeScript CLI paths are written for Windows but untested.
- **Hardware:** one machine. `clutor bench` is how you learn yours. Disk numbers from the generated test file are real reads of real (zero-filled) bytes.

## Models

The catalog covers five architectures whose published hyperparameters the planner reproduces to within 2% of their stated size. Models newer than that, or whose architecture is not in the catalog, are planned from their `.gguf` file, which needs no catalog entry.
