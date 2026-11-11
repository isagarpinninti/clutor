# Troubleshooting

Start with `clutor doctor`: it checks Node.js, the platform, the configuration, llama-server, the native engine, the OS sandbox, CPU features, GPU and NPU presence, and your models folder, and says what is missing. Everything below is a message Clutor actually prints, or a symptom with a cause that has been seen.

## "Is it fast enough?" and "why is it slow?"

| You see | Why | What to do |
|---|---|---|
| The model card says **Streams from SSD** or `clutor plan` says **SSD STREAMING REQUIRED** | The model is bigger than the memory the plan can use, so unused experts stay on the disk | It runs, slower. Look at the report's speed range and the **THEORETICAL** disk limit. A smaller quantisation or a smaller model that fits is faster |
| Replies start slow and speed up | The expert cache starts empty | The plan's speed is for a warm cache; chat a little before judging |
| `Very slow: under half a token per second` | The estimate (or your measured speed) is below 0.5 tokens/s | Use it for background jobs, or choose a smaller model |
| Speed was fine, then dropped to a few seconds per token | Another program took the memory and the system evicted the model's pages (7 s/token was measured on a 9B model this way) | Close memory-hungry apps; keep "Keep the model in memory" on in Settings (models over 40% of your RAM are never pinned) |
| `Tight in memory` | The model uses more than 55% of memory | It should run; close large apps first |
| Numbers in `clutor plan` look too good or too bad | Confidence says why: assumed disk speed, assumed memory speed, a hit-rate fit from one measurement | Run `clutor plan <file.gguf> --measure-disk`, then `clutor bench --file <file.gguf> --record out.json` for measured values |
| `That is faster than a disk can be` after `--measure-disk` | The file was served from memory because it was read recently | Try a file you have not read lately, or a larger one |

## Setup problems

| Message | Cause | Fix |
|---|---|---|
| `Native engine not built` / `the native engine is not installed` | The C engine is missing | `clutor engine install` (prebuilt) or `clutor engine build` (needs a C compiler) |
| `llama-server not found` | Models the engine cannot run, and the HTTP server, use llama.cpp | Install llama.cpp (`brew install llama.cpp`) or point the config at a binary; the planner, simulator and engine-supported models still work |
| `No clutor.config.json: using defaults` | Informational | `clutor init` writes one |
| `No model backend configured` | Chat and the agent need a model source | Set `backend.kind` in the configuration |
| An architecture the engine does not support | Only seven architectures have a forward pass | The model runs through llama-server instead (see [LIMITS.md](LIMITS.md)) |
| A model file that will not load in the app | Wrong or partial download, or an unsupported file | `clutor inspect <file>` reads the header and says what it is |

## The agent and its safety

| Message | Cause | Fix |
|---|---|---|
| `refused: the policy file is invalid` / `the policy file for this folder cannot be used, so no task was started` | `clutor.policy.json` has an unknown key, a wrong type, or bad JSON. Clutor refuses rather than ignoring it | Fix the file; the message names the key |
| `refused by policy ...: commands need an operating-system sandbox` | The policy sets `requireSandbox` and there is no sandbox. The desktop app has one only on macOS | Run on macOS, or install `bwrap` and use the command line on Linux, or ask for the policy to be changed |
| "Auto-run checks" is greyed out | The folder's policy forbids auto-approval | By design: every command asks |
| `No operating-system sandbox for the agent's commands` (`clutor doctor`) | macOS needs `sandbox-exec`, Linux needs `bwrap`; Windows has none | Commands still run in a scratch copy under the command policy, with your approval. `CLUTOR_SANDBOX=off` turns the sandbox off for a tool that cannot run inside it |
| A command fails only in the agent | It ran in a scratch copy with no network and no credentials folders | Installs are allowed network when you approve them; anything that needs `~/.config` or similar will not work there |
| `clutor receipts` says the chain is BROKEN at entry N | A receipt was edited, removed or reordered, or signed by a different key | The message names the entry. Receipts from before version 3 show `-` for reads, tokens and time; that is normal |
| The agent writes malformed tool calls | Small models often do | Those calls are rejected and nothing is written; try a larger model |

## The app

| Symptom | Cause | Fix |
|---|---|---|
| Nothing in the models list | The folder is empty or not scanned | Add a folder in Models, then "Read the folder again" |
| The Models dialog opens on first launch with no model | There is nothing loaded yet | Choose or download a model |
| `codesign -v` warns "code has no resources but signature indicates they must be present" | Seen on the locally built app; the cause is not known | Not known to affect running it |

## Reporting a problem

Include the output of `clutor doctor`, the model file name and size, and for a speed question the JSON from `clutor bench --file <model.gguf> --record out.json`. Say what you expected and what happened, with the numbers.
