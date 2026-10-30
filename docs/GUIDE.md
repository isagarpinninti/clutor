# Guide

## Install and first run

Clutor is a command-line tool (`npm install -g @clutor/cli`, then `clutor --help` or `clt --help`; Node.js 20 or newer). It reads `clutor.config.json`. For the native engine, build it with `npm run engine` in a source checkout (needs a C compiler).

```bash
clutor init                # writes clutor.config.json and a models/ folder
clutor doctor              # checks this machine, the settings, and whether llama-server is installed
```

## 1. Find out whether a model fits (no model needed)

```bash
clutor models                         # the catalog and any .gguf files in your models folder
clutor plan deepseek-v3 --ram 16      # architecture from the catalog
clutor plan qwen3-235b-a22b --quant Q3_K_M --ctx 16384 --kv 8
clutor plan ./models/my-model.gguf    # exact figures from the file
clutor inspect ./models/my-model.gguf # its header: layers, experts, quantisation mix
```

`--ram` is the amount the model may use; the operating system's share is subtracted first.

Read the verdict, the memory breakdown, and the speed ceilings together. Warnings appear when the dense weights are re-read for every token, or when the cache cannot hold a token's experts.

Replace the assumed disk speed with a measurement:

```bash
clutor bench --file ./models/my-model.gguf   # best: a file bigger than your RAM reads from the SSD
clutor bench                                  # writes a ~1 GiB test file (the result is an upper bound: RAM may serve it)
clutor plan deepseek-v3 --ram 16 --disk 5.2
```

## 2. Understand expert caching

```bash
clutor simulate qwen3-30b-a3b --cache 2 --locality 0.9
```

`clutor simulate` replays a synthetic routing trace through plain LRU and Clutor's cache and shows hit rate, bytes read per token, time waiting for the disk, and a heatmap of which experts of one layer end up cached. It is a teaching and planning tool. Why it is synthetic, and what it showed, is in [ENGINE.md](ENGINE.md).

## 3. Run a model

Clutor does not generate text itself. Pick one:

**llama.cpp (recommended for big models).** Install it (`brew install llama.cpp`, or a release from the llama.cpp project), then in `clutor.config.json`:

```json
{
  "backend": { "kind": "llama-server", "llamaServer": { "model": "/path/to/model.gguf" } },
  "memory": { "contextTokens": 8192, "kvBits": 8 }
}
```

`clutor backend` shows the exact command Clutor would run, built from your memory plan and from the flags that particular `llama-server` build supports (it reads `--help`). `clutor serve` loads the model; the app's **Server** page has "Load the model" and "Stop the model". Loading a very large model takes minutes; the server answers immediately and you can watch the state.

**Any OpenAI-compatible server** (LM Studio, Ollama's `/v1`, a remote API):

```json
{ "backend": { "kind": "openai", "baseUrl": "http://localhost:1234/v1", "model": "the-model-name", "apiKeyEnv": "MY_API_KEY" } }
```

Keys are read from the environment variable you name; never put a key in the file.

## 4. Use it from your editor

The server speaks the OpenAI and Ollama APIs. In Cursor, Continue, or any OpenAI client, set the base URL to `http://127.0.0.1:4320/v1`, any API key, and your model's name. For Ollama-style clients use `http://127.0.0.1:4320`. The model API does not need the access token; the server refuses requests from web pages and from other computers (see [SECURITY_MODEL.md](SECURITY_MODEL.md)).

There is also a read-only MCP endpoint at `/api/mcp` (it needs the token) with `list_files`, `read_file`, `search`, `repo_map` and `plan_memory`. It cannot change files or run commands.

## 5. The coding agent

From a terminal:

```bash
clutor code "Add input validation to src/signup.ts and a test for it" --dir ./my-project
clutor code "fix the failing test" --verify "npm test"       # re-runs the check after the agent finishes
clutor code "update the changelog" --apply                    # write without asking at the end
```

While it works you see each step. When it wants to run something that executes project code (tests, a build), you are asked first; it runs in a scratch copy with a clean environment. When it finishes you get the exact diff and the checks. **Nothing is written until you press "Write N files to the folder"** (or pass `--apply`). If a file changed on disk in the meantime, nothing is written and the files are named. After applying, **Undo** restores the folder from a checkpoint (`clutor rollback <id>` from a terminal).

Tips: give it a way to check its own work (`--verify "npm test"`); a model that cannot follow the tool format will show a high share of invalid replies in check CL-105, so try another model or `agent.toolMode: "json"`; small local models are much weaker than hosted ones at this, and Clutor does not hide that.

Details and limits: [AGENT.md](AGENT.md).

## Settings reference

| Key | Default | Meaning |
| --- | --- | --- |
| `port`, `host` | `4320`, `127.0.0.1` | Where the server listens |
| `models` | `models` | Folder scanned for `.gguf` files |
| `backend.kind` | `none` | `none`, `llama-server` or `openai` |
| `backend.llamaServer` | | `binary` (found on PATH if omitted), `model`, `port` (8081), `threads`, `extraArgs` |
| `backend.baseUrl`, `model`, `apiKeyEnv` | | For `kind: "openai"` |
| `memory` | | `contextTokens` (8192), `kvBits` (16, 8 or 4), `ramCeilingGiB`, `osHeadroomGiB`, `diskGBps` |
| `cache` | adaptive | `policy` (`lru`, `wtinylfu`, `adaptive`) and `smartRoute` (`enabled` false, `maxDeltaP` 0.015, ...) for the lab and plans |
| `agent` | | `toolMode` (`json`), `maxSteps` (30), `verifyCommand`, `autoApproveProjectCommands` (false), `commandTimeoutSeconds` (120) |
| `corsOrigins` | `[]` | Web origins allowed to call the server |

Unknown keys are an error.
