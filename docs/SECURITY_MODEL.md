# Security model

## The local server

The server listens on `127.0.0.1` by default. It protects against three things a local server must worry about:

1. **Other computers.** It binds to the loopback address. `clutor doctor` warns if you change `host`.
2. **Web pages in your browser.** A request carrying an `Origin` header is refused (403) unless the origin is a listed `corsOrigins` entry or the server's own address. Without this, any website you visit could call `localhost:4320`.
3. **DNS rebinding.** The `Host` header must be `localhost`, `127.0.0.1` or `[::1]`; anything else gets 421. This stops a hostile name that resolves to 127.0.0.1 from being treated as same-origin.

### The access token

Everything that touches files, processes or the model's lifecycle needs `Authorization: Bearer <token>` (or `x-clutor-token`). The token is 24 random bytes, generated each time the server starts, compared in constant time. Required for: the agent API, folder registration, the MCP endpoint, model-file planning and benchmarks, backend control, receipts.

Open without the token, so editors and scripts can use the server as a local model: health, hardware, the catalog, planning of **catalog** models, the simulator (with bounded work per request), the OpenAI and Ollama endpoints. Anything that can reach the loopback address as you can use your local model; if that matters, do not run the server on a shared machine.

### Folders

The agent works only in folders that were explicitly opened (registered). The path is resolved with symbolic links followed before the decision is made, so a link inside an allowed folder cannot lead outside it. The root of a drive is refused.

### Bounded work

Request bodies are limited to 8 MiB. The simulator caps tokens at 1,500, cache size and other inputs. Agent sessions are capped at 50 in memory.

## Receipts

Receipts are signed with ECDSA P-256. The private key is in `.clutor/keys/` (mode 0600, unreadable and unwritable by the agent). The public key and its RFC 7638 thumbprint are inside each signed receipt, so a third party verifies a log with `clutor receipt verify <file>` and no secret, and pins it with `--key-id` or `--pubkey`. Without pinning, a valid chain only shows consistency, not whose key signed it.

## The native engine

`clutor-engine` takes a file path and options on its command line and reads that file. It opens no sockets, runs no commands, and writes nothing except its report. The GGUF parser treats the file as hostile: bounded reads, capped string and array sizes, tensors that point outside the file are refused. The server exposes no route that runs it; `clutor stream`, `bench` and the `/cache` chat command invoke it locally.
