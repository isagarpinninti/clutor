# @clutor/cli

The `clutor` command and local server.

```bash
npm install -g @clutor/cli
clutor plan deepseek-v3 --ram 16     # will it fit, and how fast could it possibly go
clutor plan ./my-model.gguf          # exact figures from a real file
clutor serve                         # OpenAI- and Ollama-compatible API on 127.0.0.1:4320, agent, MCP
clutor code "fix the failing test" --verify "npm test"
```

Run `clutor --help`. Documentation: <https://github.com/isagarpinninti/clutor/blob/main/docs/GUIDE.md>.

It does not run models itself; it supervises [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` or talks to any OpenAI-compatible server.
