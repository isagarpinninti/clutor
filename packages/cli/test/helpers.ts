import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ConfigSchema, createClutorServer, type ClutorConfig, type ClutorServer, type ServerOptions } from "../src";

export const dirs: string[] = [];
export const tmp = (): string => { const d = mkdtempSync(join(tmpdir(), "clutor-cli-")); dirs.push(d); return d; };
export const cleanup = (): void => { for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true }); };
export const put = (root: string, rel: string, text: string): void => { mkdirSync(join(root, rel, ".."), { recursive: true }); writeFileSync(join(root, rel), text); };

type ReplyObject = { content?: string | null; tool_calls?: unknown[]; logprobs?: unknown };
export type Reply = string | ReplyObject | ((body: Record<string, unknown>) => string | ReplyObject);

/** A stand-in for llama-server or any OpenAI-compatible API: answers each chat request with the next scripted reply. */
export interface FakeLlm { url: string; requests: Array<Record<string, unknown>>; headers: Array<Record<string, string | string[] | undefined>>; script: Reply[]; close(): Promise<void> }
export async function fakeLlm(script: Reply[] = []): Promise<FakeLlm> {
  const state: FakeLlm = { url: "", requests: [], headers: [], script, close: async () => undefined };
  let i = 0;
  const srv: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const body = chunks.length ? (JSON.parse(Buffer.concat(chunks).toString()) as Record<string, unknown>) : {};
      if (req.url?.endsWith("/chat/completions")) {
        state.requests.push(body);
        state.headers.push(req.headers);
        const r0 = state.script[Math.min(i++, state.script.length - 1)] ?? "ok";
        const r = typeof r0 === "function" ? r0(body) : r0;
        const { logprobs, ...msg } = typeof r === "string" ? { content: r } as ReplyObject : r;
        if (body["stream"] === true) {
          res.writeHead(200, { "content-type": "text/event-stream" });
          const text = msg.content ?? "";
          for (const word of text.split(/(?<= )/)) res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: word } }] })}\n\n`);
          res.write(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 7, completion_tokens: 3 } })}\n\n`);
          res.end("data: [DONE]\n\n");
          return;
        }
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ choices: [{ message: { role: "assistant", ...msg }, ...(logprobs ? { logprobs } : {}), finish_reason: "stop" }], usage: { prompt_tokens: 7, completion_tokens: 3 } }));
        return;
      }
      res.writeHead(404).end();
    });
  });
  await new Promise<void>((r) => srv.listen(0, "127.0.0.1", r));
  state.url = `http://127.0.0.1:${(srv.address() as AddressInfo).port}/v1`;
  state.close = () => new Promise((r) => { srv.close(() => r()); srv.closeAllConnections(); });
  return state;
}

export const tool = (name: string, args: Record<string, unknown>): string => JSON.stringify({ tool: name, args });

export interface Running { server: ClutorServer; base: string; token: string; root: string; llm: FakeLlm; get(path: string, opts?: Req): Promise<R>; post(path: string, body?: unknown, opts?: Req): Promise<R>; stop(): Promise<void> }
export interface Req { token?: boolean | string; headers?: Record<string, string> }
export interface R { status: number; body: any; headers: Headers; text: string } // eslint-disable-line @typescript-eslint/no-explicit-any

export async function start(over: Partial<ClutorConfig> = {}, script: Reply[] = [], opts: { roots?: string[]; withBackend?: boolean; extra?: Partial<ServerOptions> } = {}): Promise<Running> {
  const root = tmp();
  const llm = await fakeLlm(script);
  const config = ConfigSchema.parse({ port: 0, backend: opts.withBackend === false ? { kind: "none" } : { kind: "openai", baseUrl: llm.url, model: "fake-model" }, ...over });
  const server = createClutorServer({ root, config, version: "test", roots: opts.roots ?? [], ...(opts.extra ?? {}) });
  const a = await server.listen();
  const base = `http://127.0.0.1:${a.port}`;
  const call = async (method: string, path: string, body: unknown, o: Req = {}): Promise<R> => {
    const headers: Record<string, string> = { ...(body !== undefined ? { "content-type": "application/json" } : {}), ...(o.headers ?? {}) };
    if (o.token !== false) headers["authorization"] = `Bearer ${typeof o.token === "string" ? o.token : server.token}`;
    const res = await fetch(`${base}${path}`, { method, headers, ...(body !== undefined ? { body: typeof body === "string" ? body : JSON.stringify(body) } : {}) });
    const text = await res.text();
    let parsed: unknown = text;
    try { parsed = JSON.parse(text); } catch { /* not JSON */ }
    return { status: res.status, body: parsed, headers: res.headers, text };
  };
  return { server, base, token: server.token, root, llm, get: (p, o) => call("GET", p, undefined, o), post: (p, b, o) => call("POST", p, b ?? {}, o), stop: async () => { await server.close(); await llm.close(); } };
}

/** A script that acts like llama-server: prints --help, loads for a while (503 on /health), then answers chat. */
export function writeFakeLlamaServer(dir: string, o: { flags?: string[]; flashValue?: boolean; loadMs?: number; crash?: boolean; neverReady?: boolean } = {}): string {
  const path = join(dir, process.platform === "win32" ? "llama-server.exe" : "llama-server");
  const flags = o.flags ?? ["--jinja", "--cache-type-k", "--cache-type-v", "--flash-attn", "--mlock", "--no-mmap"];
  const help = flags.map((f) => (f === "--flash-attn" ? (o.flashValue ? "  -fa, --flash-attn [on|off|auto]  set Flash Attention use" : "  -fa, --flash-attn  enable flash attention") : `  ${f}  something`)).join("\n");
  writeFileSync(path, `#!/usr/bin/env node
const http = require("http");
const a = process.argv.slice(2);
if (a.includes("--help")) { console.log(${JSON.stringify(help)}); process.exit(0); }
if (${o.crash ? "true" : "false"}) { console.error("error: failed to load model"); process.exit(3); }
const port = Number(a[a.indexOf("--port") + 1]);
const started = Date.now();
process.on("SIGTERM", () => process.exit(0));
http.createServer((req, res) => {
  if (req.url === "/health") { const ok = ${o.neverReady ? "false" : "true"} && Date.now() - started >= ${o.loadMs ?? 0}; res.writeHead(ok ? 200 : 503); return res.end("{}"); }
  if (req.url.endsWith("/chat/completions")) { res.writeHead(200, { "content-type": "application/json" }); return res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "from fake llama-server with " + JSON.stringify(a) } }] })); }
  res.writeHead(404); res.end();
}).listen(port, "127.0.0.1");
`);
  chmodSync(path, 0o755);
  return path;
}
