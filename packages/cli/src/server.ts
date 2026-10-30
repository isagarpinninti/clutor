import { agentLlm } from "./agentllm";
import { randomBytes, timingSafeEqual } from "node:crypto";
import { createServer as createHttpServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { Readable } from "node:stream";
import {
  BASELINE, CATALOG, CLUTOR_DEFAULTS, GgufError, GiB, buildModelLayout, parseGguf, QUANTS, ShadowWorkspace, approximateLayout, evaluateGates, expertSizeBytes, generateRoutingTrace, planMemory, simulateStreaming, type LlmClient, type MemoryPlan,
} from "@clutor/core";
import { DecideError, runDecide, sealDecision } from "./decide";
import { NodeProject, hostInfo, listCheckpoints, ConflictError, fileReader, measureExpertReads, readReceipts, verifyReceiptLog, writeSyntheticGguf } from "@clutor/core/node";
import { z } from "zod";
import type { BackendManager } from "./backend";
import type { ClutorConfig } from "./config";
import { handleMcp } from "./mcp";
import { ModelError, listLocalModels, resolveModel } from "./models";
import { fromOpenAiReply, streamToOllama, toOpenAiBody, type OllamaChatRequest } from "./ollama";
import { AgentSession, isInside, normaliseRoot } from "./sessions";

export interface ServerOptions {
  root: string;
  config: ClutorConfig;
  version: string;
  backend?: BackendManager;
  /** Starts the model (llama-server) on request. Provided by whoever owns the configuration. */
  startBackend?: () => Promise<void>;
  /** Folders the server may work in from the start (the CLI passes the current folder). */
  roots?: string[];
  /** Fixed token for tests; otherwise random. */
  token?: string;
  fetch?: typeof globalThis.fetch;
  log?: (line: string) => void;
}

export interface ClutorServer {
  readonly http: Server;
  readonly token: string;
  listen(): Promise<{ host: string; port: number }>;
  close(): Promise<void>;
  addRoot(dir: string): Promise<string>;
  readonly sessions: Map<string, AgentSession>;
}

class HttpError extends Error {
  constructor(readonly status: number, message: string, readonly extra: Record<string, unknown> = {}) {
    super(message);
  }
}

const MAX_BODY = 8 * 1024 * 1024;
/** The desktop app's renderer. */
const APP_ORIGINS = ["app://clutor", "clutor://app"];
const LOCAL_HOST = /^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/i;

export function createClutorServer(o: ServerOptions): ClutorServer {
  const cfg = o.config;
  const token = o.token ?? randomBytes(24).toString("hex");
  const f = o.fetch ?? globalThis.fetch;
  const roots = new Set<string>();
  const sessions = new Map<string, AgentSession>();
  const host = hostInfo();
  let boundPort = cfg.port;
  const ready = (o.roots ?? []).map((r) => normaliseRoot(r).then((x) => roots.add(x)));

  const send = (res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void => {
    const text = typeof body === "string" ? body : JSON.stringify(body);
    res.writeHead(status, { "content-type": typeof body === "string" ? "text/plain; charset=utf-8" : "application/json; charset=utf-8", "cache-control": "no-store", "x-content-type-options": "nosniff", ...headers });
    res.end(text);
  };

  const allowedOrigin = (origin: string): boolean => APP_ORIGINS.includes(origin) || cfg.corsOrigins.includes(origin) || origin === `http://localhost:${boundPort}` || origin === `http://127.0.0.1:${boundPort}`;

  const readBody = async (req: IncomingMessage): Promise<unknown> => {
    const chunks: Buffer[] = [];
    let n = 0;
    for await (const c of req) {
      n += (c as Buffer).length;
      if (n > MAX_BODY) throw new HttpError(413, "request body is too large");
      chunks.push(c as Buffer);
    }
    if (n === 0) return {};
    try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch { throw new HttpError(400, "the body is not valid JSON"); }
  };
  const parse = <T extends z.ZodType>(schema: T, v: unknown): z.infer<T> => {
    const r = schema.safeParse(v);
    if (!r.success) throw new HttpError(400, r.error.issues.map((i) => `${i.path.join(".") || "body"}: ${i.message}`).join("; "));
    return r.data;
  };

  const authed = (req: IncomingMessage): boolean => {
    const h = req.headers["authorization"];
    const t = typeof h === "string" && h.startsWith("Bearer ") ? h.slice(7) : (req.headers["x-clutor-token"] as string | undefined) ?? "";
    const a = Buffer.from(t), b = Buffer.from(token);
    return a.length === b.length && timingSafeEqual(a, b);
  };
  const needAuth = (req: IncomingMessage): void => { if (!authed(req)) throw new HttpError(401, "this needs the access token (Authorization: Bearer <token>)"); };

  const llm = (): LlmClient => {
    const b = cfg.backend;
    if (b.kind === "openai") {
      if (!b.baseUrl) throw new HttpError(409, "backend.baseUrl is not set");
      return agentLlm(cfg, { baseUrl: b.baseUrl, model: b.model ?? "default", ...(b.apiKeyEnv && process.env[b.apiKeyEnv] ? { apiKey: process.env[b.apiKeyEnv] as string } : {}), local: false }, f);
    }
    if (b.kind === "llama-server") {
      if (o.backend?.state !== "ready") throw new HttpError(409, "the model is not loaded; start it first (POST /api/backend/start)");
      return agentLlm(cfg, { baseUrl: o.backend.baseUrl, model: "local", local: true }, f);
    }
    throw new HttpError(409, "no model backend is configured: set backend.kind to \"openai\" or \"llama-server\" in clutor.config.json");
  };
  const upstream = (): { url: string; headers: Record<string, string>; model: string } => {
    const b = cfg.backend;
    if (b.kind === "openai" && b.baseUrl) return { url: b.baseUrl.replace(/\/$/, ""), headers: b.apiKeyEnv && process.env[b.apiKeyEnv] ? { authorization: `Bearer ${process.env[b.apiKeyEnv]}` } : {}, model: b.model ?? "default" };
    if (b.kind === "llama-server" && o.backend?.state === "ready") return { url: o.backend.baseUrl, headers: {}, model: "local" };
    throw new HttpError(503, "no model is available to answer: configure a backend and start it", { error: { message: "no model backend is ready", type: "unavailable" } });
  };

  const rootFor = async (dir: unknown): Promise<string> => {
    if (typeof dir !== "string" || dir === "") throw new HttpError(400, "dir is required");
    await Promise.all(ready);
    let real: string;
    try { real = await normaliseRoot(dir); } catch (e) { throw new HttpError(400, e instanceof Error ? e.message : String(e)); }
    for (const r of roots) if (isInside(r, real)) return real;
    throw new HttpError(403, `${real} has not been opened in Clutor. Open it in the app, or start the server with --dir.`);
  };

  const planFor = (m: Awaited<ReturnType<typeof resolveModel>>, b: z.infer<typeof PlanBody>): MemoryPlan =>
    planMemory(m.layout, {
      hostRamBytes: (b.hostRamGiB ?? host.totalRamBytes / GiB) * GiB,
      ...(b.ramCeilingGiB ?? cfg.memory.ramCeilingGiB ? { ramCeilingBytes: (b.ramCeilingGiB ?? (cfg.memory.ramCeilingGiB as number)) * GiB } : {}),
      ...(cfg.memory.osHeadroomGiB !== undefined ? { osHeadroomBytes: cfg.memory.osHeadroomGiB * GiB } : {}),
      contextTokens: b.contextTokens ?? cfg.memory.contextTokens, kvBits: b.kvBits ?? cfg.memory.kvBits,
      ...((b.diskGBps ?? cfg.memory.diskGBps) ? { diskBytesPerSec: (b.diskGBps ?? (cfg.memory.diskGBps as number)) * GiB, diskMeasured: b.diskMeasured ?? cfg.memory.diskGBps !== undefined } : {}),
    });

  const DecideBody = z.object({
    prompt: z.string().min(1).max(200_000), choices: z.array(z.string().min(1).max(200)).min(2).max(26), model: z.string().max(200).optional(),
  }).strict();

  const PlanBody = z.object({
    model: z.string().max(1000), hostRamGiB: z.number().min(1).max(4096).optional(), ramCeilingGiB: z.number().min(1).max(4096).optional(),
    contextTokens: z.number().int().min(256).max(2_000_000).optional(), kvBits: z.union([z.literal(16), z.literal(8), z.literal(4)]).optional(),
    diskGBps: z.number().min(0.01).max(100).optional(), diskMeasured: z.boolean().optional(),
  }).strict();

  async function route(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname;
    const method = req.method ?? "GET";
    // A web page on another site must not be able to talk to this server through the user's browser, even by DNS rebinding.
    if (!LOCAL_HOST.test(req.headers.host ?? "")) throw new HttpError(421, "this server only answers on localhost");
    const origin = req.headers.origin;
    const cors: Record<string, string> = {};
    if (origin) {
      if (!allowedOrigin(origin)) throw new HttpError(403, `the origin ${origin} is not allowed; add it to corsOrigins in clutor.config.json`);
      Object.assign(cors, { "access-control-allow-origin": origin, vary: "Origin", "access-control-allow-headers": "authorization, content-type, x-clutor-token", "access-control-allow-methods": "GET, POST, DELETE, OPTIONS" });
    }
    if (method === "OPTIONS") { res.writeHead(204, cors); return void res.end(); }
    const reply = (status: number, body: unknown): void => send(res, status, body, cors);

    /* ── public ── */
    if (method === "GET" && path === "/api/health") return reply(200, { ok: true, version: o.version, backend: { kind: cfg.backend.kind, state: o.backend?.state ?? (cfg.backend.kind === "openai" ? "external" : "none") } });
    if (method === "GET" && path === "/api/hardware") return reply(200, host);
    if (method === "GET" && path === "/api/catalog") return reply(200, { models: CATALOG.map((c) => ({ id: c.id, name: c.name, experts: c.experts, expertsUsed: c.expertsUsed, layers: c.layers, publishedParamsB: c.publishedParamsB, source: c.source })), quants: QUANTS });

    if (method === "POST" && path === "/api/plan") {
      const b = parse(PlanBody, await readBody(req));
      const isFile = b.model.toLowerCase().endsWith(".gguf");
      if (isFile) needAuth(req);
      let m;
      try { m = await resolveModel(b.model, { modelsDir: resolveModelsDir(), allowFiles: isFile }); } catch (e) { throw new HttpError(e instanceof ModelError || e instanceof GgufError ? 400 : 500, e instanceof Error ? e.message : String(e)); }
      const plan = planFor(m, b);
      const gates = evaluateGates({ hostRamBytes: (b.hostRamGiB ?? host.totalRamBytes / GiB) * GiB, plan, cache: { policy: cfg.cache.policy }, smartRoute: { enabled: cfg.cache.smartRoute.enabled, maxDeltaP: cfg.cache.smartRoute.maxDeltaP }, ...(m.gguf ? { gguf: m.gguf } : {}) });
      return reply(200, { model: { id: m.id, name: m.name, source: m.source, quant: m.quant, approximate: m.layout.approximate, totalBytes: m.layout.totalBytes, layers: m.layout.layerCount, experts: m.layout.expertCount, expertsUsed: m.layout.expertsUsed }, plan, gates });
    }

    if (method === "POST" && path === "/api/simulate") {
      const b = parse(z.object({ model: z.string().max(100), quant: z.string().max(20).optional(), cacheGiB: z.number().min(0).max(4096), tokens: z.number().int().min(20).max(1500).default(300), locality: z.number().min(0).max(0.99).default(0.9), diskGBps: z.number().min(0.01).max(100).default(3), computeMsPerLayer: z.number().min(0).max(1000).default(0.5), seed: z.number().int().default(1), smartRoute: z.boolean().default(false) }).strict(), await readBody(req));
      const spec = CATALOG.find((c) => c.id === b.model);
      if (!spec || spec.experts === 0) throw new HttpError(400, "pick a mixture-of-experts model from the catalog");
      const bpw = QUANTS[b.quant ?? "Q4_K_M"];
      if (bpw === undefined) throw new HttpError(400, "unknown quantisation");
      const layout = approximateLayout(spec, bpw);
      const moe = layout.layers.filter((l) => l.experts.length).map((l) => l.index);
      const trace = generateRoutingTrace({ moeLayers: moe, experts: spec.experts, used: spec.expertsUsed, tokens: b.tokens, seed: b.seed, locality: b.locality });
      const base = { cacheBytes: b.cacheGiB * GiB, disk: { bytesPerSec: b.diskGBps * GiB, requestLatencySec: 100e-6 }, computePerLayerSec: b.computeMsPerLayer / 1000 };
      const baseline = simulateStreaming(layout, trace, { ...BASELINE, ...base });
      const clutor = simulateStreaming(layout, trace, { ...CLUTOR_DEFAULTS, smartRoute: { ...CLUTOR_DEFAULTS.smartRoute, enabled: b.smartRoute }, ...base });
      return reply(200, { synthetic: true, note: "A synthetic routing trace: it shows how each policy behaves on a workload with this locality, not how any real model routes.", expertBytes: expertSizeBytes(layout), baseline, clutor });
    }

    /* ── model API: OpenAI and Ollama compatible ── */
    if (method === "GET" && path === "/v1/models") {
      const u = upstream();
      return reply(200, { object: "list", data: [{ id: u.model, object: "model", created: 0, owned_by: "clutor" }] });
    }
    if (method === "POST" && path === "/v1/decide") {
      // It signs a receipt into the project's log, so unlike the model passthrough it needs the access token.
      needAuth(req);
      const b = parse(DecideBody, await readBody(req));
      const u = upstream();
      const model = u.model === "local" ? "local" : b.model ?? u.model;
      const authorization = u.headers["authorization"];
      const startedAt = new Date().toISOString();
      let result;
      try { result = await runDecide({ baseUrl: u.url, model, ...(authorization ? { apiKey: authorization.replace(/^Bearer /, "") } : {}) }, b.prompt, b.choices, f); }
      catch (e) { if (e instanceof DecideError) throw new HttpError(e.status, e.message); throw e; }
      const root = roots.values().next().value as string | undefined;
      const receipt = root ? await sealDecision(root, model, b.prompt, b.choices, result, startedAt) : null;
      return reply(200, { choice: result.choice, probabilities: result.probabilities, probabilityMassOnChoices: result.probabilityMassOnChoices, latencyMs: result.latencyMs, receipt });
    }
    if (method === "POST" && path === "/v1/chat/completions") {
      const u = upstream();
      const body = (await readBody(req)) as Record<string, unknown>;
      const r = await f(`${u.url}/chat/completions`, { method: "POST", headers: { "content-type": "application/json", ...u.headers }, body: JSON.stringify({ ...body, model: u.model === "local" ? (body["model"] ?? "local") : u.model }) });
      res.writeHead(r.status, { "content-type": r.headers.get("content-type") ?? "application/json", "cache-control": "no-store", ...cors });
      if (!r.body) return void res.end();
      const stream = Readable.fromWeb(r.body as never);
      req.on("close", () => stream.destroy());
      stream.pipe(res);
      return;
    }
    if (method === "GET" && path === "/api/tags") {
      const u = (() => { try { return upstream(); } catch { return null; } })();
      const local = await listLocalModels(resolveModelsDir()).catch(() => []);
      return reply(200, { models: [...(u ? [{ name: u.model, model: u.model, modified_at: new Date().toISOString(), size: 0, digest: "", details: { format: "gguf" } }] : []), ...local.map((m) => ({ name: m.file, model: m.file, modified_at: new Date().toISOString(), size: m.bytes, digest: "", details: { format: "gguf" } }))] });
    }
    if (method === "POST" && (path === "/api/chat" || path === "/api/generate")) {
      const kind = path === "/api/chat" ? "chat" : "generate";
      const req0 = (await readBody(req)) as OllamaChatRequest;
      const u = upstream();
      const model = req0.model ?? u.model;
      const stream = req0.stream !== false;
      const r = await f(`${u.url}/chat/completions`, { method: "POST", headers: { "content-type": "application/json", ...u.headers }, body: JSON.stringify({ ...toOpenAiBody(req0, u.model === "local" ? model : u.model, kind), ...(stream ? { stream_options: { include_usage: true } } : {}) }) });
      if (!r.ok) throw new HttpError(r.status === 404 ? 404 : 502, `the model server answered ${r.status}: ${(await r.text()).slice(0, 200)}`);
      if (!stream) return reply(200, fromOpenAiReply(kind, model, (await r.json()) as never));
      res.writeHead(200, { "content-type": "application/x-ndjson", "cache-control": "no-store", ...cors });
      for await (const line of streamToOllama(kind, model, r.body as ReadableStream<Uint8Array>)) { if (!res.write(line)) await new Promise((x) => res.once("drain", x)); }
      return void res.end();
    }

    /* ── everything below touches files, processes or the model's lifecycle: token required ── */
    if (path.startsWith("/api/")) needAuth(req);

    if (method === "GET" && path === "/api/models") return reply(200, { local: await listLocalModels(resolveModelsDir()), catalog: CATALOG.map((c) => c.id) });
    if (method === "GET" && path === "/api/roots") return reply(200, { roots: [...roots] });
    if (method === "POST" && path === "/api/roots") {
      const b = parse(z.object({ dir: z.string().min(1).max(2000) }).strict(), await readBody(req));
      try { return reply(200, { root: await addRoot(b.dir) }); } catch (e) { throw new HttpError(400, e instanceof Error ? e.message : String(e)); }
    }
    if (method === "GET" && path === "/api/backend") return reply(200, { kind: cfg.backend.kind, state: o.backend?.state ?? "n/a", error: o.backend?.error ?? null, log: o.backend?.recentLog.slice(-40) ?? [] });
    if (method === "POST" && path === "/api/backend/stop") { await o.backend?.stop(); return reply(200, { state: o.backend?.state ?? "n/a" }); }
    if (method === "POST" && path === "/api/backend/start") {
      if (!o.startBackend || !o.backend) throw new HttpError(409, "this server was not set up to start a model: set backend.kind to llama-server and give it a model file");
      if (o.backend.state === "ready" || o.backend.state === "starting") return reply(200, { state: o.backend.state });
      // Loading a large model takes minutes: answer now and let the client poll /api/backend.
      void o.startBackend().catch((e: unknown) => o.log?.(`backend: ${e instanceof Error ? e.message : String(e)}`));
      return reply(202, { state: "starting" });
    }
    if (method === "POST" && path === "/api/bench") {
      const b = parse(z.object({ model: z.string().max(2000).optional() }).strict(), await readBody(req));
      let file = b.model;
      let cleanup: (() => Promise<void>) | undefined;
      if (file) { try { file = (await resolveModel(file, { modelsDir: resolveModelsDir() })).path; } catch (e) { throw new HttpError(400, e instanceof Error ? e.message : String(e)); } }
      else {
        const { mkdtemp, rm } = await import("node:fs/promises");
        const { tmpdir } = await import("node:os");
        const dir = await mkdtemp(`${tmpdir()}/clutor-bench-`);
        cleanup = () => rm(dir, { recursive: true, force: true });
        file = `${dir}/bench.gguf`;
        await writeSyntheticGguf(file, { arch: "qwen2moe", name: "bench-moe", layers: 8, embedding: 1024, heads: 16, kvHeads: 4, contextLength: 4096, expertCount: 32, expertsUsed: 4, expertFfn: 1024, denseFfn: 0, vocab: 4096, sharedExperts: 0 });
      }
      try {
        const r = await fileReader(file as string);
        let layout;
        try { layout = buildModelLayout(await parseGguf(r)); } finally { await r.close(); }
        if (!layout.expertCount) throw new HttpError(400, "this model has no experts to read");
        const m = await measureExpertReads(file as string, layout, host.totalRamBytes, 96);
        return reply(200, { measurement: m, synthetic: b.model === undefined });
      } finally { await cleanup?.(); }
    }

    if (method === "POST" && path === "/api/mcp") {
      const r = await handleMcp((await readBody(req)) as never, { isRoot: (d) => rootFor(d).catch(() => null) });
      return r.status === 202 ? (res.writeHead(202, cors), void res.end()) : reply(r.status, r.body);
    }

    if (method === "POST" && path === "/api/agent/run") {
      const b = parse(z.object({ dir: z.string().min(1), task: z.string().min(1).max(20_000) }).strict(), await readBody(req));
      const dir = await rootFor(b.dir);
      const client = llm();
      const project = new NodeProject(dir);
      const s = new AgentSession(dir, b.task, project, new ShadowWorkspace(project), cfg, host.totalRamBytes);
      sessions.set(s.id, s);
      if (sessions.size > 50) for (const [k, v] of sessions) { if (v.status !== "running") { sessions.delete(k); break; } }
      s.start(client);
      return reply(202, { id: s.id });
    }
    const m = /^\/api\/agent\/(task_[a-f0-9]+)(?:\/([a-z]+))?$/.exec(path);
    if (m) {
      const s = sessions.get(m[1] as string);
      if (!s) throw new HttpError(404, "no such task");
      const action = m[2];
      if (method === "GET" && !action) return reply(200, await s.info());
      if (method === "GET" && action === "events") {
        res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-store", connection: "keep-alive", ...cors });
        const from = Number(url.searchParams.get("from") ?? 0) || 0;
        const stop = s.subscribe(from, (e) => res.write(`data: ${JSON.stringify(e)}\n\n`), () => res.end());
        req.on("close", stop);
        return;
      }
      if (method === "POST" && action === "approve") {
        const b = parse(z.object({ requestId: z.string(), allow: z.boolean() }).strict(), await readBody(req));
        return s.resolveApproval(b.requestId, b.allow) ? reply(200, { ok: true }) : reply(404, { error: "no such approval request" });
      }
      if (method === "POST" && action === "stop") { s.stop(); return reply(200, { ok: true }); }
      if (method === "POST" && action === "apply") {
        const b = parse(z.object({ force: z.boolean().default(false) }).strict(), await readBody(req));
        try { return reply(200, await s.apply(b.force)); } catch (e) {
          if (e instanceof ConflictError) return reply(409, { error: e.message, conflicts: e.conflicts });
          throw new HttpError(400, e instanceof Error ? e.message : String(e));
        }
      }
      if (method === "POST" && action === "rollback") {
        const b = parse(z.object({ force: z.boolean().default(false) }).strict(), await readBody(req));
        try { await s.rollback(b.force); return reply(200, { ok: true }); } catch (e) {
          if (e instanceof ConflictError) return reply(409, { error: e.message, conflicts: e.conflicts });
          throw new HttpError(400, e instanceof Error ? e.message : String(e));
        }
      }
    }
    if (method === "GET" && path === "/api/receipts") {
      const dir = await rootFor(url.searchParams.get("dir"));
      return reply(200, { receipts: (await readReceipts(dir)).slice(-100), verification: await verifyReceiptLog(dir), checkpoints: await listCheckpoints(new NodeProject(dir)) });
    }
    throw new HttpError(404, "no such endpoint");
  }

  function resolveModelsDir(): string {
    return cfg.models.startsWith("/") || /^[A-Za-z]:/.test(cfg.models) ? cfg.models : `${o.root}/${cfg.models}`;
  }
  async function addRoot(dir: string): Promise<string> {
    const real = await normaliseRoot(dir);
    roots.add(real);
    return real;
  }

  const http = createHttpServer((req, res) => {
    route(req, res).catch((e: unknown) => {
      const status = e instanceof HttpError ? e.status : 500;
      if (status === 500) o.log?.(`error: ${e instanceof Error ? e.stack ?? e.message : String(e)}`);
      if (res.headersSent) return void res.destroy();
      send(res, status, { error: e instanceof HttpError ? e.message : "internal error", ...(e instanceof HttpError ? e.extra : {}) });
    });
  });
  http.headersTimeout = 15_000;
  http.requestTimeout = 0; // chat and event streams are long-lived

  return {
    http, token, sessions, addRoot,
    listen: () => new Promise((resolve, reject) => {
      http.once("error", reject);
      http.listen(cfg.port, cfg.host, () => { const a = http.address() as AddressInfo; boundPort = a.port; resolve({ host: a.address, port: a.port }); });
    }),
    close: async () => {
      for (const s of sessions.values()) s.stop();
      await new Promise<void>((r) => { http.close(() => r()); http.closeAllConnections(); });
      await o.backend?.stop();
    },
  };
}
