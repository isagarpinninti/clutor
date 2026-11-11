import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  BASELINE, CATALOG, CLUTOR_DEFAULTS, GiB, QUANTS, ShadowWorkspace, OpenAiCompatibleClient, buildModelLayout, evaluateGates, expertSizeBytes, fmtGiB, generateRoutingTrace, parseGguf, planMemory, runAgent, simulateStreaming, validateBounds, approximateLayout, fmtBytes, buildReport, emptyResults, type BenchRecord,
  keyThumbprint, sealReceipt, verifyReceiptChain, type AgentEvent, type LlmClient, type PublicJwk, type Receipt,
} from "@clutor/core";
import { NodeProject, applyChanges, diskModelFrom, fileReader, hostInfo, lastReceiptTag, listCheckpoints, loadOrCreateReceiptSigner, measureExpertReads, appendReceipt, readReceipts, rollbackCheckpoint, loadPolicy, sandboxMode, detectCapabilities, scratchRunner, verifyReceiptLog, writeSyntheticGguf, ConflictError } from "@clutor/core/node";
import { UsageError, flagBool, flagInt, flagString, parseArgs } from "./args";
import { agentLlm, ensureSlotDir } from "./agentllm";
import { runDecide } from "./decide";
import { BackendManager, buildLlamaArgs, findLlamaServer, probeLlamaServer } from "./backend";
import { CONFIG_FILE, ConfigError, DEFAULT_PORT, loadConfig, type ClutorConfig } from "./config";
import { formatGates, formatPlan, formatReport, formatSimulation, pct, table, tps } from "./format";
import { EngineError, buildEngine, installPrebuilt, run, engineChatBench, engineInfo, engineRun, engineStream, findEngine, type StreamOptions, type StreamReport } from "./engine";
import { DOWNLOADS, downloadFile, downloadUrl, listLocalModels, resolveModel } from "./models";
import { createClutorServer } from "./server";
import { discoverModels, modelFolders } from "./tui/discover";

export interface Io {
  out(line: string): void;
  /** Writes text with no newline added (a model's reply as it is generated). Falls back to `out` when absent. */
  write?(text: string): void;
  err(line: string): void;
  cwd: string;
  /** Asks the user a question and returns the answer. Absent when there is no terminal. */
  prompt?(question: string): Promise<string>;
  /** Opens the full-screen interactive chat (tui/app.ts). Present only on a real terminal; tests and pipes use the plain session. */
  tui?(run: (argv: string[], session: SessionState) => Promise<number>, version: string, argv: string[]): Promise<number>;
}

/** State that survives from one task to the next inside an interactive session (see repl.ts). One-shot commands run without it. */
export interface SessionState {
  /** The model server, started on the first task and kept for the rest of the session. */
  backend?: BackendManager;
  /** Check to run after each task; set with /verify. It is the user's own command, so it runs without asking. */
  verify?: string;
  /** Run commands without asking (/yes). */
  auto: boolean;
  /** Checkpoint of the last applied task, for /undo. */
  lastCheckpoint?: { id: string; dir: string };
  /** One line per finished task, given to the next task so the agent knows what already happened. */
  notes: string[];
  /** Tokens the model server reported, summed over the session (measured, not estimated). */
  usage: { promptTokens: number; completionTokens: number };
}

export const version = (): string => {
  try {
    const here = dirname(fileURLToPath(import.meta.url));
    return (JSON.parse(readFileSync(join(here, "..", "package.json"), "utf8")) as { version: string }).version;
  } catch { return "0.0.0"; }
};

const HELP = `clutor: run very large models from disk on small machines, and a coding agent that works on your project

Usage: clutor <command> [options]
       clutor                      In a terminal: an interactive session. Type a task, or /help for commands.

  init [dir]                         Create clutor.config.json and a models folder
  models [dir]                       List the model catalog and the .gguf files in your models folder
  pull [id] [--yes]                  List the small test models you can download, or download one into your models folder
  inspect <file.gguf>                Read a model file's header: architecture, layers, experts, sizes
  plan <model> [--ram GiB] [--ctx N] [--kv 16|8|4] [--disk GiB/s] [--quant Q4_K_M] [--json]
                                     Will it fit? Where does the memory go? How fast could it possibly run?
  simulate <model> [--cache GiB] [--tokens N] [--locality 0.9] [--disk GiB/s] [--compute-ms N] [--smart-route] [--json]
                                     Replay a routing trace through the expert cache: baseline LRU against clutor
  bench [--file model.gguf] [--engine] [--cache-mb N] [--tokens N] [--compute kernel|checksum]
                                     Measure real disk reads on this machine; with the native engine, also compare cache and prefetch strategies
  stream <model.gguf> [--tokens N] [--cache-mb N] [--policy lru|wtinylfu|adaptive] [--smart J,M,TAU|off] [--prefetch on|off] [--dense auto|resident|ring] [--compute kernel|checksum] [--trace FILE] [--json]
                                     Stream a model's experts from disk through the native engine (synthetic routing; measured I/O and cache)
  generate <model.gguf> "<prompt>" [--n N] [--cache-mb N] [--smart J,M,TAU] [--prefetch off|prev|ahead] [--trace FILE]
                                     Generate text with the native engine's real forward pass (olmoe models). With --cache-mb the experts stream from disk through a cache of that size
  quality <model.gguf> <text-file> --cache-mb N [--smart J,M,TAU] [--max-tokens N] [--json]
                                     Perplexity, KL divergence and top-token agreement of a streamed (optionally smart-route) run against exact routing
  engine install|build|where         Install the prebuilt native engine (checked against a SHA-256), or compile it from c/ (needs a C compiler), or show which one is used
  code "<task>" [--dir folder] [--apply] [--yes] [--verify "npm test"] [--max-steps N]
                                     Run the coding agent on a folder; review the diff before anything is written
  serve [dir] [--port ${DEFAULT_PORT}] [--folder dir]...   Start the local server: OpenAI and Ollama compatible API, agent, MCP
  backend [--model file.gguf]        Find llama-server, see which flags it supports and how clutor would start it
  status [dir]                       One screen: machine, model backend, memory plan, receipts
  chat [--prompt "<text>"] [--system "<text>"]   Talk to the configured model in the terminal (one-shot with --prompt, else a REPL)
  decide --prompt "<text>" --choices "a,b,c" [--json]   One forward pass, no chat loop: probability of each choice's first token
  receipts [dir]                     Verify the agent's receipt log and list checkpoints (also: ledger verify|list)
  receipt verify <clt_token|file> [dir] [--key-id id|--pubkey file]   Check a receipt log offline; receipt key shows your key id
  rollback <checkpoint> [--dir folder] [--force]   Undo an applied task
  doctor [dir]                       Check this machine and the configuration

Models are catalog ids (${CATALOG.map((c) => c.id).join(", ")}) or the path of a .gguf file.
Common options: --config <file>  --json  -h, --help  -v, --version
`;

const BOOLEANS = ["json", "help", "version", "apply", "yes", "force", "smart-route", "no-start", "engine", "no-os-cache", "measure-disk"] as const;

function cfgFor(args: ReturnType<typeof parseArgs>, io: Io, dirArg?: string) {
  return loadConfig(dirArg ? resolve(io.cwd, dirArg) : io.cwd, flagString(args, "config"));
}

export async function main(argv: string[], io: Io, session?: SessionState): Promise<number> {
  const args = parseArgs(argv, BOOLEANS);
  const [command, ...rest] = args.positional;
  const json = flagBool(args, "json");
  try {
    if (flagBool(args, "version") || command === "version" || argv.includes("-v")) return io.out(version()), 0;
    if (!flagBool(args, "help") && !argv.includes("-h") && io.tui && !session && (!command || (command === "chat" && flagString(args, "prompt") === undefined))) return await io.tui((a, s) => main(a, io, s), version(), argv.filter((x) => x !== "chat"));
    if (!command && !flagBool(args, "help") && !argv.includes("-h") && io.prompt && !session) return await (await import("./repl")).runRepl(io, (a, s) => main(a, io, s), (cfg, what) => engineSlashCommand(io, cfg, what), version());
    if (!command || flagBool(args, "help") || command === "help" || argv.includes("-h")) return io.out(HELP), 0;

    switch (command) {
      case "init": return await init(io, rest[0]);
      case "models": return await models(io, args, rest[0], json);
      case "stream": return await streamCmd(io, args, rest[0], json);
      case "generate": return await generateCmd(io, args, rest[0], rest.slice(1).join(" "));
      case "quality": return await qualityCmd(io, args, rest[0], rest[1], json);
      case "engine": return await engineCmd(io, rest[0]);
      case "pull": return await pull(io, args, rest[0]);
      case "inspect": return await inspect(io, rest[0], json);
      case "plan": return await plan(io, args, rest[0], json);
      case "simulate": return await simulate(io, args, rest[0], json);
      case "bench": return await bench(io, args, json);
      case "code": return await code(io, args, rest.join(" "), session);
      case "serve": return await serve(io, args, rest[0]);
      case "backend": return await backendCmd(io, args);
      case "receipts": return await receipts(io, args, rest[0], json);
      case "ledger": return await receipts(io, args, rest[1], json);
      case "receipt": return await receiptVerify(io, args, rest[0], rest[1], rest[2]);
      case "status": return await status(io, args, rest[0], json);
      case "chat": return await chat(io, args);
      case "decide": return await decide(io, args, json);
      case "rollback": return await rollback(io, args, rest[0]);
      case "doctor": return await doctor(io, args, rest[0], json);
      default: throw new UsageError(`unknown command "${command}"; run clutor --help`);
    }
  } catch (e) {
    if (e instanceof UsageError || e instanceof ConfigError) { io.err(e.message); return 2; }
    io.err(e instanceof Error ? e.message : String(e));
    return 1;
  }
}

/* ───────────── commands ───────────── */

function init(io: Io, dirArg?: string): number {
  const dir = resolve(io.cwd, dirArg ?? ".");
  mkdirSync(join(dir, "models"), { recursive: true });
  const path = join(dir, CONFIG_FILE);
  if (existsSync(path)) { io.err(`${path} already exists; leaving it alone`); return 1; }
  writeFileSync(path, `${JSON.stringify({ port: DEFAULT_PORT, models: "models", backend: { kind: "none" }, memory: { contextTokens: 8192, kvBits: 16 }, cache: { policy: "adaptive" }, agent: { toolMode: "json", maxSteps: 30 } }, null, 2)}\n`);
  const h = hostInfo();
  io.out(`Created ${path} and ${join(dir, "models")}.`);
  io.out(`This machine: ${h.platformName} ${h.arch}, ${h.cpuModel}, ${fmtGiB(h.totalRamBytes)} RAM.`);
  io.out("");
  io.out("Next: put a .gguf file in the models folder, then");
  io.out("  clutor plan <file.gguf>          to see whether and how it fits");
  io.out("  clutor backend --model <file>    to see how llama-server would be started for it");
  return 0;
}

async function models(io: Io, args: ReturnType<typeof parseArgs>, dirArg: string | undefined, json: boolean): Promise<number> {
  const { config, root } = cfgFor(args, io, dirArg);
  const local = await listLocalModels(resolve(root, config.models));
  const found = discoverModels(root, config.models);
  if (json) { io.out(JSON.stringify({ local, found: found.map((m) => ({ file: m.path, bytes: m.bytes })), catalog: CATALOG.map((c) => c.id) }, null, 2)); return 0; }
  io.out("Catalog (architectures you can plan for before downloading anything):");
  io.out(table(CATALOG.map((c) => [c.id, c.name, `${c.publishedParamsB}B`, c.experts ? `${c.experts} experts, ${c.expertsUsed} used` : "dense"])));
  io.out("");
  io.out(found.length ? `Model files found:\n${table(found.map((m) => [m.path, fmtBytes(m.bytes)]))}` : `No .gguf files found. Looked in: ${modelFolders(root, config.models).join(", ")}.`);
  return 0;
}

async function inspect(io: Io, file: string | undefined, json: boolean): Promise<number> {
  if (!file) throw new UsageError("usage: clutor inspect <file.gguf>");
  const path = resolve(io.cwd, file);
  const r = await fileReader(path);
  try {
    const g = await parseGguf(r);
    const l = buildModelLayout(g);
    validateBounds(g);
    if (json) { io.out(JSON.stringify({ version: g.version, tensors: g.tensorCount, arch: l.arch, name: l.name, layers: l.layerCount, experts: l.expertCount, expertsUsed: l.expertsUsed, bytes: l.totalBytes, quantisations: l.quantisations, residentBytes: l.residentBytes, fileBytes: g.fileSize }, null, 2)); return 0; }
    io.out(`${l.name} (${l.arch}), GGUF v${g.version}, ${g.tensorCount} tensors, ${fmtBytes(g.fileSize)}`);
    io.out(`${l.layerCount} layers${l.expertCount ? `, ${l.expertCount} experts per layer (${l.expertsUsed} used per token${l.sharedExperts ? `, ${l.sharedExperts} shared` : ""})` : ", dense"}`);
    io.out(`Always needed in RAM: ${fmtBytes(l.residentBytes)}   Per-layer dense weights: ${fmtBytes(l.layers.reduce((s, x) => s + x.denseBytes, 0))}   Experts: ${fmtBytes(l.layers.reduce((s, x) => s + x.experts.length * x.expertBytes, 0))}`);
    io.out(`Quantisations: ${Object.entries(l.quantisations).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${fmtBytes(v)}`).join(", ")}`);
    io.out(`KV cache: ${l.kv.kvHeads} KV heads x ${l.kv.headDim} dims, up to ${l.kv.maxContext.toLocaleString("en-US")} tokens`);
    if (l.unsizedTensors) io.out(`! ${l.unsizedTensors} tensors use a quantisation this reader cannot size.`);
    io.out("All tensors lie inside the file.");
    return 0;
  } finally { await r.close(); }
}

function numFlag(args: ReturnType<typeof parseArgs>, name: string, min: number, max: number): number | undefined {
  const v = flagString(args, name);
  if (v === undefined) return undefined;
  const n = Number(v);
  if (!Number.isFinite(n) || n < min || n > max) throw new UsageError(`--${name} must be a number from ${min} to ${max}`);
  return n;
}

async function plan(io: Io, args: ReturnType<typeof parseArgs>, ref: string | undefined, json: boolean): Promise<number> {
  if (!ref) throw new UsageError("usage: clutor plan <model> [--ram GiB]");
  const { config, root } = cfgFor(args, io);
  const quant = flagString(args, "quant");
  const m = await resolveModel(ref.toLowerCase().endsWith(".gguf") ? resolve(io.cwd, ref) : quant ? `${ref}@${quant}` : ref, { modelsDir: resolve(root, config.models) });
  const h = hostInfo();
  const ram = numFlag(args, "ram", 1, 4096) ?? config.memory.ramCeilingGiB;
  let disk = numFlag(args, "disk", 0.01, 100) ?? config.memory.diskGBps;
  let measuredDisk = false;
  // --measure-disk times the real file with the engine's own read path and plans from that, instead of an assumed or typed-in speed.
  if (flagBool(args, "measure-disk")) {
    if (!ref.toLowerCase().endsWith(".gguf")) throw new UsageError("--measure-disk needs a .gguf file on this computer: clutor plan <file.gguf> --measure-disk");
    const bin = findEngine();
    if (!bin) throw new UsageError("--measure-disk needs the native engine: run `clutor engine install` (or `clutor engine build`)");
    const r = await run(bin, ["diskbench", resolve(io.cwd, ref), "--seconds", "6"], { timeoutMs: 120_000 });
    if (r.code !== 0) throw new UsageError(`could not time the disk: ${r.stderr.trim() || r.stdout.trim()}`);
    const b = JSON.parse(r.stdout) as { sequentialBytesPerSec: number; scatteredBytesPerSec: number };
    disk = b.scatteredBytesPerSec / GiB;
    measuredDisk = true;
    io.out(`Measured on this computer: ${(b.sequentialBytesPerSec / 1e9).toFixed(2)} GB/s sequential, ${(b.scatteredBytesPerSec / 1e9).toFixed(2)} GB/s scattered (1 MiB reads, the way experts are fetched). Planning from the scattered figure.`);
    if (b.sequentialBytesPerSec > 12e9 || b.scatteredBytesPerSec > 9e9) io.out("! That is faster than a disk can be: the system probably served the file from memory. Try a file that has not been read recently.");
  }
  const kv = flagInt(args, "kv", 4, 16);
  if (kv !== undefined && ![4, 8, 16].includes(kv)) throw new UsageError("--kv must be 16, 8 or 4");
  // --ram describes the machine to plan for (an 8 GiB laptop, say): the operating system's share comes out of it. The config's ramCeilingGiB stays a ceiling on this machine.
  const ramFlag = numFlag(args, "ram", 1, 4096);
  const hostRam = ramFlag !== undefined ? ramFlag * GiB : h.totalRamBytes;
  const p = planMemory(m.layout, {
    hostRamBytes: hostRam,
    ...(ramFlag === undefined && ram ? { ramCeilingBytes: ram * GiB } : {}), ...(config.memory.osHeadroomGiB !== undefined ? { osHeadroomBytes: config.memory.osHeadroomGiB * GiB } : {}),
    contextTokens: flagInt(args, "ctx", 256, 2_000_000) ?? config.memory.contextTokens, kvBits: (kv as 16 | 8 | 4 | undefined) ?? config.memory.kvBits,
    ...(disk ? { diskBytesPerSec: disk * GiB, diskMeasured: measuredDisk || flagString(args, "disk") !== undefined || config.memory.diskGBps !== undefined } : {}),
  });
  const gates = evaluateGates({ hostRamBytes: hostRam, plan: p, cache: { policy: config.cache.policy }, smartRoute: { enabled: config.cache.smartRoute.enabled, maxDeltaP: config.cache.smartRoute.maxDeltaP }, ...(m.gguf ? { gguf: m.gguf } : {}) });
  const report = buildReport(m.layout, p);
  if (json) { io.out(JSON.stringify({ model: { id: m.id, name: m.name, approximate: m.layout.approximate }, plan: p, report, gates }, null, 2)); return p.fits ? 0 : 1; }
  io.out(formatReport(m.name, m.quant, report, { ram: `${(hostRam / GiB).toFixed(0)} GiB`, disk: `${(disk ?? p.disk.bytesPerSec / GiB).toFixed(1)} GiB/s ${measuredDisk || flagString(args, "disk") !== undefined || config.memory.diskGBps !== undefined ? "(given or measured)" : "(assumed)"}`, cpu: `${h.cpuModel} (${h.logicalCores} cores)` }));
  io.out("");
  io.out("DETAILS");
  io.out("-".repeat(32));
  io.out(formatPlan(m.name, m.quant, p));
  io.out("");
  io.out(formatGates(gates));
  return p.fits ? 0 : 1;
}

async function simulate(io: Io, args: ReturnType<typeof parseArgs>, id: string | undefined, json: boolean): Promise<number> {
  if (!id) throw new UsageError("usage: clutor simulate <model> --cache <GiB>");
  const spec = CATALOG.find((c) => c.id === id);
  if (!spec || spec.experts === 0) throw new UsageError(`pick a mixture-of-experts model: ${CATALOG.filter((c) => c.experts).map((c) => c.id).join(", ")}`);
  const quant = flagString(args, "quant") ?? "Q4_K_M";
  const bpw = QUANTS[quant];
  if (bpw === undefined) throw new UsageError(`unknown quantisation ${quant}`);
  const layout = approximateLayout(spec, bpw);
  const cacheGiB = numFlag(args, "cache", 0, 4096) ?? 4;
  const tokens = flagInt(args, "tokens", 20, 3000) ?? 300;
  const moe = layout.layers.filter((l) => l.experts.length).map((l) => l.index);
  const trace = generateRoutingTrace({ moeLayers: moe, experts: spec.experts, used: spec.expertsUsed, tokens, seed: flagInt(args, "seed", 0, 2 ** 31) ?? 1, locality: numFlag(args, "locality", 0, 0.99) ?? 0.9 });
  const base = { cacheBytes: cacheGiB * GiB, disk: { bytesPerSec: (numFlag(args, "disk", 0.01, 100) ?? 3) * GiB, requestLatencySec: 100e-6 }, computePerLayerSec: (numFlag(args, "compute-ms", 0, 1000) ?? 0.5) / 1000 };
  const b = simulateStreaming(layout, trace, { ...BASELINE, ...base });
  const c = simulateStreaming(layout, trace, { ...CLUTOR_DEFAULTS, smartRoute: { ...CLUTOR_DEFAULTS.smartRoute, enabled: flagBool(args, "smart-route") }, ...base });
  if (json) { io.out(JSON.stringify({ synthetic: true, baseline: b, clutor: c }, null, 2)); return 0; }
  io.out(`${spec.name} (${quant}), ${cacheGiB} GiB expert cache, ${tokens} tokens, expert size ${(expertSizeBytes(layout) / 2 ** 20).toFixed(1)} MiB`);
  io.out(formatSimulation(b, c));
  io.out("");
  io.out("This replays a SYNTHETIC routing trace (skewed expert popularity, neighbouring tokens that pick similar experts). It shows how each policy");
  io.out("behaves on a workload like that; it is not a measurement of any real model, and speed is modelled from the disk speed you give it.");
  return 0;
}

async function writeBenchRecord(io: Io, args: ReturnType<typeof parseArgs>, o: { recordPath: string; file: string | undefined; path: string; layout: { name: string; arch: string; expertCount: number; expertsUsed: number }; fileBytes: number; disk: { bytesPerSec: number; samples: number; likelyCold: boolean } | undefined; bin: string | undefined; startedAt: string }): Promise<void> {
  const h = hostInfo();
  const { recordPath, file, path, layout, bin, startedAt } = o;
  const caps = detectCapabilities();
  const notes: string[] = [];
  const results = emptyResults();
  if (o.disk) results.diskReadBytesPerSec = { value: o.disk.bytesPerSec, basis: "measured", runs: o.disk.samples };
  else notes.push("Disk read speed is not recorded for a model without experts (the test reads expert-sized pieces).");
  let promptText: string | null = null, genTokens: number | null = null;
  if (!file) notes.push("Synthetic test model: it cannot generate text, so only the disk read speed is recorded.");
  else if (!bin) notes.push("Native engine not installed: generation was not measured.");
  else {
    promptText = "The harbour town woke slowly. Fishermen carried nets down to the boats while gulls argued over scraps on the quay, and the baker propped her door open so the smell of warm bread drifted along the lane. ".repeat(3);
    genTokens = flagInt(args, "tokens", 8, 4096) ?? 64;
    try {
      io.err(`Measuring generation on ${path} (one fresh engine process, ${genTokens} greedy tokens)...`);
      const c = await engineChatBench(bin, path, { prompt: promptText, maxTokens: genTokens });
      Object.assign(results, { loadSeconds: { value: c.loadSeconds, basis: "measured", runs: 1 }, firstTokenSeconds: { value: c.firstTokenSeconds, basis: "measured", runs: 1 }, generationTokensPerSec: { value: c.generationTokensPerSec, basis: "measured", runs: 1 }, promptTokensPerSec: { value: c.promptTokensPerSec, basis: "measured", runs: 1 } });
      genTokens = c.generated;
      notes.push(`One run of ${c.generated} tokens after a ${c.promptTokens}-token prompt. The file cache was not dropped, so load time is for a warm file unless it was not read recently.`);
    } catch (e) { notes.push(`Generation was not measured: ${e instanceof Error ? e.message : String(e)}`); }
  }
  notes.push("Cache hit rate and peak memory are not recorded by this command (null).");
  if (o.disk) notes.unshift(o.disk.likelyCold ? "File larger than 1.5x RAM: most reads came from the SSD." : "File not much larger than RAM: reads may have been served from the operating system's cache, so the disk figure is an upper bound.");
  const rec: BenchRecord = {
    schema: 1, clutorVersion: version(), createdAt: startedAt,
    machine: { os: h.platformName, osRelease: h.osRelease, arch: h.arch, cpu: h.cpuModel, cores: h.logicalCores, ramBytes: h.totalRamBytes, cpuFeatures: caps.cpuFeatures, accelerators: caps.accelerators },
    model: { name: layout.name, arch: layout.arch, quantization: null, fileBytes: o.fileBytes, experts: layout.expertCount, expertsUsed: layout.expertsUsed, synthetic: !file },
    runtime: { engine: bin ?? null, contextTokens: file && bin ? 1024 : null, cacheBytes: null, cachePolicy: null, prompt: promptText, tokens: genTokens },
    results, notes,
  };
  writeFileSync(resolve(io.cwd, recordPath), `${JSON.stringify(rec, null, 2)}\n`);
  io.err(`Wrote ${recordPath}`);
}

async function bench(io: Io, args: ReturnType<typeof parseArgs>, json: boolean): Promise<number> {
  const h = hostInfo();
  const file = flagString(args, "file");
  let path: string;
  let cleanup: (() => void) | undefined;
  if (file) path = resolve(io.cwd, file);
  else {
    const dir = mkdtempSync(join(tmpdir(), "clutor-bench-"));
    cleanup = () => rmSync(dir, { recursive: true, force: true });
    path = join(dir, "bench.gguf");
    io.err("No --file given: writing a test model of about 1 GiB to the temp folder (it will be deleted).");
    await writeSyntheticGguf(path, { arch: "qwen2moe", name: "bench-moe", layers: 8, embedding: 1024, heads: 16, kvHeads: 4, contextLength: 4096, expertCount: 32, expertsUsed: 4, expertFfn: 1024, denseFfn: 0, vocab: 4096, sharedExperts: 0 });
  }
  try {
    const r = await fileReader(path);
    const layout = buildModelLayout(await parseGguf(r));
    await r.close();
    const recordPath = flagString(args, "record");
    if (!layout.expertCount) {
      if (!recordPath || !file) throw new UsageError("this model has no experts; bench measures expert-sized reads (with --file and --record it still records generation speed)");
      await writeBenchRecord(io, args, { recordPath, file, path, layout, fileBytes: statSync(path).size, disk: undefined, bin: findEngine(), startedAt: new Date().toISOString() });
      return 0;
    }
    const m = await measureExpertReads(path, layout, h.totalRamBytes, 96);
    const dm = diskModelFrom(m, expertSizeBytes(layout), layout.layers.find((l) => l.experts.length)?.experts[0]?.length ?? 1);
    const bin = findEngine();
    if (flagBool(args, "engine") && !bin) throw new UsageError("the native engine is not installed: run `clutor engine install` (prebuilt) or `clutor engine build` (needs a C compiler)");
    const startedAt = new Date().toISOString();
    const cmp = bin ? await compareStrategies(bin, path, flagInt(args, "cache-mb", 1, 1_000_000) ?? Math.max(64, Math.floor(h.totalRamBytes / 2 ** 20 / 16)), flagInt(args, "tokens", 10, 100_000) ?? 300, (flagString(args, "dense") as StreamOptions["dense"] | undefined), computeFlag(args)) : undefined;
    if (recordPath) await writeBenchRecord(io, args, { recordPath, file, path, layout, fileBytes: m.fileBytes, disk: { bytesPerSec: m.bytesPerSec, samples: m.samples, likelyCold: m.likelyCold }, bin, startedAt });
    if (json) { io.out(JSON.stringify({ measurement: m, diskModel: dm, ...(cmp ? { engine: cmp } : {}) }, null, 2)); return 0; }
    io.out(`${m.samples} random expert reads from ${path} (${fmtBytes(m.fileBytes)}):`);
    io.out(table([["throughput", `${(m.bytesPerSec / 2 ** 30).toFixed(2)} GiB/s`], ["mean per expert", `${(m.meanReadSec * 1000).toFixed(2)} ms`], ["95th percentile", `${(m.p95ReadSec * 1000).toFixed(2)} ms`], ["modelled request overhead", `${(dm.requestLatencySec * 1e6).toFixed(0)} us`]]));
    io.out("");
    io.out(m.likelyCold ? "The file is larger than 1.5x your RAM, so most reads came from the SSD." : `! The file (${fmtBytes(m.fileBytes)}) is not much bigger than your RAM (${fmtGiB(h.totalRamBytes)}), so the operating system's cache probably served many of these reads. Treat this as an upper bound; run it on your real model file for a true figure.`);
    io.out(`Use it: clutor plan <model> --disk ${(m.bytesPerSec / 2 ** 30).toFixed(1)}`);
    if (cmp) io.out(`\n${formatComparison(cmp)}`);
    if (cmp && flagBool(args, "receipt")) {
      const dir = resolve(io.cwd, flagString(args, "dir") ?? ".");
      const signer = await loadOrCreateReceiptSigner(dir);
      const finishedAt = new Date().toISOString();
      const brief = cmp.map((x) => `${x.name}: hit ${(x.report.totals.hitRate * 100).toFixed(1)}%, ${x.report.totals.missesPerToken.toFixed(1)} misses/token, ${x.report.totals.tokensPerSecond.toFixed(2)} passes/s`).join(" | ");
      const first = cmp[0]!.report;
      const rc = await sealReceipt(signer, { v: 2, id: `bench_${Date.now().toString(36)}`, task: `bench ${path.split("/").pop()} (${first.model.layers} layers x ${first.model.experts} experts, top-${first.model.expertsUsed}, ${first.config.cacheSlots} cached, ${first.config.tokens} tokens, synthetic routing, peak RSS ${(Math.max(...cmp.map((x) => x.report.rss.peakBytes)) / 2 ** 20).toFixed(0)} MiB) -> ${brief}`, model: first.model.name, startedAt, finishedAt, status: "finished", files: [], commands: [], steps: cmp.length, previous: await lastReceiptTag(dir) });
      await appendReceipt(dir, rc);
      io.out(`\nSigned receipt ${rc.tag}\nVerify it with: clutor receipt verify ${rc.tag} ${dir}`);
    }
    if (!cmp) io.out("\nThe native engine is not built, so cache and prefetch strategies were not compared. Build it with: clutor engine build");
    return 0;
  } finally { cleanup?.(); }
}

/** Starts llama-server if the configuration asks for it, and returns the manager. */
export async function startBackendIfConfigured(config: ClutorConfig, io: Io, existing?: BackendManager): Promise<BackendManager | undefined> {
  if (config.backend.kind !== "llama-server") return undefined;
  const ls = config.backend.llamaServer;
  if (!ls.model) { io.err("backend.llamaServer.model is not set, so no model is loaded."); return undefined; }
  const binary = ls.binary ?? findLlamaServer();
  if (!binary) throw new ConfigError("llama-server was not found. Install llama.cpp (for example `brew install llama.cpp`) or set backend.llamaServer.binary.");
  const caps = await probeLlamaServer(binary);
  const r = await fileReader(ls.model);
  let layout;
  try { layout = buildModelLayout(await parseGguf(r)); } finally { await r.close(); }
  const h = hostInfo();
  const p = planMemory(layout, { hostRamBytes: h.totalRamBytes, contextTokens: config.memory.contextTokens, kvBits: config.memory.kvBits });
  const { args, notes } = buildLlamaArgs({ modelPath: ls.model, port: ls.port, contextTokens: config.memory.contextTokens, kvBits: config.memory.kvBits, ...(ls.threads ? { threads: ls.threads } : {}), plan: p, caps, extraArgs: ls.extraArgs, ...(config.agent.prefixSnapshots ? { slotSavePath: ensureSlotDir() } : {}) });
  for (const n of notes) io.err(`note: ${n}`);
  io.err(`Loading ${ls.model} (this can take minutes for a large model)...`);
  const b = existing ?? new BackendManager(ls.port);
  await b.start(binary, args);
  io.err("Model ready.");
  return b;
}

async function serve(io: Io, args: ReturnType<typeof parseArgs>, dirArg?: string): Promise<number> {
  const { config, root } = cfgFor(args, io, dirArg);
  const port = flagInt(args, "port", 0, 65535);
  const folders = [...new Set([root, ...(flagString(args, "folder") ? [resolve(io.cwd, flagString(args, "folder") as string)] : [])])];
  const backend = config.backend.kind === "llama-server" ? new BackendManager(config.backend.llamaServer.port) : undefined;
  if (backend && !flagBool(args, "no-start")) await startBackendIfConfigured(config, io, backend);
  const server = createClutorServer({
    root, config: { ...config, ...(port !== undefined ? { port } : {}) }, version: version(), ...(backend ? { backend, startBackend: async () => void (await startBackendIfConfigured(config, { ...io, out: () => undefined }, backend)) } : {}), roots: folders, log: (l) => io.err(l),
  });
  const a = await server.listen();
  const base = `http://${a.host === "::" ? "localhost" : a.host}:${a.port}`;
  io.out(`Clutor ${version()} listening on ${base}`);
  io.out(`  OpenAI API   ${base}/v1       Ollama API   ${base}`);
  io.out(`  Agent, MCP   ${base}/api/agent/run   ${base}/api/mcp   (need the token below)`);
  io.out(`  Token        ${server.token}`);
  io.out(`  Folders      ${folders.join(", ")}`);
  io.out(`  Backend      ${config.backend.kind}${backend?.state === "ready" ? " (loaded)" : ""}`);
  await new Promise<void>((res) => { const stop = (): void => res(); process.once("SIGINT", stop); process.once("SIGTERM", stop); });
  await server.close();
  return 0;
}

async function backendCmd(io: Io, args: ReturnType<typeof parseArgs>): Promise<number> {
  const { config } = cfgFor(args, io);
  const binary = config.backend.llamaServer.binary ?? findLlamaServer();
  if (!binary) { io.err("llama-server was not found on PATH. Install llama.cpp (for example `brew install llama.cpp`), or set backend.llamaServer.binary."); return 1; }
  const caps = await probeLlamaServer(binary);
  io.out(`llama-server: ${binary}`);
  io.out(`Supports: ${["--jinja", "--cache-type-k", "--flash-attn", "--mlock", "--no-mmap"].map((f) => `${f} ${caps.flags.has(f) ? "yes" : "no"}`).join(", ")}`);
  const model = flagString(args, "model") ?? config.backend.llamaServer.model;
  if (!model) { io.out("Pass --model <file.gguf> to see the command clutor would run."); return 0; }
  const r = await fileReader(resolve(io.cwd, model));
  let layout;
  try { layout = buildModelLayout(await parseGguf(r)); } finally { await r.close(); }
  const p = planMemory(layout, { hostRamBytes: hostInfo().totalRamBytes, contextTokens: config.memory.contextTokens, kvBits: config.memory.kvBits });
  const { args: a, notes } = buildLlamaArgs({ modelPath: resolve(io.cwd, model), port: config.backend.llamaServer.port, contextTokens: config.memory.contextTokens, kvBits: config.memory.kvBits, plan: p, caps, extraArgs: config.backend.llamaServer.extraArgs });
  io.out(`\n${binary} ${a.map((x) => (/\s/.test(x) ? JSON.stringify(x) : x)).join(" ")}`);
  for (const n of notes) io.out(`note: ${n}`);
  return 0;
}

/** The conversation has to fit the model's context window, not a fixed 60,000 characters: about 2.5 characters per token leaves room for the reply. */
function contextCharsFor(contextTokens: number): number { return Math.floor(contextTokens * 2.5); }

async function code(io: Io, args: ReturnType<typeof parseArgs>, task: string, session?: SessionState): Promise<number> {
  if (!task.trim()) throw new UsageError('usage: clutor code "<what to do>" [--dir folder]');
  const { config, root } = cfgFor(args, io);
  const dir = resolve(io.cwd, flagString(args, "dir") ?? ".");
  if (!existsSync(dir)) throw new UsageError(`${dir} does not exist`);
  const backend = session ? (session.backend ??= await startBackendIfConfigured(config, io)) : await startBackendIfConfigured(config, io);
  try {
    const llm = llmFrom(config, backend);
    const project = new NodeProject(dir);
    const ws = new ShadowWorkspace(project);
    const policy = loadPolicy(dir);
    const wantsAuto = flagBool(args, "yes") || config.agent.autoApproveProjectCommands || session?.auto === true;
    if (wantsAuto && policy.forbidAutoApprove) io.err(`note: ${policy.source} forbids running commands without asking; you will be asked.`);
    const auto = wantsAuto && !policy.forbidAutoApprove;
    if (auto) io.err("note: commands that run project code will run without asking (--yes). Only do this for projects you trust.");
    const key = await loadOrCreateReceiptSigner(dir);
    const verifyFlag = flagString(args, "verify") ?? session?.verify;
    const verify = verifyFlag ?? config.agent.verifyCommand;
    const show = (e: AgentEvent): void => {
      if (e.type === "thought") io.out(`  ${e.text.split("\n")[0]?.slice(0, 200)}`);
      else if (e.type === "tool_call") io.out(`> ${e.name} ${JSON.stringify(e.args).slice(0, 160)}`);
      else if (e.type === "tool_result") io.out(`  ${e.ok ? "ok" : "failed"}: ${e.output.split("\n")[0]?.slice(0, 200)}`);
      else if (e.type === "invalid") io.out(`  (invalid reply: ${e.error})`);
      else if (e.type === "verify") io.out(`  verify ${!e.ran ? "NOT RUN" : e.ok ? "passed" : "FAILED"}${e.attempt > 0 ? ` (after ${e.attempt} retr${e.attempt === 1 ? "y" : "ies"})` : ""}: ${e.command}${!e.ran ? ` (${e.output.split("\n")[0]})` : ""}`);
    };
    const result = await runAgent({
      task, llm, ws, maxSteps: flagInt(args, "max-steps", 1, 200) ?? config.agent.maxSteps, contextChars: contextCharsFor(config.memory.contextTokens), ...(verify ? { verifyCommand: verify } : {}), ...(session?.notes.length ? { systemExtra: `Earlier in this session (for context only):\n${session.notes.slice(-6).map((n) => `- ${n}`).join("\n")}` } : {}), onEvent: show,
      runCommand: scratchRunner(project, ws, { timeoutMs: config.agent.commandTimeoutSeconds * 1000 }),
      approve: async ({ command, network }) => {
        if (auto) return true;
        // The command typed after --verify is the user's own choice, so that exact command line runs (in the scratch copy) without a prompt. A verify command that only came from the config file is not pre-approved.
        if (verifyFlag !== undefined && command === verifyFlag) return true;
        if (!io.prompt) return false;
        const a = await io.prompt(`Run \`${command}\`${network ? " (uses the network)" : ""} in a scratch copy of the project? [y/N] `);
        return /^y(es)?$/i.test(a.trim());
      },
      receipt: { signer: key, previous: await lastReceiptTag(dir) },
    });
    if (result.receipt) await appendReceipt(dir, result.receipt);
    if (session) {
      session.usage.promptTokens += result.usage.promptTokens;
      session.usage.completionTokens += result.usage.completionTokens;
      session.notes.push(`"${task.slice(0, 120)}" -> ${result.status}: ${result.summary.slice(0, 200)}`);
    }
    io.out("");
    io.out(`${result.status}: ${result.summary}`);
    for (const d of result.diffs) io.out(`\n${d.patch || `${d.status} ${d.path}`}`);
    io.out(`\n${result.diffs.length} file${result.diffs.length === 1 ? "" : "s"} changed, ${result.steps} steps, ${result.invalidReplies} invalid replies, ${result.violations} blocked attempts.`);
    if (result.verification) io.out(`Final check \`${result.verification.command}\`: ${!result.verification.ran ? "NOT RUN" : result.verification.ok ? "passed" : "FAILED"}${result.verification.retries ? ` (${result.verification.retries} self-heal ${result.verification.retries === 1 ? "turn" : "turns"})` : ""}`);
    if (result.diffs.length === 0) return result.status === "finished" ? 0 : 1;
    // With a check requested, nothing is written unless the check passed on the final state.
    if (verifyFlag !== undefined && !(result.verification?.ok === true)) {
      io.out(`Nothing was written: the check \`${verifyFlag}\` ${result.verification && !result.verification.ran ? "was not run" : "did not pass"}. The diff above is what the agent proposed.`);
      return 1;
    }
    let apply = flagBool(args, "apply");
    if (!apply && io.prompt) apply = /^y(es)?$/i.test((await io.prompt("Apply these changes to the folder? [y/N] ")).trim());
    if (!apply) { io.out("Nothing was written. Run again with --apply to write the changes."); return result.status === "finished" ? 0 : 1; }
    try {
      const r = await applyChanges(project, await ws.changeSet());
      if (session) session.lastCheckpoint = { id: r.id, dir };
      io.out(`Applied ${r.applied.length} files. Undo with: ${session ? "/undo" : `clutor rollback ${r.id} --dir ${dir}`}`);
    } catch (e) {
      if (e instanceof ConflictError) { io.err(`${e.message}\nNothing was written.`); return 1; }
      throw e;
    }
    return result.status === "finished" ? 0 : 1;
  } finally { if (!session) await backend?.stop(); void root; }
}

async function receipts(io: Io, args: ReturnType<typeof parseArgs>, dirArg: string | undefined, json: boolean): Promise<number> {
  const dir = resolve(io.cwd, dirArg ?? flagString(args, "dir") ?? ".");
  const v = await verifyReceiptLog(dir);
  const all = await readReceipts(dir);
  const cps = await listCheckpoints(new NodeProject(dir));
  if (json) { io.out(JSON.stringify({ verification: v, receipts: all, checkpoints: cps }, null, 2)); return v.ok ? 0 : 1; }
  io.out(v.ok ? `Receipt chain intact: ${v.count} receipt${v.count === 1 ? "" : "s"}, hash chain and ECDSA signatures verified${v.keyIds.length ? ` (key id ${v.keyIds.join(", ")})` : ""}.` : `Receipt chain BROKEN at entry ${v.index}: ${v.reason}`);
  if (all.length) io.out(table(all.slice(-15).map((r) => [r.finishedAt.slice(0, 19), r.status, `${r.files.length} files`, r.task.slice(0, 50)]), ["finished", "status", "changed", "task"]));
  if (cps.length) io.out(`\nCheckpoints:\n${table(cps.slice(0, 10).map((c) => [c.id, c.createdAt.slice(0, 19), `${c.files} files`]))}`);
  return v.ok ? 0 : 1;
}

async function rollback(io: Io, args: ReturnType<typeof parseArgs>, id: string | undefined): Promise<number> {
  if (!id) throw new UsageError("usage: clutor rollback <checkpoint id> [--dir folder]");
  const project = new NodeProject(resolve(io.cwd, flagString(args, "dir") ?? "."));
  try {
    const r = await rollbackCheckpoint(project, id, { force: flagBool(args, "force") });
    io.out(`Restored ${r.length} files.`);
    return 0;
  } catch (e) {
    if (e instanceof ConflictError) { io.err(`${e.message}\nUse --force to restore anyway.`); return 1; }
    throw e;
  }
}

async function doctor(io: Io, args: ReturnType<typeof parseArgs>, dirArg: string | undefined, json: boolean): Promise<number> {
  const h = hostInfo();
  const checks: Array<{ level: "ok" | "info" | "warn" | "fail"; text: string }> = [];
  const add = (level: "ok" | "info" | "warn" | "fail", text: string): void => void checks.push({ level, text });
  add(Number(h.nodeVersion.split(".")[0]) >= 20 ? "ok" : "fail", `Node.js ${h.nodeVersion}`);
  add("ok", `${h.platformName} ${h.arch}, ${h.cpuModel}, ${h.logicalCores} cores, ${fmtGiB(h.totalRamBytes)} RAM`);
  if (h.totalRamBytes < 8 * GiB) add("warn", "Under 8 GiB of RAM: only small models will run comfortably, and large mixture-of-experts models will be very slow.");
  let cfg: ClutorConfig | undefined;
  let root = resolve(io.cwd, dirArg ?? ".");
  try { const l = cfgFor(args, io, dirArg); cfg = l.config; root = l.root; add(l.path ? "ok" : "info", l.path ? `Configuration ${l.path}` : `No ${CONFIG_FILE}: using defaults (run clutor init)`); } catch (e) { add("fail", e instanceof Error ? e.message : String(e)); }
  const ls = findLlamaServer(process.env, cfg?.backend.llamaServer.binary ? [dirname(cfg.backend.llamaServer.binary)] : []);
  add(ls ? "ok" : "warn", ls ? `llama-server at ${ls}` : "llama-server not found (install llama.cpp to run models; the planner, simulator and agent tools still work)");
  const caps = detectCapabilities();
  add("info", `CPU features: ${caps.cpuFeatures.length ? caps.cpuFeatures.join(", ") : "none detected"}`);
  for (const a of caps.accelerators) add("info", `${a.kind.toUpperCase()}: ${a.name} (${a.api}) is present; Clutor does not use it yet`);
  for (const n of caps.notes) if (!n.startsWith("The engine computes")) add("info", n);
  if (!caps.accelerators.length) add("info", "No GPU or NPU detected; inference runs on the CPU, which is all the engine uses today.");
  add("info", "Disk speed is not measured here: run `clutor bench --file <model.gguf>` or `clutor plan <model.gguf> --measure-disk`.");
  const sbx = sandboxMode();
  add(sbx === "none" ? "warn" : "ok", sbx === "none" ? "No operating-system sandbox for the agent's commands (macOS needs sandbox-exec, Linux needs bwrap installed; Windows has none). They still run in a scratch copy with the command policy." : `Agent commands run in an OS sandbox (${sbx}): writes only in the scratch copy, no credentials folders, no network except installs.`);
  const eng = findEngine();
  add(eng ? "ok" : "info", eng ? `Native engine at ${eng}` : "Native engine not built (clutor engine build): cache and prefetch strategies cannot be measured without it");
  if (cfg) {
    const local = await listLocalModels(resolve(root, cfg.models));
    add(local.length ? "ok" : "info", `${local.length} .gguf file${local.length === 1 ? "" : "s"} in ${resolve(root, cfg.models)}`);
    if (cfg.backend.kind === "none") add("info", "No model backend configured: chat and the agent need one (backend.kind in the configuration).");
    if (cfg.backend.kind === "openai" && !cfg.backend.baseUrl) add("fail", "backend.kind is openai but backend.baseUrl is missing");
    if (cfg.backend.kind === "llama-server" && !cfg.backend.llamaServer.model) add("fail", "backend.kind is llama-server but backend.llamaServer.model is missing");
    if (cfg.agent.autoApproveProjectCommands) add("warn", "agent.autoApproveProjectCommands is on: commands that run project code run without asking.");
    if (cfg.host !== "127.0.0.1" && cfg.host !== "localhost") add("warn", `The server listens on ${cfg.host}: anyone who can reach it and learns the token can have the agent edit files.`);
  }
  if (json) { io.out(JSON.stringify(checks, null, 2)); return checks.some((c) => c.level === "fail") ? 1 : 0; }
  const tag = { ok: "ok  ", info: "info", warn: "warn", fail: "FAIL" } as const;
  for (const c of checks) (c.level === "fail" ? io.err : io.out)(`${tag[c.level]}  ${c.text}`);
  return checks.some((c) => c.level === "fail") ? 1 : 0;
}

export { pct, tps };

function modelEndpoint(config: ClutorConfig, backend?: BackendManager): { baseUrl: string; model: string; apiKey?: string } {
  const b = config.backend;
  if (b.kind === "openai" && b.baseUrl) return { baseUrl: b.baseUrl, model: b.model ?? "default", ...(b.apiKeyEnv && process.env[b.apiKeyEnv] ? { apiKey: process.env[b.apiKeyEnv] as string } : {}) };
  if (backend) return { baseUrl: backend.baseUrl, model: "local" };
  throw new ConfigError('no model is configured. In clutor.config.json set backend.kind to "openai" (with baseUrl) or "llama-server" (with llamaServer.model).');
}
function llmFrom(config: ClutorConfig, backend?: BackendManager): LlmClient {
  const e = modelEndpoint(config, backend);
  return agentLlm(config, { ...e, local: config.backend.kind === "llama-server" });
}

async function chat(io: Io, args: ReturnType<typeof parseArgs>): Promise<number> {
  const { config } = cfgFor(args, io);
  const backend = await startBackendIfConfigured(config, io);
  try {
    const llm = new OpenAiCompatibleClient({ ...modelEndpoint(config, backend), toolMode: "json" });
    const history: Array<{ role: "system" | "user" | "assistant"; content: string }> = [];
    const system = flagString(args, "system");
    if (system) history.push({ role: "system", content: system });
    const turn = async (text: string): Promise<void> => {
      history.push({ role: "user", content: text });
      const t0 = Date.now();
      const r = await llm.chat(history);
      const reply = r.content ?? "";
      history.push({ role: "assistant", content: reply });
      io.out(reply);
      const secs = (Date.now() - t0) / 1000;
      const gen = (r as { usage?: { completionTokens?: number } }).usage?.completionTokens;
      io.err(`[${secs.toFixed(1)} s${gen ? `, ${gen} tokens, ${(gen / secs).toFixed(2)} tok/s measured end to end` : ""}]`);
    };
    const one = flagString(args, "prompt");
    if (one !== undefined) { await turn(one); return 0; }
    if (!io.prompt) throw new UsageError("chat needs a terminal, or --prompt \"<text>\"");
    io.out("clutor chat. /help lists the commands, /exit leaves.");
    for (;;) {
      const line = (await io.prompt("clt> ")).trim();
      if (!line) continue;
      if (line === "/exit" || line === "/quit") return 0;
      if (line === "/layers" || line === "/cache") { try { await engineSlashCommand(io, config, line.slice(1) as "layers" | "cache"); } catch (e) { io.err(e instanceof Error ? e.message : String(e)); } continue; }
      if (line === "/help") { io.out("/layers  where the model file's bytes are, per layer\n/cache   a streaming pass over the model file: cache, prefetch and disk waits (native engine)\n/reset   clear the conversation\n/exit    leave"); continue; }
      if (line === "/reset") { history.length = system ? 1 : 0; io.out("(conversation cleared)"); continue; }
      try { await turn(line); } catch (e) { io.err(e instanceof Error ? e.message : String(e)); }
    }
  } finally { await backend?.stop(); }
}

/** One forward pass, one token: the model's probability for each choice's letter. Not calibrated; it is the model's own first-token distribution. */
async function decide(io: Io, args: ReturnType<typeof parseArgs>, json: boolean): Promise<number> {
  const prompt = flagString(args, "prompt");
  const choices = (flagString(args, "choices") ?? "").split(",").map((c) => c.trim()).filter(Boolean);
  if (!prompt || choices.length < 2 || choices.length > 26) throw new UsageError('usage: clutor decide --prompt "<text>" --choices "approve,reject,escalate"  (2 to 26 choices)');
  const { config } = cfgFor(args, io);
  const backend = await startBackendIfConfigured(config, io);
  try {
    const out = await runDecide(modelEndpoint(config, backend), prompt, choices);
    if (json) io.out(JSON.stringify(out, null, 2));
    else {
      io.out(`Decision: ${out.choice}`);
      choices.forEach((c) => io.out(`  ${c.padEnd(Math.max(...choices.map((x) => x.length)))}  ${"#".repeat(Math.round((out.probabilities[c] as number) * 30)).padEnd(30)} ${pct(out.probabilities[c] as number)}`));
      io.out(`(the model put ${pct(out.probabilityMassOnChoices)} of its first-token probability on these letters; not a calibrated confidence; ${out.latencyMs} ms)`);
    }
    return 0;
  } finally { await backend?.stop(); }
}

async function status(io: Io, args: ReturnType<typeof parseArgs>, dirArg: string | undefined, json: boolean): Promise<number> {
  const h = hostInfo();
  const { config, root, path } = cfgFor(args, io, dirArg);
  const local = await listLocalModels(resolve(root, config.models));
  const ls = findLlamaServer(process.env, config.backend.llamaServer.binary ? [dirname(config.backend.llamaServer.binary)] : []);
  const v = await verifyReceiptLog(root);
  const s = { host: `${h.platformName} ${h.arch}, ${h.cpuModel}, ${h.logicalCores} cores`, ramBytes: h.totalRamBytes, configuration: path ?? null, backend: config.backend.kind, llamaServer: ls ?? null, models: local.length, receipts: v };
  if (json) { io.out(JSON.stringify(s, null, 2)); return 0; }
  io.out(`Host        ${s.host}, ${fmtGiB(h.totalRamBytes)} RAM`);
  io.out(`Config      ${path ?? "none (defaults)"}`);
  io.out(`Backend     ${config.backend.kind}${ls ? `, llama-server at ${ls}` : ", llama-server not found"}`);
  io.out(`Models      ${local.length} .gguf file${local.length === 1 ? "" : "s"} in ${resolve(root, config.models)}`);
  io.out(`Receipts    ${v.ok ? `intact (${v.count})` : `BROKEN at entry ${v.index}`}`);
  return 0;
}

async function receiptVerify(io: Io, args: ReturnType<typeof parseArgs>, sub: string | undefined, target: string | undefined, dirArg: string | undefined): Promise<number> {
  if (sub === "key") {
    const dir = resolve(io.cwd, target ?? flagString(args, "dir") ?? ".");
    const signer = await loadOrCreateReceiptSigner(dir);
    io.out(`Key id  ${await keyThumbprint(signer.publicJwk)}`);
    io.out(`Public key (share this; keep ${join(".clutor", "keys")} private):\n${JSON.stringify(signer.publicJwk, null, 2)}`);
    return 0;
  }
  if (sub !== "verify" || !target) throw new UsageError("usage: clutor receipt verify <clt_token | receipts.jsonl> [dir] [--key-id <id> | --pubkey <file.json>]   or   clutor receipt key [dir]");
  // Trust: a key id you recorded earlier, or a public key file. Without either, the log is only checked for internal consistency.
  let trusted: string[] | undefined;
  const keyId = flagString(args, "key-id");
  const pubkey = flagString(args, "pubkey");
  if (keyId) trusted = [keyId];
  if (pubkey) trusted = [...(trusted ?? []), await keyThumbprint(JSON.parse(readFileSync(resolve(io.cwd, pubkey), "utf8")) as PublicJwk)];
  const isToken = target.startsWith("clt_");
  let all: Receipt[];
  if (isToken) all = await readReceipts(resolve(io.cwd, dirArg ?? flagString(args, "dir") ?? "."));
  else all = readFileSync(resolve(io.cwd, target), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as Receipt);
  const v = await verifyReceiptChain(all, trusted ? { trustedKeyIds: trusted } : {});
  if (!v.ok) { io.err(`INVALID at entry ${v.index}: ${v.reason}`); return 1; }
  const found = isToken ? all.find((r) => r.tag === target) : undefined;
  if (isToken && !found) { io.err("That receipt is not in this log."); return 1; }
  io.out(isToken && found ? `Valid: ${found.status} "${found.task.slice(0, 60)}" at ${found.finishedAt}, ${found.files.length} files, signed by ${found.keyId}.` : `Valid: ${all.length} receipt${all.length === 1 ? "" : "s"}, chain and ECDSA signatures check out.`);
  if (isToken && found) { const { tag, ...body } = found; io.out(`Decoded payload (this is exactly what the signature covers):\n${JSON.stringify(body, null, 2)}\nSignature tag: ${tag}`); }
  io.out(`Signed by key${v.keyIds.length === 1 ? "" : "s"}: ${v.keyIds.join(", ") || "none"}`);
  io.out(trusted ? "The signing key is one you trust." : "No --key-id or --pubkey given: this shows the log is consistent and unedited, not whose key signed it.");
  return 0;
}

async function pull(io: Io, args: ReturnType<typeof parseArgs>, id: string | undefined): Promise<number> {
  const { config, root } = cfgFor(args, io);
  const dir = resolve(root, config.models);
  if (!id) {
    io.out(table(DOWNLOADS.map((d) => [d.id, fmtBytes(d.bytes), d.repo, d.note]), ["id", "size", "from", "what it is"]));
    io.out("\nDownload one with: clutor pull <id>");
    return 0;
  }
  const d = DOWNLOADS.find((x) => x.id === id);
  if (!d) throw new UsageError(`unknown download "${id}"; run clutor pull to see the list`);
  const dest = join(dir, d.file);
  if (existsSync(dest)) { io.out(`${dest} is already there.`); return 0; }
  if (!flagBool(args, "yes")) {
    if (!io.prompt) throw new UsageError(`this downloads ${fmtBytes(d.bytes)} from huggingface.co; run again with --yes to confirm`);
    if (!/^y(es)?$/i.test((await io.prompt(`Download ${d.file} (${fmtBytes(d.bytes)}) from huggingface.co to ${dir}? [y/N] `)).trim())) { io.out("Nothing downloaded."); return 0; }
  }
  let last = 0;
  const bytes = await downloadFile(downloadUrl(d), dest, { expectedBytes: d.bytes, expectedSha256: d.sha256, onProgress: (done, total) => { const p = Math.floor((done / total) * 10); if (p > last) { last = p; io.err(`  ${p * 10}%`); } } });
  io.out(`Saved ${dest} (${fmtBytes(bytes)}).`);
  return 0;
}

/** The same synthetic routing trace and the same cache memory, run under three strategies. */
function shiftFlag(args: ReturnType<typeof parseArgs>): { shift?: { at: number; len: number } } {
  const v = flagString(args, "shift");
  if (v === undefined) return {};
  const m = /^(\d+),(\d+)$/.exec(v);
  if (!m || Number(m[2]) < 1) throw new UsageError("--shift wants AT,LEN, for example 80,10: measured tokens AT..AT+LEN-1 are uniform scan noise");
  return { shift: { at: Number(m[1]), len: Number(m[2]) } };
}

function computeFlag(args: ReturnType<typeof parseArgs>): StreamOptions["compute"] | undefined {
  const c = flagString(args, "compute");
  if (c !== undefined && c !== "kernel" && c !== "checksum") throw new UsageError("--compute must be kernel or checksum");
  return c;
}

async function compareStrategies(bin: string, file: string, cacheMb: number, tokens: number, dense?: StreamOptions["dense"], compute?: StreamOptions["compute"]): Promise<Array<{ name: string; report: StreamReport }>> {
  const common: StreamOptions = { dense: dense ?? "auto", ...(compute ? { compute } : {}), tokens, warmup: Math.min(50, Math.floor(tokens / 5)), cacheMb, seed: 1, noOsCache: true };
  const runs: Array<[string, StreamOptions]> = [
    ["LRU, no prefetch", { ...common, policy: "lru", prefetch: false, ioThreads: 1 }],
    ["adaptive cache + prefetch", { ...common, policy: "adaptive", prefetch: true, ioThreads: 2 }],
    ["+ smart-route (2,12,0.015)", { ...common, policy: "adaptive", prefetch: true, ioThreads: 2, smart: "2,12,0.015" }],
  ];
  const out: Array<{ name: string; report: StreamReport }> = [];
  for (const [name, o] of runs) out.push({ name, report: await engineStream(bin, file, o) });
  return out;
}

function formatComparison(c: Array<{ name: string; report: StreamReport }>): string {
  const r0 = c[0]?.report;
  if (!r0) return "";
  const gb = (x: StreamReport): string => `${x.totals.gbPerToken.toFixed(3)}`;
  const rows: string[][] = [
    ["hit rate", ...c.map((x) => pct(x.report.totals.hitRate))],
    ["misses per token", ...c.map((x) => x.report.totals.missesPerToken.toFixed(1))],
    ["GB read per token", ...c.map((x) => gb(x.report))],
    ["time waiting for disk (s)", ...c.map((x) => x.report.totals.ioWaitSeconds.toFixed(2))],
    ["wall time (s)", ...c.map((x) => x.report.totals.wallSeconds.toFixed(2))],
    ["passes per second", ...c.map((x) => x.report.totals.tokensPerSecond.toFixed(1))],
    ["scan-pollution rejects (admission filter)", ...c.map((x) => String(x.report.totals.admitRejects))],
    ["prefetch issued / late / unused", ...c.map((x) => `${x.report.totals.prefetchIssued} / ${x.report.totals.prefetchLate} / ${x.report.totals.prefetchWasted}`)],
    ["prefetch hit / late (hits served by prefetch)", ...c.map((x) => `${x.report.totals.stageHits} / ${x.report.totals.prefetchLate}`)],
    ["lookahead overlap (read time hidden)", ...c.map((x) => pct(x.report.totals.overlap))],
    ["disk wait / compute time (s)", ...c.map((x) => `${x.report.totals.ioWaitSeconds.toFixed(2)} / ${x.report.totals.computeSeconds.toFixed(2)}`)],
    ["substitutions per token", ...c.map((x) => (x.report.totals.substituted / Math.max(1, x.report.config.tokens)).toFixed(2))],
    ["peak RSS of the engine (MiB, ru_maxrss)", ...c.map((x) => (x.report.rss.peakBytes / 2 ** 20).toFixed(0))],
    ["substitutions (largest probability moved)", ...c.map((x) => `${x.report.totals.substituted} (${x.report.totals.maxDeltaP.toFixed(4)})`)],
  ];
  return [
    `MEASURED by the native engine on this machine: ${r0.model.name}, ${r0.config.cacheSlots} experts cached (${((r0.config.cacheSlots * r0.model.expertBytes) / 2 ** 20).toFixed(0)} MiB), ${r0.config.tokens} tokens after ${r0.config.warmup} warm-up; dense weights ${r0.config.denseMode === "resident" ? "resident" : "re-read every token"}.`,
    `Disk reads, waits and cache behaviour are real. The routing is SYNTHETIC (a skewed random router with drifting topics), so hit rates here are not what a real model would see. ${r0.config.computeMode === "kernel" ? `The compute is a real Q4_0 x Q8_0 kernel (${r0.config.computeKernel}) run over every loaded byte, read as Q4_0 whatever its true type: genuine CPU work, not this model's maths.` : "The compute is a checksum, so this run measures I/O alone."}`,
    table(rows, ["", ...c.map((x) => x.name)]),
  ].join("\n");
}

/** The engine flags `generate` and `quality` share, checked here so a typo fails before a model is opened. */
function engineFlags(args: ReturnType<typeof parseArgs>): string[] {
  const out: string[] = [];
  const num = (n: string, lo: number, hi: number): void => { const v = flagInt(args, n, lo, hi); if (v !== undefined) out.push(`--${n}`, String(v)); };
  num("cache-mb", 1, 10_000_000); num("io-threads", 1, 16); num("prefetch-n", 1, 64); num("stage", 1, 100_000);
  const pol = flagString(args, "policy");
  if (pol) { if (!["lru", "wtinylfu", "adaptive"].includes(pol)) throw new UsageError("--policy must be lru, wtinylfu or adaptive"); out.push("--policy", pol); }
  const pf = flagString(args, "prefetch");
  if (pf) { if (!["on", "off", "prev", "ahead"].includes(pf)) throw new UsageError("--prefetch must be off, prev or ahead"); out.push("--prefetch", pf); }
  const sm = flagString(args, "smart");
  if (sm && sm !== "off") { if (!/^\d+,\d+,[\d.]+$/.test(sm)) throw new UsageError("--smart wants J,M,TAU, for example 4,16,0.02"); out.push("--smart", sm); }
  if (flagBool(args, "no-os-cache")) out.push("--no-os-cache");
  return out;
}

async function generateCmd(io: Io, args: ReturnType<typeof parseArgs>, file: string | undefined, prompt: string): Promise<number> {
  if (!file || !prompt.trim()) throw new UsageError('usage: clutor generate <model.gguf> "<prompt>" [--n N] [--cache-mb N] [--smart J,M,TAU]');
  const bin = findEngine();
  if (!bin) throw new UsageError("the native engine is not installed: run `clutor engine install` (prebuilt) or `clutor engine build` (needs a C compiler)");
  const path = resolve(io.cwd, file);
  if (!existsSync(path)) throw new UsageError(`${path} does not exist`);
  const n = flagInt(args, "n", 1, 100_000) ?? 64;
  const trace = flagString(args, "trace");
  const argv = ["generate", path, prompt, "--n", String(n), ...engineFlags(args), ...(trace ? ["--trace", resolve(io.cwd, trace)] : [])];
  const r = await engineRun(bin, argv, (s) => (io.write ?? ((t: string) => io.out(t.replace(/\n$/, ""))))(s));
  io.write?.("\n");
  if (r.stderr.trim()) io.err(r.stderr.trim());
  if (r.code !== 0) { io.err(`the engine exited with ${r.code}`); return 1; }
  return 0;
}

async function qualityCmd(io: Io, args: ReturnType<typeof parseArgs>, file: string | undefined, text: string | undefined, json: boolean): Promise<number> {
  if (!file || !text) throw new UsageError("usage: clutor quality <model.gguf> <text-file> --cache-mb N [--smart J,M,TAU] [--max-tokens N]");
  if (flagInt(args, "cache-mb", 1, 10_000_000) === undefined) throw new UsageError("quality compares against a streamed run: give --cache-mb N");
  const bin = findEngine();
  if (!bin) throw new UsageError("the native engine is not installed: run `clutor engine install` (prebuilt) or `clutor engine build` (needs a C compiler)");
  const path = resolve(io.cwd, file), textPath = resolve(io.cwd, text);
  if (!existsSync(path)) throw new UsageError(`${path} does not exist`);
  if (!existsSync(textPath)) throw new UsageError(`${textPath} does not exist`);
  const mt = flagInt(args, "max-tokens", 2, 1_000_000);
  const r = await run(bin, ["quality", path, textPath, ...engineFlags(args), ...(mt ? ["--max-tokens", String(mt)] : [])], { timeoutMs: 3_600_000 });
  if (r.code !== 0) { io.err(r.stderr.trim() || `the engine exited with ${r.code}`); return 1; }
  if (json) { io.out(r.stdout.trim()); return 0; }
  let q: Record<string, number | boolean | string>;
  try { q = JSON.parse(r.stdout) as typeof q; } catch { throw new EngineError("the engine printed something that is not a report"); }
  const n = (k: string): number => q[k] as number;
  io.out([
    `MEASURED  ${n("tokensScored")} tokens scored, teacher-forced`,
    `MEASURED  perplexity ${n("perplexityExact").toFixed(3)} exact routing, ${n("perplexityRun").toFixed(3)} this run (x${n("perplexityRatio").toFixed(4)})`,
    `MEASURED  KL divergence from exact ${n("meanKL").toFixed(4)}; the top token is unchanged at ${(n("top1Agreement") * 100).toFixed(1)}% of positions`,
    q["smart"] ? `MEASURED  smart-route used a resident expert for ${(100 * n("substitutions") / n("expertSlots")).toFixed(1)}% of expert slots; the largest probability moved was ${n("maxProbMoved").toFixed(4)}` : "smart-route off",
    `MEASURED  expert cache hit rate ${(n("hitRate") * 100).toFixed(1)}%, ${n("gbPerToken").toFixed(3)} GB read per token`,
    "The exact reference maps the whole file, so the OS cache is warm: time is not reported here.",
  ].join("\n"));
  return 0;
}

async function streamCmd(io: Io, args: ReturnType<typeof parseArgs>, file: string | undefined, json: boolean): Promise<number> {
  if (!file) throw new UsageError("usage: clutor stream <model.gguf> [--tokens N] [--cache-mb N] [--policy lru|wtinylfu|adaptive] [--smart J,M,TAU|off] [--prefetch on|off]");
  const bin = findEngine();
  if (!bin) throw new UsageError("the native engine is not installed: run `clutor engine install` (prebuilt) or `clutor engine build` (needs a C compiler)");
  const num = (n: string, lo: number, hi: number): number | undefined => flagInt(args, n, lo, hi);
  const pol = flagString(args, "policy");
  if (pol && !["lru", "wtinylfu", "adaptive"].includes(pol)) throw new UsageError("--policy must be lru, wtinylfu or adaptive");
  const dn = flagString(args, "dense");
  if (dn && !["auto", "resident", "ring"].includes(dn)) throw new UsageError("--dense must be auto, resident or ring");
  const pf = flagString(args, "prefetch");
  if (pf && !["on", "off"].includes(pf)) throw new UsageError("--prefetch must be on or off");
  const o: StreamOptions = {
    ...(num("tokens", 1, 1_000_000) !== undefined ? { tokens: num("tokens", 1, 1_000_000) as number } : {}),
    ...(num("warmup", 0, 1_000_000) !== undefined ? { warmup: num("warmup", 0, 1_000_000) as number } : {}),
    ...(num("cache-mb", 1, 10_000_000) !== undefined ? { cacheMb: num("cache-mb", 1, 10_000_000) as number } : {}),
    ...(num("io-threads", 1, 16) !== undefined ? { ioThreads: num("io-threads", 1, 16) as number } : {}),
    ...(num("seed", 0, 2 ** 31) !== undefined ? { seed: num("seed", 0, 2 ** 31) as number } : {}),
    ...(pol ? { policy: pol as StreamOptions["policy"] } : {}),
    ...(flagString(args, "smart") ? { smart: flagString(args, "smart") as string } : {}),
    ...(pf ? { prefetch: pf === "on" } : {}),
    ...(dn ? { dense: dn as StreamOptions["dense"] } : {}),
    ...(computeFlag(args) ? { compute: computeFlag(args) as StreamOptions["compute"] } : {}),
    ...(flagBool(args, "no-os-cache") ? { noOsCache: true } : {}),
    ...(flagString(args, "trace") ? { trace: flagString(args, "trace") as string } : {}),
    ...(num("window", 1, 100_000) !== undefined ? { window: num("window", 1, 100_000) as number } : {}),
    ...shiftFlag(args),
  };
  const path = resolve(io.cwd, file);
  if (!existsSync(path)) throw new UsageError(`${path} does not exist`);
  const r = await engineStream(bin, path, o);
  if (json) { io.out(JSON.stringify(r, null, 2)); return 0; }
  io.out(describeStream(r));
  return 0;
}

function describeStream(r: StreamReport): string {
  const t = r.totals;
  const lines = [
    `${r.model.name} (${r.model.arch}): ${r.model.layers} layers, ${r.model.experts} experts (${r.model.expertsUsed} used), ${(r.model.expertBytes / 2 ** 20).toFixed(2)} MiB each`,
    `policy ${r.config.policy}, ${r.config.cacheSlots} cached + ${r.config.stageSlots} staged experts, smart-route ${r.config.smartRoute ? `on (J=${r.config.J}, M=${r.config.M}, tau=${r.config.tau})` : "off"}, prefetch ${r.config.prefetch ? "on" : "off"}, dense weights ${r.config.denseMode === "resident" ? `resident (${(r.config.densePreloadBytes / 2 ** 20).toFixed(0)} MiB read once)` : "re-read every token"}`,
    `MEASURED  ${((t.demandBytes + t.prefetchBytes + t.denseBytes) / 1e9).toFixed(2)} GB read; waited ${t.ioWaitSeconds.toFixed(2)} s of ${t.wallSeconds.toFixed(2)} s; ${pct(t.overlap)} of read time overlapped with other work`,
    `MEASURED  cache ${pct(t.hitRate)} hits (${t.stageHits} served by prefetch), ${t.missesPerToken.toFixed(1)} misses per token, ${t.admitRejects} admission rejects`,
    `MEASURED  prefetch ${t.prefetchIssued} issued, ${t.prefetchLate} late, ${t.prefetchWasted} never used`,
  ];
  if (r.config.smartRoute) lines.push(`MEASURED  smart-route ${t.substituted} substitutions; the largest probability moved was ${t.maxDeltaP.toFixed(4)} (limit ${r.config.tau})`);
  lines.push(`MEASURED  peak resident memory of the engine process (ru_maxrss): ${(r.rss.peakBytes / 2 ** 20).toFixed(1)} MiB; COMPUTED from the buffers and tables it allocates: ${(r.rss.predictedBytes / 2 ** 20).toFixed(1)} MiB (${(r.rss.startBytes / 2 ** 20).toFixed(1)} MiB of that was there before the run)`);
  if (r.windows.length) {
    const note = r.scenario.shiftLen > 0 ? `; tokens ${r.scenario.shiftAt + 1}-${r.scenario.shiftAt + r.scenario.shiftLen} are uniform scan noise, the rest one tight topic` : "";
    lines.push(`MEASURED  hit rate and peak RSS per window of tokens${note}:\n${table(r.windows.map((w, i) => [`${i ? r.windows[i - 1]!.tokEnd + 1 : 1}-${w.tokEnd}`, pct(w.accesses ? w.hits / w.accesses : 0), `${(w.peakRssBytes / 2 ** 20).toFixed(0)} MiB`]), ["tokens", "hit rate", "peak RSS"])}`);
  }
  lines.push(`${t.tokensPerSecond.toFixed(1)} passes per second over ${r.provenance.routing === "real" ? "REAL (replayed) routing recorded from this model" : "SYNTHETIC routing"}; the compute is ${r.config.computeMode === "kernel" ? `a Q4_0 x Q8_0 kernel (${r.config.computeKernel}) over the loaded bytes (${t.computeSeconds.toFixed(2)} s in total), not this model's maths` : "a checksum"}: this is not model tokens per second.`);
  return lines.join("\n");
}

async function engineCmd(io: Io, sub: string | undefined): Promise<number> {
  if (sub === "where") {
    const bin = findEngine();
    io.out(bin ?? "not built (run: clutor engine build)");
    return bin ? 0 : 1;
  }
  if (sub === "build") {
    io.err("Compiling the native engine...");
    try { io.out(`Built ${await buildEngine()}`); return 0; } catch (e) { if (e instanceof EngineError) { io.err(e.message); return 1; } throw e; }
  }
  if (sub === "install") {
    io.err("Downloading the prebuilt native engine and checking its SHA-256...");
    try { io.out(`Installed ${await installPrebuilt(version())}`); return 0; } catch (e) { if (e instanceof EngineError) { io.err(e.message); return 1; } throw e; }
  }
  throw new UsageError("usage: clutor engine install | build | where");
}

/** `/layers` and `/cache` in the chat REPL. They describe the model FILE and a streaming pass over it; the chat itself runs in llama-server. */
async function engineSlashCommand(io: Io, config: ClutorConfig, what: "layers" | "cache"): Promise<void> {
  const file = config.backend.llamaServer.model;
  const bin = findEngine();
  if (!file) { io.out("No model file is configured (backend.llamaServer.model), so there is nothing to describe."); return; }
  if (!bin) { io.out("The native engine is not built: run `clutor engine build`."); return; }
  if (what === "layers") {
    const info = await engineInfo(bin, file);
    io.out(`${info.name} (${info.arch}): ${info.layers} layers, ${info.experts} experts (${info.expertsUsed} used)`);
    const rows = info.layerTable.map((l, i) => [String(i), `${(l.denseBytes / 2 ** 20).toFixed(1)} MiB`, `${(l.expertBytes / 2 ** 20).toFixed(2)} MiB`, `${(l.routerBytes / 2 ** 10).toFixed(0)} KiB`]);
    io.out(table(rows.length > 24 ? [...rows.slice(0, 12), ["...", "", "", ""], ...rows.slice(-6)] : rows, ["layer", "dense (read every token)", "one expert", "router (resident)"]));
    return;
  }
  io.out("Running a short streaming pass over this model's file (synthetic routing; the chat itself is served by llama-server, which reports no cache data)...");
  const r = await engineStream(bin, file, { tokens: 30, warmup: 5, cacheMb: 256, policy: "adaptive", prefetch: true, ioThreads: 2, noOsCache: true });
  io.out(describeStream(r));
  const worst = [...r.layers].filter((l) => l.accesses > 0).sort((a, b) => b.ioWaitSeconds - a.ioWaitSeconds).slice(0, 5);
  io.out(`Layers that waited longest: ${worst.map((l) => `${l.layer} (${l.ioWaitSeconds.toFixed(2)} s, ${pct(l.hits / l.accesses)} hits)`).join(", ")}`);
}
