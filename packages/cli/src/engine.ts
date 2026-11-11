import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { accessSync, chmodSync, constants, copyFileSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * The native engine (c/): a C program that reads a GGUF file directly and streams its experts from disk through Clutor's cache.
 * `stream` replays a synthetic router over the real file and measures real reads (bytes, time waiting, cache and prefetch behaviour); `chat` runs
 * the model itself. Every report says which it is.
 */

export class EngineError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EngineError";
  }
}

const EXE = process.platform === "win32" ? "clutor-engine.exe" : "clutor-engine";
const executable = (p: string): boolean => { try { accessSync(p, constants.X_OK); return true; } catch { return false; } };

/** The engine's C source: `c/` inside the installed package, else the `c/` folder of a source checkout. */
export function engineSourceDir(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  const packaged = join(here, "..", "c");
  return existsSync(join(packaged, "Makefile")) ? packaged : join(here, "..", "..", "..", "c");
}

/** True when the source came from the npm package: its folder may be read-only, so the binary is built under ~/.clutor/bin instead. */
const isPackaged = (): boolean => existsSync(join(dirname(fileURLToPath(import.meta.url)), "..", "c", "Makefile"));

export function findEngine(env: NodeJS.ProcessEnv = process.env): string | undefined {
  const candidates = [env["CLUTOR_ENGINE"], join(engineSourceDir(), "build", EXE), join(homedir(), ".clutor", "bin", EXE), ...(env["PATH"] ?? "").split(delimiter).map((d) => join(d, EXE))];
  return candidates.find((c): c is string => !!c && executable(c));
}

export function run(bin: string, args: string[], opts: { timeoutMs?: number; cwd?: string } = {}): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, { stdio: ["ignore", "pipe", "pipe"], ...(opts.cwd ? { cwd: opts.cwd } : {}) });
    let stdout = "", stderr = "";
    const timer = setTimeout(() => { child.kill("SIGKILL"); reject(new EngineError(`${bin} did not finish within ${(opts.timeoutMs ?? 600_000) / 1000} seconds`)); }, opts.timeoutMs ?? 600_000);
    child.stdout.on("data", (b: Buffer) => { stdout += b.toString(); });
    child.stderr.on("data", (b: Buffer) => { stderr += b.toString(); });
    child.on("error", (e) => { clearTimeout(timer); reject(new EngineError(`could not start ${bin}: ${e.message}`)); });
    child.on("close", (code) => { clearTimeout(timer); resolve({ code: code ?? 1, stdout, stderr }); });
  });
}

/** Compiles the engine from source with `make` (needs a C compiler). Returns the path of the binary. */
export async function buildEngine(): Promise<string> {
  const dir = engineSourceDir();
  const out = isPackaged() ? join(homedir(), ".clutor", "bin") : join(dir, "build");
  const r = await run("make", ["-C", dir, ...(isPackaged() ? [`BUILD=${out}`] : [])], { timeoutMs: 300_000 }).catch((e: unknown) => { throw new EngineError(`could not run make: ${e instanceof Error ? e.message : String(e)}`); });
  if (r.code !== 0) throw new EngineError(`the engine did not compile:\n${(r.stderr || r.stdout).split("\n").slice(-15).join("\n")}`);
  const bin = join(out, EXE);
  if (!executable(bin)) throw new EngineError(`make finished but ${bin} is missing`);
  return bin;
}

export interface EngineInfo {
  arch: string; name: string; layers: number; experts: number; expertsUsed: number; moeLayers: number;
  residentBytes: number; maxDenseBytes: number; maxExpertBytes: number; totalBytes: number; unsizedTensors: number;
  layerTable: Array<{ denseBytes: number; expertBytes: number; routerBytes: number }>;
}

export interface StreamOptions {
  tokens?: number; warmup?: number; cacheMb?: number; stage?: number;
  policy?: "lru" | "wtinylfu" | "adaptive";
  smart?: string; prefetch?: boolean; ioThreads?: number; seed?: number; locality?: number; predictCorr?: number; computeUs?: number; windowPct?: number;
  noOsCache?: boolean;
  /** A route trace recorded by `generate --trace`: the model's real router scores, replayed instead of the synthetic router. */
  trace?: string;
  /** auto: keep the dense weights in RAM when they fit in a quarter of it; ring: re-read two layers at a time every token. */
  dense?: "auto" | "resident" | "ring";
  /** kernel: a Q4_0 x Q8_0 dot-product kernel over every loaded byte (default); checksum: a cheap hash, so the run measures I/O alone. */
  compute?: "kernel" | "checksum";
  /** Scenario: measured tokens [at, at+len) route uniformly at random (scan noise); all others use one fixed tight topic. */
  shift?: { at: number; len: number };
  /** Record hits and peak RSS every N measured tokens. */
  window?: number;
}

export interface StreamReport {
  provenance: { routing: string; disk: string; compute: string };
  model: { name: string; arch: string; layers: number; experts: number; expertsUsed: number; expertBytes: number };
  config: { policy: string; cacheSlots: number; stageSlots: number; smartRoute: boolean; J: number; M: number; tau: number; denseMode: "resident" | "ring"; densePreloadBytes: number; computeMode: "kernel" | "checksum"; computeKernel: string; prefetch: boolean; ioThreads: number; osCacheBypass: boolean; tokens: number; warmup: number; seed: number; locality: number; predictCorr: number };
  totals: {
    accesses: number; hits: number; stageHits: number; misses: number; bypass: number; hitRate: number; substituted: number; droppedMass: number; maxDeltaP: number; admitRejects: number; evictions: number;
    prefetchIssued: number; prefetchLate: number; prefetchWasted: number; demandBytes: number; prefetchBytes: number; denseBytes: number;
    wallSeconds: number; ioWaitSeconds: number; computeSeconds: number; readSecondsTotal: number; overlap: number; tokensPerSecond: number; gbPerToken: number; missesPerToken: number; checksum: string;
  };
  /** ru_maxrss of the engine process, bytes: before it allocated anything, and at the end of the run. */
  rss: { startBytes: number; peakBytes: number; /** Process baseline plus every buffer and table the run allocates, all pages touched. */ predictedBytes: number };
  scenario: { shiftAt: number; shiftLen: number };
  windows: Array<{ tokEnd: number; accesses: number; hits: number; peakRssBytes: number }>;
  layers: Array<{ layer: number; accesses: number; hits: number; misses: number; substituted: number; ioWaitSeconds: number }>;
}

function streamArgs(file: string, o: StreamOptions): string[] {
  const a = ["stream", file, "--json"];
  const add = (k: string, v: string | number | undefined): void => { if (v !== undefined) a.push(k, String(v)); };
  add("--tokens", o.tokens); add("--warmup", o.warmup); add("--cache-mb", o.cacheMb); add("--stage", o.stage); add("--policy", o.policy);
  add("--smart", o.smart); if (o.prefetch !== undefined) add("--prefetch", o.prefetch ? "on" : "off");
  add("--io-threads", o.ioThreads); add("--seed", o.seed); add("--locality", o.locality); add("--predict-corr", o.predictCorr); add("--compute-us", o.computeUs); add("--window-pct", o.windowPct);
  add("--dense", o.dense); add("--compute", o.compute); add("--window", o.window);
  if (o.shift) add("--shift", `${o.shift.at},${o.shift.len}`);
  if (o.noOsCache) a.push("--no-os-cache");
  if (o.trace) a.push("--trace", o.trace);
  return a;
}

export async function engineInfo(bin: string, file: string): Promise<EngineInfo> {
  const r = await run(bin, ["info", file, "--json"], { timeoutMs: 60_000 });
  if (r.code !== 0) throw new EngineError(r.stderr.trim() || `engine exited with ${r.code}`);
  return JSON.parse(r.stdout) as EngineInfo;
}

export async function engineStream(bin: string, file: string, o: StreamOptions = {}, timeoutMs = 900_000): Promise<StreamReport> {
  const r = await run(bin, streamArgs(file, o), { timeoutMs });
  if (r.code !== 0) throw new EngineError(r.stderr.trim() || `engine exited with ${r.code}`);
  try { return JSON.parse(r.stdout) as StreamReport; } catch { throw new EngineError("the engine printed something that is not a report"); }
}

/** Runs an engine command whose output should reach the user as it is produced (`generate` writes each token as it is chosen). */
export function engineRun(bin: string, args: string[], onOut: (chunk: string) => void, timeoutMs = 3_600_000): Promise<{ code: number; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    const timer = setTimeout(() => { child.kill("SIGKILL"); reject(new EngineError(`${bin} did not finish within ${timeoutMs / 1000} seconds`)); }, timeoutMs);
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", onOut);
    child.stderr.on("data", (b: Buffer) => { stderr += b.toString(); });
    child.on("error", (e) => { clearTimeout(timer); reject(new EngineError(`could not start ${bin}: ${e.message}`)); });
    child.on("close", (code) => { clearTimeout(timer); resolve({ code: code ?? 1, stderr }); });
  });
}

/** The platform name used in release asset names, or undefined where no prebuilt binary is published. */
export function enginePlatform(): string | undefined {
  const os = process.platform === "darwin" ? "darwin" : process.platform === "linux" ? "linux" : undefined;
  const arch = process.arch === "arm64" ? "arm64" : process.arch === "x64" ? "x64" : undefined;
  return os && arch ? `${os}-${arch}` : undefined;
}

const RELEASE_BASE = (version: string): string => process.env["CLUTOR_RELEASE_BASE"] ?? `https://github.com/isagarpinninti/clutor/releases/download/v${version}`;
const MAX_ASSET = 64 * 2 ** 20;

async function download(url: string): Promise<Buffer> {
  if (!url.startsWith("https://") && !/^http:\/\/(127\.0\.0\.1|localhost)[:/]/.test(url)) throw new EngineError(`refusing to download over an insecure connection: ${url}`);
  let res: Response;
  try { res = await fetch(url); } catch (e) { throw new EngineError(`could not reach ${url}: ${e instanceof Error ? e.message : String(e)}`); }
  if (!res.ok) throw new EngineError(`${url} answered ${res.status}${res.status === 404 ? ": no prebuilt engine is published for this version and platform; use `clutor engine build` instead" : ""}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length > MAX_ASSET) throw new EngineError(`${url} is larger than ${MAX_ASSET / 2 ** 20} MiB; refusing it`);
  return buf;
}

/**
 * Installs the prebuilt engine for this platform from the GitHub release of `version`: downloads SHA256SUMS and the tarball, refuses it unless the
 * SHA-256 matches, unpacks it and puts the binary in `dest` (default ~/.clutor/bin). Returns the path of the installed binary.
 */
export async function installPrebuilt(version: string, dest = join(homedir(), ".clutor", "bin")): Promise<string> {
  const platform = enginePlatform();
  if (!platform) throw new EngineError(`no prebuilt engine is published for ${process.platform}/${process.arch}; use \`clutor engine build\` (needs a C compiler)`);
  const base = RELEASE_BASE(version), asset = `clutor-engine-${platform}.tar.gz`;
  const sums = (await download(`${base}/SHA256SUMS`)).toString("utf8");
  const line = sums.split("\n").map((l) => l.trim().split(/\s+/)).find((p) => p.length >= 2 && (p[1] as string).replace(/^\*/, "") === asset);
  if (!line) throw new EngineError(`SHA256SUMS lists no ${asset}`);
  const tarball = await download(`${base}/${asset}`);
  const got = createHash("sha256").update(tarball).digest("hex");
  if (got !== (line[0] as string).toLowerCase()) throw new EngineError(`checksum mismatch for ${asset}: expected ${line[0]}, got ${got}. Nothing was installed`);
  const work = mkdtempSync(join(tmpdir(), "clutor-engine-install-"));
  try {
    const file = join(work, asset);
    writeFileSync(file, tarball);
    const x = await run("tar", ["-xzf", file, "-C", work], { timeoutMs: 60_000 });
    if (x.code !== 0) throw new EngineError(`could not unpack ${asset}: ${x.stderr.trim()}`);
    const unpacked = join(work, `clutor-engine-${platform}`, EXE);
    if (!existsSync(unpacked)) throw new EngineError(`${asset} does not contain ${EXE}`);
    chmodSync(unpacked, 0o755);
    const probe = await run(unpacked, ["--version"], { timeoutMs: 15_000 }).catch((e: unknown) => { throw new EngineError(`the downloaded engine does not run here: ${e instanceof Error ? e.message : String(e)}`); });
    if (probe.code !== 0) throw new EngineError(`the downloaded engine does not run here (exit ${probe.code}); use \`clutor engine build\``);
    mkdirSync(dest, { recursive: true });
    const out = join(dest, EXE);
    copyFileSync(unpacked, out);
    chmodSync(out, 0o755);
    return out;
  } finally { rmSync(work, { recursive: true, force: true }); }
}

export interface ChatBench { loadSeconds: number; promptTokens: number; promptTokensPerSec: number; firstTokenSeconds: number; generationTokensPerSec: number; generated: number }

/** One fresh engine process on a real model: wait for `ready`, feed a prompt, decode greedily. These are the engine's own `chat` events, timed here; nothing is estimated. */
export function engineChatBench(bin: string, file: string, o: { prompt: string; maxTokens: number; ctx?: number; timeoutMs?: number }): Promise<ChatBench> {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const p = spawn(bin, ["chat", file, "--ctx", String(o.ctx ?? 1024)], { stdio: ["pipe", "pipe", "ignore"] });
    const timer = setTimeout(() => { p.kill(); reject(new EngineError("the engine did not finish the benchmark in time")); }, o.timeoutMs ?? 600_000);
    const r: Partial<ChatBench> = {};
    let buf = "", sentGen = false, gotFirst = false, tGen = 0;
    const send = (x: unknown): void => void p.stdin.write(`${JSON.stringify(x)}\n`);
    const finish = (err?: Error): void => { clearTimeout(timer); p.stdin.end(); p.kill(); if (err) reject(err); else resolve(r as ChatBench); };
    p.on("error", (e) => finish(e));
    p.on("close", () => { if (r.generationTokensPerSec === undefined) { clearTimeout(timer); reject(new EngineError("the engine exited before the benchmark finished")); } });
    p.stdout.on("data", (d: Buffer) => {
      buf += d.toString();
      let i: number;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        let ev: { ev?: string; tokens?: number; seconds?: number; tokensPerSecond?: number };
        try { ev = JSON.parse(line); } catch { continue; }
        if (ev.ev === "ready") { r.loadSeconds = (Date.now() - t0) / 1000; send({ op: "append", text: o.prompt, raw: true }); }
        else if (ev.ev === "appended" && !sentGen) {
          r.promptTokens = ev.tokens ?? 0; r.promptTokensPerSec = ev.seconds ? (ev.tokens ?? 0) / ev.seconds : 0; r.firstTokenSeconds = ev.seconds ?? 0;
          sentGen = true; tGen = Date.now(); send({ op: "generate", max: o.maxTokens, temp: 0 });
        } else if (ev.ev === "t" && !gotFirst && r.firstTokenSeconds !== undefined) { gotFirst = true; r.firstTokenSeconds += (Date.now() - tGen) / 1000; }
        else if (ev.ev === "done") { r.generationTokensPerSec = ev.tokensPerSecond ?? 0; r.generated = ev.tokens ?? 0; finish(); }
      }
    });
  });
}
