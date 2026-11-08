import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { homedir, totalmem } from "node:os";
import { dirname, join, relative, resolve, sep } from "node:path";
import { BackendManager, buildLlamaArgs, findLlamaServer, probeLlamaServer } from "../backend";
import { ConfigError, loadConfig, type ClutorConfig } from "../config";
import { flagInt, flagString, parseArgs } from "../args";
import type { Io, SessionState } from "../cli";
import { Region, THEMES, fit, makeStyle, stripAnsi, wrap, type ThemeName } from "./ansi";
import { runAgentTask, type AgentHost } from "./agent";
import { welcomeLines } from "./welcome";
import { discoverModels, displayName, gb, modelFolders, type FoundModel } from "./discover";
import { Editor, type Cmd } from "./editor";
import { KeyParser } from "./keys";
import { MarkdownStream } from "./markdown";
import { formatElapsed, spinnerLine, verbFor } from "./spinner";
import { streamChat, type ChatMessage, type StreamEvent } from "./stream";
import { NativeEngine } from "./native";
import { engineInfo, findEngine, run as runBinary } from "../engine";

/**
 * The interactive session: a prompt box at the bottom, the model's reply streamed above it as Markdown, an animated status line while
 * it works, and slash commands. Plain text is a chat message; /agent hands a task to the sandboxed coding agent.
 */

export type Run = (argv: string[], session: SessionState) => Promise<number>;

interface Target {
  name: string;
  baseUrl: string;
  model: string;
  apiKey?: string;
  path?: string;
  bytes?: number;
  backend?: BackendManager;
  /** Clutor's own engine, when it runs this model; otherwise the model is served by llama-server (or a remote server). */
  native?: NativeEngine;
  /** Which engine answers, in words, for the welcome panel and /engine. */
  engine: string;
  /** Why the native engine was not used, when it was not. */
  engineNote?: string;
  loadMs?: number;
}

/** The architectures Clutor's engine runs (see docs/MODELS.md). */
const NATIVE_ARCHS = new Set(["olmoe", "qwen3moe", "qwen3vlmoe", "granitemoe", "qwen2", "llama", "qwen35"]);

export interface TuiOptions {
  io: Io;
  run: Run;
  version: string;
  argv: string[];
  stdin?: NodeJS.ReadStream;
  stdout?: NodeJS.WriteStream;
  color?: boolean;
  /** Where chat transcripts are kept for /resume. */
  home?: string;
}

const COMMANDS: Cmd[] = [
  { name: "help", help: "what you can do here" },
  { name: "clear", help: "start a new conversation" },
  { name: "compact", help: "summarise the conversation to free up context" },
  { name: "model", help: "choose another model file" },
  { name: "engine", help: "which engine runs this model, and why" },
  { name: "think", args: "[on|off|show]", help: "let thinking models think first (slower, often better)" },
  { name: "system", args: "<text>", help: "set the instructions the model always follows" },
  { name: "temp", args: "<0-2>", help: "how adventurous the replies are" },
  { name: "agent", args: "<task>", help: "hand a task to the coding agent (changes wait for your approval)" },
  { name: "undo", help: "restore the files the last agent task changed" },
  { name: "verify", args: "[command|off]", help: "a check the agent must pass before changes are offered" },
  { name: "yes", help: "toggle: let the agent run project commands without asking" },
  { name: "theme", args: "[dark|light|ocean|mono]", help: "change the colours" },
  { name: "resume", help: "continue an earlier conversation" },
  { name: "copy", help: "copy the last reply to the clipboard" },
  { name: "save", args: "[file]", help: "write this conversation to a Markdown file" },
  { name: "cost", help: "tokens and speed this session (measured)" },
  { name: "status", help: "machine, model backend, memory plan, receipts" },
  { name: "doctor", help: "check this machine and the configuration" },
  { name: "models", help: "model catalog and the .gguf files you have" },
  { name: "plan", args: "<model>", help: "will it fit, and how fast could it run" },
  { name: "bench", help: "measure real disk reads" },
  { name: "pull", args: "[id]", help: "download a small test model" },
  { name: "init", help: "create clutor.config.json here" },
  { name: "exit", help: "leave (also ctrl-d, or ctrl-c twice)" },
];

/** Commands that are the ordinary clutor commands, run exactly as they are outside the session. */
const FORWARD = new Set(["status", "doctor", "models", "plan", "bench", "pull", "init", "simulate", "inspect", "ledger", "decide", "backend", "stream"]);

type Mode = "chat" | "agent" | "auto";
const MODES: Mode[] = ["chat", "agent", "auto"];

const SYSTEM_DEFAULT = "You are a helpful assistant running locally in a terminal through Clutor. Answer directly and concisely. Use Markdown only where it helps (short lists, code blocks).";

export function freePort(): Promise<number> {
  return new Promise((ok, fail) => {
    const s = createServer();
    s.once("error", fail);
    s.listen(0, "127.0.0.1", () => { const p = (s.address() as { port: number }).port; s.close(() => ok(p)); });
  });
}

export function expandFiles(text: string, cwd: string): { text: string; attached: string[] } {
  const attached: string[] = [];
  let extra = "";
  for (const m of text.matchAll(/(^|\s)@(\S+)/g)) {
    const p = resolve(cwd, m[2] as string);
    const rel = relative(cwd, p);
    if (rel.startsWith("..") || rel.split(sep).includes("..") || !existsSync(p)) continue;
    try {
      const s = statSync(p);
      if (!s.isFile() || s.size > 100_000) continue;
      const body = readFileSync(p, "utf8");
      if (body.includes("\0")) continue;
      extra += `\n\n<file path="${rel}">\n${body}\n</file>`;
      attached.push(`${rel} (${body.split("\n").length} lines)`);
    } catch { /* unreadable: leave the @word as typed */ }
  }
  return { text: text + extra, attached };
}

/** What llama-server is doing, from the last lines it printed. */
export function loadPhase(log: string[]): string | undefined {
  for (let i = log.length - 1; i >= 0; i--) {
    const l = stripAnsi(log[i] as string).toLowerCase();
    if (l.includes("listening")) return "starting the server";
    if (l.includes("warm")) return "warming up";
    if (l.includes("llama_context") || l.includes("kv buffer") || l.includes("compute buffer")) return "creating the context";
    if (l.includes("load_tensors") || l.includes("llama_model_load") || l.includes("mapped")) return "mapping the weights";
    if (l.includes("print_info") || l.includes("load:")) return "reading the model's header";
  }
  return undefined;
}

const trunc = (s: string, n: number): string => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

const compactNum = (n: number): string => (n >= 10_000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

export async function runTui(o: TuiOptions): Promise<number> {
  const inp = o.stdin ?? process.stdin;
  const out = o.stdout ?? process.stdout;
  const colour = o.color ?? true;
  const cwd = o.io.cwd;
  const args = parseArgs(o.argv);
  const stateDir = join(o.home ?? homedir(), ".clutor");
  const settingsFile = join(stateDir, "settings.json");
  type Settings = { theme?: ThemeName; mode?: Mode; lastModel?: string; modelDirs?: string[] };
  const settings = ((): Settings => { try { return JSON.parse(readFileSync(settingsFile, "utf8")) as Settings; } catch { return {}; } })();
  const saveSettings = (): void => { try { mkdirSync(stateDir, { recursive: true, mode: 0o700 }); writeFileSync(settingsFile, JSON.stringify(settings), { mode: 0o600 }); } catch { /* a setting is a convenience */ } };
  const st = makeStyle(colour, THEMES.includes(settings.theme as ThemeName) ? (settings.theme as ThemeName) : "dark");
  let mode: Mode = flagString(args, "mode") === "agent" ? "agent" : "chat";
  const agentState = { allowCommands: false };
  const cols = (): number => (out.columns && out.columns > 0 ? out.columns : 80);

  let config: ClutorConfig;
  try { config = loadConfig(cwd).config; } catch (e) { o.io.err(e instanceof ConfigError ? e.message : String(e)); return 1; }

  // Terminal state: restored on every way out.
  let raw = false;
  const setRaw = (on: boolean): void => {
    if (inp.isTTY && on !== raw) { inp.setRawMode(on); raw = on; }
    if (on) { inp.resume(); inp.setEncoding("utf8"); out.write("\x1b[?2004h"); } else out.write("\x1b[?2004l");
  };
  const restore = (): void => { try { setRaw(false); out.write("\x1b[?25h\x1b[0m"); } catch { /* terminal already gone */ } };
  // Whatever way the process ends (window closed, kill, crash), the terminal is put back and the model server does not outlive it.
  let starting: BackendManager | undefined;
  const onExit = (): void => { restore(); starting?.killNow(); target?.backend?.killNow(); target?.native?.killNow(); };
  process.once("exit", onExit);
  const onSignal = (code: number) => (): void => { onExit(); process.exit(code); };
  const sighup = onSignal(129), sigterm = onSignal(143);
  process.once("SIGHUP", sighup);
  process.once("SIGTERM", sigterm);

  const history = (() => { try { return readFileSync(join(stateDir, "history"), "utf8").split("\n").filter(Boolean).slice(-500); } catch { return []; } })();
  const editor = new Editor(history, COMMANDS);
  const session: SessionState = { auto: false, notes: [], usage: { promptTokens: 0, completionTokens: 0 } };

  // Conversation state.
  const chat = { messages: [] as ChatMessage[], system: SYSTEM_DEFAULT, thinking: args.flags.has("think"), showThinking: false, temperature: undefined as number | undefined, ctx: flagInt(args, "ctx", 512, 2_000_000) ?? config.memory.contextTokens };
  const stats = { turns: 0, prompt: 0, completion: 0, lastTps: 0, lastPrompt: 0, ctxUsed: 0, replyTime: 0, replyTokens: 0 };
  let lastReply = "";
  let target: Target | undefined;
  let sessionFile: string | undefined;

  const println = (s = ""): void => void out.write(`${s}\n`);
  const mark = (glyph: string, paint: (s: string) => string, s: string): void => {
    for (const [i, r] of wrap(paint(s), cols() - 4, 0).entries()) println(i === 0 ? `  ${paint(glyph)} ${r}` : `    ${r}`);
  };
  const note = (s: string): void => mark("⎿", st.dim, s);
  const fail = (s: string): void => mark("⎿", st.bad, s);

  // ---- welcome ----
  const recentTitles = (): string[] => {
    try {
      return readdirSync(join(stateDir, "sessions")).filter((f) => f.endsWith(".json")).sort().reverse().slice(0, 3).flatMap((f) => {
        const j = JSON.parse(readFileSync(join(stateDir, "sessions", f), "utf8")) as { messages?: ChatMessage[] };
        const first = j.messages?.find((m) => m.role === "user")?.content.replace(/\s+/g, " ").trim();
        return first ? [first] : [];
      });
    } catch { return []; }
  };
  const welcome = (): void => {
    println();
    for (const l of welcomeLines(st, cols(), { version: o.version, cwd, model: target?.name ?? "", detail: target ? `${target.engine} · ${target.bytes ? `${gb(target.bytes)} · ` : ""}${compactNum(chat.ctx)} ctx${target.loadMs ? ` · ${(target.loadMs / 1000).toFixed(1)} s` : ""}` : "", recent: recentTitles() })) println(l);
    println();
  };

  // ---- input ----
  let notice: string | undefined;
  const MODE_TEXT: Record<Mode, string> = {
    chat: "chat · shift+tab: coding agent",
    agent: "agent · edits need approval · shift+tab: next mode",
    auto: "auto-apply · edits are written for you · shift+tab: next mode",
  };
  const footer = (): { left: string; right: string } => ({
    left: `${MODE_TEXT[mode]}`,
    right: `${trunc(target?.name ?? "no model", 34)} · ${stats.ctxUsed > 0 ? `${compactNum(stats.ctxUsed)}/${compactNum(chat.ctx)}` : compactNum(chat.ctx)} ctx`,
  });
  const look = (): { border?: (s: string) => string; glyph?: string } => (mode === "chat" ? {} : { border: mode === "auto" ? st.warn : st.accent, glyph: mode === "auto" ? "⏵⏵" : "⏵" });

  const readInput = (): Promise<string | null> => new Promise((done) => {
    let region = new Region(out);
    const parser = new KeyParser();
    let timer: NodeJS.Timeout | undefined;
    const draw = (): void => { const v = editor.view(st, cols(), footer(), notice, look()); region.draw(v.lines, v.cursor); };
    const finish = (value: string | null): void => {
      inp.off("data", onData);
      out.off("resize", draw);
      if (timer) clearTimeout(timer);
      region.clear();
      done(value);
    };
    const onData = (d: string | Buffer): void => {
      notice = undefined;
      for (const k of parser.feed(String(d))) {
        const a = editor.handle(k);
        if (!a) continue;
        if (a.type === "submit") { finish(a.text); return; }
        if (a.type === "exit") { finish(null); return; }
        if (a.type === "notice") { notice = a.text; timer = setTimeout(() => { notice = undefined; draw(); }, 2000); }
        if (a.type === "cycle-mode") { mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length] as Mode; settings.mode = mode; saveSettings(); }
        if (a.type === "clear-screen") { out.write("\x1b[2J\x1b[H"); region = new Region(out); }
      }
      draw();
    };
    setRaw(true);
    inp.on("data", onData);
    out.on("resize", draw);
    draw();
  });

  /** A list to choose from with the arrow keys. Returns the index, or undefined on escape. */
  const pick = (title: string, items: Array<{ label: string; hint?: string }>, start = 0): Promise<number | undefined> => new Promise((done) => {
    const region = new Region(out);
    const parser = new KeyParser();
    let at = Math.min(start, items.length - 1);
    const draw = (): void => {
      const lines = [st.bold(title), ""];
      items.forEach((it, i) => lines.push(i === at ? `${st.accent("❯")} ${st.accent(it.label)}${it.hint ? ` ${st.dim(it.hint)}` : ""}` : `  ${it.label}${it.hint ? ` ${st.dim(it.hint)}` : ""}`));
      lines.push("", st.dim("↑↓ to move · enter to choose · esc to cancel"));
      region.draw(lines.map((l) => fit(l, cols() - 1)));
    };
    const onData = (d: string | Buffer): void => {
      for (const k of parser.feed(String(d))) {
        if (k.name === "up") at = (at + items.length - 1) % items.length;
        else if (k.name === "down") at = (at + 1) % items.length;
        else if (k.name === "enter") { end(); done(at); return; }
        else if (k.name === "escape" || k.name === "ctrl-c") { end(); done(undefined); return; }
      }
      draw();
    };
    const end = (): void => { inp.off("data", onData); region.clear(); };
    setRaw(true);
    inp.on("data", onData);
    draw();
  });

  // ---- a status line that animates while something runs ----
  const busy = async <T>(label: string, work: (setDetail: (s: string) => void) => Promise<T>): Promise<T> => {
    const region = new Region(out);
    const t0 = Date.now();
    let detail = "";
    const tick = (): void => region.draw([fit(spinnerLine(st, Date.now() - t0, label, [formatElapsed(Date.now() - t0), ...(detail ? [detail] : [])]), cols() - 1)]);
    const timer = setInterval(tick, 80);
    tick();
    try { return await work((s) => { detail = s; }); } finally { clearInterval(timer); region.clear(); }
  };

  // ---- model ----
  const stopTarget = async (): Promise<void> => { try { await target?.backend?.stop(); await target?.native?.close(); } catch { /* already gone */ } target = undefined; };

  /** Whether Clutor's engine can run this file, and if not, why not (said to the user, not hidden). */
  const nativeCheck = async (m: FoundModel): Promise<{ bin: string; experts: number } | { reason: string }> => {
    const bin = findEngine();
    if (!bin) return { reason: "Clutor's engine is not built here (run `clutor engine install` or `clutor engine build`)" };
    let info;
    try { info = await engineInfo(bin, m.path); } catch (e) { return { reason: `Clutor's engine could not read the file: ${e instanceof Error ? e.message : String(e)}` }; }
    if (!NATIVE_ARCHS.has(info.arch)) return { reason: `Clutor's engine does not run the "${info.arch}" architecture yet` };
    const usage = await runBinary(bin, [], { timeoutMs: 15_000 }).catch(() => ({ stdout: "" }));
    if (!/\n\s+chat <model\.gguf>/.test(usage.stdout)) return { reason: "the installed engine is older than chat support (run `clutor engine install` or `clutor engine build`)" };
    return { bin, experts: info.experts };
  };

  const startNative = async (m: FoundModel, bin: string, experts: number): Promise<Target> => {
    const ctx = flagInt(args, "ctx", 64, 2_000_000) ?? Math.min(config.memory.contextTokens, 4096);
    // A mixture-of-experts model bigger than half the memory is streamed from disk through a cache (the point of this engine).
    const stream = experts > 0 && m.bytes > totalmem() * 0.5;
    // A model that fits is pinned in memory (what a GPU runtime does by wiring its buffers): otherwise the operating system evicts its pages whenever other
    // programs want memory and every token re-reads the file from disk. Measured on a 16 GiB machine with browsers open: 7.35 s/token unpinned, 0.16 s/token pinned.
    const lock = !stream && m.bytes <= totalmem() * 0.6 && !args.flags.has("no-lock");
    const extra = stream ? ["--cache-mb", String(Math.floor((totalmem() / 1048576) * 0.3))] : lock ? ["--lock"] : [];
    const t0 = Date.now();
    const eng = await busy(`Loading ${m.name}`, async () => NativeEngine.start({ bin, model: m.path, ctx, extraArgs: extra }));
    chat.ctx = eng.ready.nCtx;
    return { name: m.name, baseUrl: "", model: "native", path: m.path, bytes: m.bytes, native: eng, engine: `Clutor engine${stream ? " · experts streamed from disk" : eng.ready.locked ? " · pinned in memory" : ""}`, loadMs: Date.now() - t0 };
  };

  const startLocal = async (m: FoundModel): Promise<Target> => {
    const want = flagString(args, "backend") ?? "auto";
    let engineNote: string | undefined;
    if (want !== "llama") {
      const c = await nativeCheck(m);
      if ("bin" in c) {
        try { return await startNative(m, c.bin, c.experts); } catch (e) {
          if (want === "native") throw e;
          engineNote = `Clutor's engine failed to start (${e instanceof Error ? e.message : String(e)})`;
        }
      } else {
        if (want === "native") throw new ConfigError(`--backend native: ${c.reason}`);
        engineNote = c.reason;
      }
    }
    const ls = config.backend.llamaServer;
    const binary = ls.binary ?? findLlamaServer();
    if (!binary) throw new ConfigError("llama-server was not found. Install llama.cpp (`brew install llama.cpp`) or set backend.llamaServer.binary in clutor.config.json.");
    const caps = await probeLlamaServer(binary);
    const port = await freePort();
    const { args: a } = buildLlamaArgs({ modelPath: m.path, port, contextTokens: chat.ctx, kvBits: config.memory.kvBits, ...(ls.threads ? { threads: ls.threads } : {}), caps, extraArgs: ["--parallel", "1", ...ls.extraArgs] });
    const backend = new BackendManager(port);
    starting = backend;
    const t0 = Date.now();
    await busy(`Loading ${m.name}`, async (detail) => {
      const poll = setInterval(() => { const p = loadPhase(backend.recentLog); if (p) detail(p); }, 300);
      try { await backend.start(binary, a); } finally { clearInterval(poll); }
    });
    return { name: m.name, baseUrl: backend.baseUrl, model: "local", path: m.path, bytes: m.bytes, backend, engine: "llama-server", ...(engineNote ? { engineNote } : {}), loadMs: Date.now() - t0 };
  };

  const chooseModel = async (forcePick: boolean): Promise<Target | undefined> => {
    const b = config.backend;
    if (b.kind === "openai" && b.baseUrl && !forcePick) {
      return { name: b.model ?? "remote model", baseUrl: b.baseUrl, model: b.model ?? "default", engine: "remote server", ...(b.apiKeyEnv && process.env[b.apiKeyEnv] ? { apiKey: process.env[b.apiKeyEnv] as string } : {}) };
    }
    const explicit = flagString(args, "model") ?? (!forcePick ? b.llamaServer.model : undefined);
    if (explicit) {
      const p = resolve(cwd, explicit);
      if (!existsSync(p)) { fail(`${p} does not exist`); return undefined; }
      return startLocal({ path: p, name: displayName(p), bytes: statSync(p).size });
    }
    const found = discoverModels(cwd, config.models, settings.modelDirs ?? []);
    if (found.length === 0) {
      fail(`no .gguf model found. Looked in: ${modelFolders(cwd, config.models, settings.modelDirs ?? []).join(", ")}. Put a model in one of them, pass --model <file.gguf>, or run /pull to download a small one.`);
      return undefined;
    }
    const last = found.findIndex((f) => f.path === (target?.path ?? settings.lastModel));
    let m = found[last >= 0 ? last : 0] as FoundModel;
    // With only one model, or one chosen last time (and no request to choose), there is nothing to ask; /model changes it.
    if (forcePick || (found.length > 1 && last < 0)) {
      const i = await pick("Choose a model", found.map((f) => ({ label: f.name, hint: gb(f.bytes) })), Math.max(0, last));
      if (i === undefined) return undefined;
      m = found[i] as FoundModel;
    }
    settings.lastModel = m.path;
    settings.modelDirs = [...new Set([dirname(m.path), ...(settings.modelDirs ?? [])])].slice(0, 5);
    saveSettings();
    return startLocal(m);
  };

  const targetStream = (t: Target, messages: ChatMessage[], o: { thinking: boolean; signal?: AbortSignal }): AsyncGenerator<StreamEvent> =>
    t.native
      ? t.native.stream(messages, { thinking: o.thinking, ...(chat.temperature !== undefined ? { temp: chat.temperature } : {}), ...(o.signal ? { signal: o.signal } : {}) })
      : streamChat({ baseUrl: t.baseUrl, model: t.model, messages, thinking: o.thinking, ...(o.signal ? { signal: o.signal } : {}), ...(t.apiKey ? { apiKey: t.apiKey } : {}), ...(chat.temperature !== undefined ? { temperature: chat.temperature } : {}) });

  // ---- one chat turn ----
  const turn = async (userText: string): Promise<void> => {
    if (!target) { fail("no model is loaded. Use /model to choose one."); return; }
    const { text, attached } = expandFiles(userText, cwd);
    for (const a of attached) note(`attached ${a}`);
    chat.messages.push({ role: "user", content: text });
    println(`${st.dim("❯")} ${st.dim(userText.split("\n").join("\n  "))}`);
    println();

    const md = new MarkdownStream(st, () => cols() - 2);
    const think = new MarkdownStream(st, () => cols() - 2);
    const region = new Region(out);
    const ac = new AbortController();
    const t0 = Date.now();
    let chunks = 0, reasoningChunks = 0, firstToken = 0, progress = "", reply = "", printed = 0, interrupted = false, live: string[] = [], thinkLive: string[] = [];
    let phase: "prompt" | "think" | "write" = "prompt";
    const dimRows = (rows: string[]): string[] => rows.map((r) => st.dim(st.italic(stripAnsi(r))));
    const paint = (): void => {
      const t = Date.now() - t0;
      const label = phase === "prompt" ? "Reading" : phase === "think" ? "Thinking" : verbFor(Math.floor(t0 / 1000));
      const details = [formatElapsed(t), ...(phase === "prompt" && progress ? [progress] : []), ...(chunks + reasoningChunks > 0 ? [`↓ ${chunks + reasoningChunks} tokens`] : []), "esc to stop"];
      const rows = [...thinkLive.map((r) => `  ${r}`), ...live.map((r, i) => (printed === 0 && i === 0 ? `${st.accent("●")} ${r}` : `  ${r}`))];
      region.draw([...rows.map((r) => fit(r, cols() - 1)), "", fit(spinnerLine(st, t, label, details), cols() - 1)]);
    };
    const commit = (rows: string[], dim = false): void => {
      if (rows.length === 0) return;
      region.clear();
      const body = (dim ? dimRows(rows) : rows).map((r, i) => (!dim && printed === 0 && i === 0 ? `${st.accent("●")} ${r}` : `  ${r}`));
      if (!dim) printed += rows.length;
      out.write(`${body.join("\n")}\n`);
    };
    const onKey = (d: string | Buffer): void => { const s = String(d); if (s === "\x1b" || s === "\x03") { interrupted = true; ac.abort(); } };
    setRaw(true);
    inp.on("data", onKey);
    const timer = setInterval(paint, 80);
    paint();
    let done: Extract<StreamEvent, { type: "done" }> | undefined;
    try {
      for await (const ev of targetStream(target, [{ role: "system" as const, content: chat.system }, ...chat.messages], { thinking: chat.thinking, signal: ac.signal })) {
        if (ev.type === "progress") { progress = `${Math.round((ev.processed / ev.total) * 100)}% of ${ev.total} prompt tokens`; continue; }
        if (ev.type === "done") { done = ev; continue; }
        if (!firstToken) firstToken = Date.now();
        if (ev.type === "reasoning") {
          phase = "think"; reasoningChunks++;
          if (chat.showThinking) { const r = think.push(ev.text); commit(r.done, true); thinkLive = dimRows(r.live); }
          continue;
        }
        if (phase !== "write") {
          phase = "write";
          if (chat.showThinking) { commit(think.end().done, true); thinkLive = []; }
        }
        chunks++; reply += ev.text;
        const r = md.push(ev.text);
        commit(r.done);
        live = r.live;
      }
    } catch (e) {
      if (!interrupted) {
        clearInterval(timer); inp.off("data", onKey); region.clear();
        chat.messages.pop();
        fail(e instanceof Error ? e.message : String(e));
        if (target.backend?.state === "failed") fail(`the model server stopped: ${target.backend.error ?? "unknown reason"}`);
        return;
      }
    }
    clearInterval(timer);
    inp.off("data", onKey);
    commit(md.end().done);
    region.clear();
    if (printed === 0 && reply === "" && !interrupted) note(done?.finish === "length" ? "the model ran out of room before answering; try /think off or /clear" : "the model returned an empty reply");
    chat.messages.push({ role: "assistant", content: reply });
    lastReply = reply;
    const secs = ((Date.now() - (firstToken || t0)) / 1000) || 0.001;
    const gen = done?.completionTokens ?? chunks + reasoningChunks;
    const tps = done?.tokensPerSecond ?? (firstToken ? gen / secs : 0);
    const bits = [`${gen} tokens`, ...(tps ? [`${tps.toFixed(1)} tok/s`] : []), ...(done?.promptTokens ? [`prompt ${done.promptTokens} tok`] : []), ...(done?.cache ? [`cache ${Math.round(done.cache.hitRate * 100)}% hit, ${done.cache.gbRead.toFixed(2)} GB read`] : []), formatElapsed(Date.now() - t0)];
    println();
    note(`${interrupted ? "stopped · " : ""}${bits.join(" · ")}`);
    stats.turns++;
    stats.prompt += done?.promptTokens ?? 0; stats.completion += gen; session.usage.promptTokens += done?.promptTokens ?? 0; session.usage.completionTokens += gen;
    if (tps) { stats.lastTps = tps; stats.replyTokens += gen; stats.replyTime += gen / tps; }
    stats.ctxUsed = (done?.promptTokens ?? 0) + gen;
    if (stats.ctxUsed > chat.ctx * 0.8) println(`  ${st.warn("⎿")} ${st.warn(`context is ${Math.round((stats.ctxUsed / chat.ctx) * 100)}% full: /compact summarises the conversation, /clear starts over`)}`);
    println();
    saveSession();
  };

  // ---- the coding agent ----
  const host: AgentHost = {
    st, cols, println: (x) => println(x), region: () => new Region(out),
    choose: (title, items) => pick(title, items),
    listen: (f) => { setRaw(true); const on = (d: string | Buffer): void => f(String(d)); inp.on("data", on); return () => void inp.off("data", on); },
  };
  const agentTask = async (task: string): Promise<void> => {
    if (!target) { fail("no model is loaded. Use /model to choose one."); return; }
    println(`${st.dim("❯")} ${st.dim(task.split("\n").join("\n  "))}`);
    println();
    try {
      await runAgentTask(host, { task, dir: cwd, config, endpoint: { baseUrl: target.baseUrl, model: target.model, ...(target.apiKey ? { apiKey: target.apiKey } : {}), local: target.backend !== undefined }, ...(target.native ? { llm: target.native.asLlm(target.name) } : {}), session, autoApply: mode === "auto", contextTokens: chat.ctx }, agentState);
    } catch (e) { fail(e instanceof Error ? e.message : String(e)); }
    println();
  };

  // ---- transcripts ----
  const sessionDir = join(stateDir, "sessions");
  const saveSession = (): void => {
    try {
      mkdirSync(sessionDir, { recursive: true, mode: 0o700 });
      sessionFile ??= join(sessionDir, `${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
      writeFileSync(sessionFile, JSON.stringify({ model: target?.name, system: chat.system, messages: chat.messages }), { mode: 0o600 });
    } catch { /* a transcript is a convenience */ }
  };
  const markdownTranscript = (): string => `# Clutor conversation (${target?.name ?? "no model"})\n\n${chat.messages.map((m) => `**${m.role === "user" ? "You" : "Model"}**\n\n${m.content}\n`).join("\n")}`;

  const compact = async (): Promise<void> => {
    if (!target || chat.messages.length < 2) { note("nothing to compact yet"); return; }
    const asked: ChatMessage[] = [{ role: "system", content: "Summarise the conversation so far in under 250 words, keeping every fact, name, number and decision the user may refer to later. Write the summary only." }, { role: "user", content: chat.messages.map((m) => `${m.role}: ${m.content}`).join("\n\n") }];
    let summary = "";
    try {
      await busy("Compacting", async () => { if (target) for await (const ev of targetStream(target, asked, { thinking: false })) if (ev.type === "content") summary += ev.text; });
    } catch (e) { fail(e instanceof Error ? e.message : String(e)); return; }
    if (!summary.trim()) { fail("the model returned no summary; the conversation is unchanged"); return; }
    chat.messages = [{ role: "user", content: `Summary of our conversation so far:\n${summary.trim()}` }, { role: "assistant", content: "Understood. I'll continue from that summary." }];
    stats.ctxUsed = 0;
    note(`conversation compacted to a ${summary.trim().split(/\s+/).length}-word summary`);
  };

  const clipboard = (text: string): boolean => {
    const cmd = process.platform === "darwin" ? ["pbcopy"] : process.platform === "win32" ? ["clip"] : ["xclip", "-selection", "clipboard"];
    try { const c = spawn(cmd[0] as string, cmd.slice(1), { stdio: ["pipe", "ignore", "ignore"] }); c.on("error", () => undefined); c.stdin?.end(text); return true; } catch { return false; }
  };

  const help = (): void => {
    println(st.bold("  Chat"));
    println(`  ${st.dim("Type a message and press enter. Text starting with / is a command; @path attaches a file.")}`);
    println();
    const w = Math.max(...COMMANDS.map((c) => c.name.length + (c.args ? c.args.length + 1 : 0))) + 3;
    for (const c of COMMANDS) println(`  ${st.accent(`/${c.name}${c.args ? ` ${c.args}` : ""}`.padEnd(w))}${st.dim(c.help)}`);
    println();
  };

  const slash = async (line: string): Promise<"exit" | undefined> => {
    const [word = "", ...restWords] = line.slice(1).split(/\s+/);
    const name = word.toLowerCase();
    const rest = line.slice(1 + word.length).trim();
    switch (name) {
      case "exit": case "quit": case "q": return "exit";
      case "help": case "?": help(); return;
      case "clear": case "new": chat.messages = []; stats.ctxUsed = 0; sessionFile = undefined; lastReply = ""; out.write("\x1b[2J\x1b[H"); welcome(); note("new conversation"); println(); return;
      case "compact": await compact(); return;
      case "model": {
        const prev = target;
        if (prev) { println(`  ${st.dim("unloading")} ${prev.name}`); await stopTarget(); }
        const t = await chooseModel(true).catch((e: unknown) => { fail(e instanceof Error ? e.message : String(e)); return undefined; });
        if (t) { target = t; chat.messages = []; stats.ctxUsed = 0; sessionFile = undefined; note(`${t.name} ready${t.loadMs ? ` in ${(t.loadMs / 1000).toFixed(1)} s` : ""}; started a new conversation`); }
        else note("no model loaded: use /model to choose one");
        return;
      }
      case "engine": {
        if (!target) { note("no model loaded"); return; }
        note(`engine: ${target.engine}`);
        if (target.native) {
          const r = target.native.ready;
          note(`${r.arch} · ${r.nLayer} layers · ${r.threads} threads · context ${r.nCtx} · prompt format ${target.native.template.family}${target.native.template.thinkTags ? " (with <think> tags)" : ""} · ${r.streaming ? "experts streamed from disk" : r.locked ? "weights pinned in memory" : "weights memory-mapped (not pinned: the OS may evict them under memory pressure)"}`);
          const a = target.native.lastAppend;
          note(`last turn sent ${a.tokens} new token${a.tokens === 1 ? "" : "s"} to the engine${a.reset ? " (started over: the conversation did not extend the cache)" : " (the rest was already in its cache)"}`);
        } else if (target.engineNote) note(`not the native engine: ${target.engineNote}`);
        return;
      }
      case "think":
        if (restWords[0] === "show") { chat.showThinking = !chat.showThinking; note(`thinking text is ${chat.showThinking ? "shown (dim)" : "hidden"}`); }
        else { chat.thinking = restWords[0] ? restWords[0] === "on" : !chat.thinking; note(`thinking ${chat.thinking ? "on: thinking models reason before answering (slower, often better)" : "off: answers come straight away"}`); }
        return;
      case "system":
        if (!rest) { note(`system: ${chat.system}`); return; }
        chat.system = rest === "reset" ? SYSTEM_DEFAULT : rest; note(rest === "reset" ? "system prompt reset" : "system prompt set"); return;
      case "temp": {
        const v = Number(restWords[0]);
        if (!restWords[0]) { note(`temperature: ${chat.temperature ?? "the model's default"}`); return; }
        if (!Number.isFinite(v) || v < 0 || v > 2) { fail("usage: /temp <0 to 2>"); return; }
        chat.temperature = v; note(`temperature ${v}`); return;
      }
      case "copy": if (!lastReply) note("nothing to copy yet"); else note(clipboard(lastReply) ? `copied ${lastReply.length} characters` : "no clipboard tool found"); return;
      case "save": {
        const f = resolve(cwd, rest || `clutor-chat-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.md`);
        try { writeFileSync(f, markdownTranscript()); note(`saved ${relative(cwd, f) || f}`); } catch (e) { fail(e instanceof Error ? e.message : String(e)); }
        return;
      }
      case "cost": {
        const avg = stats.replyTime > 0 ? (stats.replyTokens / stats.replyTime).toFixed(1) : "n/a";
        note(`${stats.prompt} prompt + ${stats.completion} generated tokens over ${stats.turns} repl${stats.turns === 1 ? "y" : "ies"}; average ${avg} tok/s (measured by the model server; local models cost no money)`);
        return;
      }
      case "resume": {
        let files: string[] = [];
        try { files = readdirSync(sessionDir).filter((f) => f.endsWith(".json")).sort().reverse().slice(0, 10); } catch { /* none yet */ }
        if (files.length === 0) { note("no earlier conversations"); return; }
        const items = files.map((f) => {
          try { const j = JSON.parse(readFileSync(join(sessionDir, f), "utf8")) as { model?: string; messages: ChatMessage[] }; const first = j.messages.find((m) => m.role === "user")?.content ?? ""; return { label: first.replace(/\s+/g, " ").slice(0, 60) || f, hint: `${j.messages.length} messages${j.model ? ` · ${j.model}` : ""}` }; } catch { return { label: f, hint: "unreadable" }; }
        });
        const i = await pick("Resume a conversation", items);
        if (i === undefined) return;
        try {
          const j = JSON.parse(readFileSync(join(sessionDir, files[i] as string), "utf8")) as { system?: string; messages: ChatMessage[] };
          chat.messages = j.messages; if (j.system) chat.system = j.system; sessionFile = join(sessionDir, files[i] as string); stats.ctxUsed = 0;
          note(`resumed ${j.messages.length} messages`);
          const last = [...j.messages].reverse().find((m) => m.role === "assistant"); if (last) { lastReply = last.content; println(); println(`  ${st.dim("last reply:")} ${last.content.slice(0, 300).replace(/\s+/g, " ")}${last.content.length > 300 ? "…" : ""}`); println(); }
        } catch { fail("that conversation could not be read"); }
        return;
      }
      case "agent": {
        if (!rest) { fail("usage: /agent <task>   (or press shift+tab to make agent mode the default)"); return; }
        await agentTask(rest); return;
      }
      case "undo": {
        if (!session.lastCheckpoint) { note("nothing to undo in this session"); return; }
        const { id, dir } = session.lastCheckpoint;
        setRaw(false);
        if (await o.run(["rollback", id, "--dir", dir], session).catch(() => 1) === 0) delete session.lastCheckpoint;
        return;
      }
      case "verify": {
        if (!rest) note(session.verify ? `verify: ${session.verify}` : "verify: none. Set one with /verify npm test");
        else if (rest === "off") { delete session.verify; note("verify: off"); }
        else { session.verify = rest; note(`verify: ${rest}. It runs in a scratch copy after each task; changes are only offered when it passes`); }
        return;
      }
      case "yes": session.auto = !session.auto; note(session.auto ? "the agent may run project commands without asking (only for projects you trust)" : "the agent asks before running project commands"); return;
      case "theme": {
        let pickName = restWords[0] as ThemeName | undefined;
        if (!pickName) {
          const i = await pick("Choose a theme", THEMES.map((t) => ({ label: t, hint: t === st.theme ? "(current)" : "" })), THEMES.indexOf(st.theme));
          if (i === undefined) return;
          pickName = THEMES[i];
        }
        if (!pickName || !THEMES.includes(pickName)) { fail(`themes: ${THEMES.join(", ")}`); return; }
        Object.assign(st, makeStyle(colour, pickName));
        settings.theme = pickName; saveSettings();
        note(`${pickName} theme`);
        return;
      }
      default:
        if (FORWARD.has(name)) {
          setRaw(false);
          await o.run([name, ...restWords.filter(Boolean)], session).catch((e: unknown) => fail(e instanceof Error ? e.message : String(e)));
          println();
          return;
        }
        fail(`unknown command /${word}. /help lists them.`);
    }
  };

  // ---- main ----
  try {
    println();
    target = await chooseModel(false).catch((e: unknown) => { fail(e instanceof Error ? e.message : String(e)); return undefined; });
    if (out.isTTY) out.write("\x1b[2J\x1b[H");
    welcome();
    if (!target) note("no model loaded: /model chooses one, /pull downloads a small one");
    else if (target.engineNote) note(`running on llama-server because ${target.engineNote}`);
    println();
    for (;;) {
      const line = await readInput();
      if (line === null) break;
      try {
        if (line.startsWith("/")) {
          println(`${st.dim("❯")} ${st.dim(line)}`);
          if ((await slash(line)) === "exit") break;
        } else if (mode === "chat") await turn(line);
        else await agentTask(line);
      } catch (e) { fail(e instanceof Error ? e.message : String(e)); }
      try { mkdirSync(stateDir, { recursive: true, mode: 0o700 }); writeFileSync(join(stateDir, "history"), `${editor.history.slice(-500).map((h) => h.replace(/\n/g, " ")).join("\n")}\n`, { mode: 0o600 }); } catch { /* history is a convenience */ }
    }
  } finally {
    restore();
    await stopTarget();
    process.off("exit", onExit); process.off("SIGHUP", sighup); process.off("SIGTERM", sigterm);
  }
  println();
  println(st.dim(stats.turns > 0 ? `  Goodbye. ${stats.completion} tokens generated in ${stats.turns} repl${stats.turns === 1 ? "y" : "ies"}.` : "  Goodbye."));
  return 0;
}
