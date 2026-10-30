import { TOOLS } from "@clutor/core";
import { loadConfig, ConfigError, type ClutorConfig } from "./config";
import { table } from "./format";
import type { Io, SessionState } from "./cli";

/**
 * The interactive session that `clutor` opens when it is started with no command in a terminal. Text you type is a task for the coding
 * agent (its changes are held aside until you approve them); a line starting with `/` is a command. Most commands are the ordinary
 * clutor commands, so they behave exactly as they do outside the session.
 */

type Run = (argv: string[], session: SessionState) => Promise<number>;
type EngineView = (config: ClutorConfig, what: "layers" | "cache") => Promise<void>;

interface Slash { usage: string; help: string }

/** Commands that are forwarded unchanged to the normal CLI. */
const FORWARD: Record<string, Slash> = {
  status: { usage: "/status", help: "machine, model backend, memory plan, receipts" },
  doctor: { usage: "/doctor", help: "check this machine and the configuration" },
  models: { usage: "/models", help: "model catalog and the .gguf files in your models folder" },
  plan: { usage: "/plan <model> [--ram GiB]", help: "will it fit, where the memory goes, how fast it could possibly run" },
  simulate: { usage: "/simulate <model> [--cache GiB]", help: "replay a routing trace through the expert cache (synthetic)" },
  inspect: { usage: "/inspect <file.gguf>", help: "read a model file's header" },
  bench: { usage: "/bench [--file model.gguf]", help: "measure real disk reads; with the engine, compare cache strategies" },
  stream: { usage: "/stream <model.gguf>", help: "stream a model's experts through the native engine (synthetic routing)" },
  pull: { usage: "/pull [id]", help: "list or download a small test model" },
  init: { usage: "/init", help: "create clutor.config.json and a models folder here" },
  backend: { usage: "/backend", help: "show how llama-server would be started" },
  ledger: { usage: "/ledger", help: "verify the receipt log and list recent tasks and checkpoints" },
  decide: { usage: '/decide --prompt "<text>" --choices "a,b"', help: "one forward pass: probability of each choice" },
};

const OWN: Record<string, Slash> = {
  help: { usage: "/help", help: "this list" },
  model: { usage: "/model", help: "which model this session talks to" },
  verify: { usage: "/verify [command|off]", help: "run this check after every task and only offer changes that pass it" },
  yes: { usage: "/yes", help: "toggle: run project commands without asking (only for projects you trust)" },
  undo: { usage: "/undo", help: "restore the files changed by the last applied task" },
  tools: { usage: "/tools", help: "what the agent can do" },
  cost: { usage: "/cost", help: "tokens used this session, as reported by the model server" },
  history: { usage: "/history", help: "tasks done in this session" },
  clear: { usage: "/clear", help: "forget this session's task history" },
  layers: { usage: "/layers", help: "where the model file's bytes are, per layer (native engine)" },
  cache: { usage: "/cache", help: "a streaming pass over the model file: cache, prefetch, disk waits" },
  exit: { usage: "/exit", help: "leave (also /quit, Ctrl-D)" },
};

/** Splits a line into words, honouring single and double quotes. */
export function splitWords(line: string): string[] {
  const out: string[] = [];
  let cur = "", quote: string | undefined, has = false;
  for (const ch of line) {
    if (quote) { if (ch === quote) quote = undefined; else cur += ch; }
    else if (ch === '"' || ch === "'") { quote = ch; has = true; }
    else if (/\s/.test(ch)) { if (cur || has) out.push(cur); cur = ""; has = false; }
    else cur += ch;
  }
  if (cur || has) out.push(cur);
  return out;
}

function helpText(): string {
  const rows = [...Object.values(OWN), ...Object.values(FORWARD)].map((c) => [c.usage, c.help]);
  return `Type a task in plain words and the agent works on this folder. Its changes are held aside; you see the diff and approve before anything is written.\n\n${table(rows)}`;
}

export async function runRepl(io: Io, run: Run, engine: EngineView): Promise<number> {
  const ask = io.prompt as NonNullable<Io["prompt"]>;
  const session: SessionState = { auto: false, notes: [], usage: { promptTokens: 0, completionTokens: 0 } };
  const load = (): ClutorConfig | undefined => {
    try { return loadConfig(io.cwd).config; } catch (e) { io.err(e instanceof ConfigError || e instanceof Error ? e.message : String(e)); return undefined; }
  };
  const cfg = load();
  const model = (c: ClutorConfig | undefined): string => !c ? "not configured" : c.backend.kind === "openai" ? `${c.backend.model ?? "default"} at ${c.backend.baseUrl}` : c.backend.llamaServer.model ?? "not configured (set backend.llamaServer.model)";
  io.out(`clutor interactive session\n  folder  ${io.cwd}\n  model   ${model(cfg)}\n  Type a task, or /help for commands. Nothing is written to your files without your approval.`);

  try {
    for (;;) {
      const line = (await ask("clt> ")).trim();
      if (!line) continue;
      if (!line.startsWith("/")) { await guarded(io, () => run(["code", line, "--dir", io.cwd], session)); continue; }

      const [word = "", ...rest] = splitWords(line.slice(1));
      const name = word.toLowerCase();
      if (name === "exit" || name === "quit") return 0;
      if (name === "help" || name === "?") { io.out(helpText()); continue; }
      if (name === "model") { io.out(`model: ${model(load())}${session.backend ? " (running)" : ""}\nverify: ${session.verify ?? "none"}   auto-approve: ${session.auto ? "on" : "off"}`); continue; }
      if (name === "verify") {
        const v = rest.join(" ").trim();
        if (!v) io.out(session.verify ? `verify: ${session.verify}` : "verify: none. Set one with /verify npm test");
        else if (v === "off") { delete session.verify; io.out("verify: off"); }
        else { session.verify = v; io.out(`verify: ${v}. It runs in a scratch copy after each task; changes are only offered when it passes.`); }
        continue;
      }
      if (name === "yes") { session.auto = !session.auto; io.out(session.auto ? "auto-approve ON: commands from the agent run without asking." : "auto-approve off."); continue; }
      if (name === "undo") {
        if (!session.lastCheckpoint) { io.out("Nothing to undo in this session."); continue; }
        const { id, dir } = session.lastCheckpoint;
        if (await guarded(io, () => run(["rollback", id, "--dir", dir], session)) === 0) delete session.lastCheckpoint;
        continue;
      }
      if (name === "tools") { io.out(table(TOOLS.map((t) => [t.name, t.description.split(". ")[0] ?? ""]))); continue; }
      if (name === "cost") {
        const u = session.usage;
        io.out(`${u.promptTokens} prompt + ${u.completionTokens} generated tokens this session (reported by the model server; local models cost no money).`);
        continue;
      }
      if (name === "history") { io.out(session.notes.length ? session.notes.map((n, i) => `${i + 1}. ${n}`).join("\n") : "No tasks yet."); continue; }
      if (name === "clear") { session.notes.length = 0; io.out("(session history cleared)"); continue; }
      if (name === "layers" || name === "cache") {
        const c = load();
        if (c) await guarded(io, async () => { await engine(c, name); return 0; });
        continue;
      }
      if (FORWARD[name]) { await guarded(io, () => run([name, ...rest], session)); continue; }
      io.err(`unknown command /${word}. /help lists them.`);
    }
  } finally {
    try { await session.backend?.stop(); } catch { /* the server may already be gone */ }
  }
}

async function guarded(io: Io, f: () => Promise<number>): Promise<number | undefined> {
  try { return await f(); } catch (e) { io.err(e instanceof Error ? e.message : String(e)); return undefined; }
}
