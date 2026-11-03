#!/usr/bin/env node
import { mkdirSync, readFileSync, appendFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { createInterface, type Interface } from "node:readline";
import { main } from "./cli";
import { useColor } from "./banner";

// One readline interface for the whole process, so arrow-key history works across prompts and Ctrl-D ends an interactive session.
let rl: Interface | undefined;
const historyFile = join(homedir(), ".clutor", "history");
function readline(): Interface {
  if (rl) return rl;
  let history: string[] = [];
  try { history = readFileSync(historyFile, "utf8").split("\n").filter(Boolean).slice(-500).reverse(); } catch { /* no history yet */ }
  rl = createInterface({ input: process.stdin, output: process.stderr, history, historySize: 500 });
  return rl;
}
const prompt = (q: string): Promise<string> => new Promise((resolve) => {
  const r = readline();
  const onClose = (): void => resolve("/exit");
  r.once("close", onClose);
  r.question(q, (a) => {
    r.off("close", onClose);
    if (a.trim()) { try { mkdirSync(join(homedir(), ".clutor"), { recursive: true, mode: 0o700 }); appendFileSync(historyFile, `${a.replace(/\n/g, " ")}\n`, { mode: 0o600 }); } catch { /* history is a convenience */ } }
    resolve(a);
  });
});

main(process.argv.slice(2), {
  out: (l) => console.log(l),
  write: (s) => void process.stdout.write(s),
  err: (l) => console.error(l),
  cwd: process.cwd(),
  ...(process.stdin.isTTY ? { prompt } : {}),
  ...(process.stdin.isTTY && process.stdout.isTTY && !process.env["CLUTOR_PLAIN"]
    ? { tui: async (run, version, argv) => (await import("./tui/app")).runTui({ io: { out: (l) => console.log(l), write: (s) => void process.stdout.write(s), err: (l) => console.error(l), cwd: process.cwd() }, run, version, argv, color: useColor() }) }
    : {}),
}).then(
  (code) => process.exit(code),
  (e: unknown) => {
    console.error(e instanceof Error ? e.message : String(e));
    process.exit(1);
  },
);
