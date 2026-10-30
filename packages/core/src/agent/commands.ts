/**
 * Command policy for the agent. Commands are never given to a shell: this module splits a command line into program and arguments
 * itself, accepts only a short list of programs with the sub-commands that are safe for each, and refuses anything that would need a
 * shell to mean something (pipes, redirects, substitution, variables, chaining other than `&&`). Nothing a model writes can then be
 * "injected", because there is no interpreter to inject into.
 *
 * What this is NOT: a sandbox. A permitted program such as `node` or `npm test` runs project code, and project code can do anything the
 * user can. That is why those commands need the user's approval by default and run in a scratch copy of the project with a minimal
 * environment. The policy keeps a confused or prompt-injected model from reaching for `rm -rf ~` or `curl | sh`; it cannot make
 * running untrusted code safe.
 */

export type Risk = "inspect" | "project";

export interface CommandStep {
  argv: string[];
  risk: Risk;
  /** True when the program is expected to download things (installs). */
  network: boolean;
}

export type CommandPlan =
  | { ok: true; steps: CommandStep[]; requiresApproval: boolean }
  | { ok: false; reason: string };

const MAX_LENGTH = 2000;
const MAX_STEPS = 6;

function deny(reason: string): CommandPlan {
  return { ok: false, reason };
}

/** Splits one command line into `&&`-separated argument lists. Returns an error string if it needs a shell. */
export function tokenize(line: string): { steps: string[][] } | { error: string } {
  if (line.length === 0) return { error: "the command is empty" };
  if (line.length > MAX_LENGTH) return { error: `the command is longer than ${MAX_LENGTH} characters` };
  if (/[\0\r\n]/.test(line)) return { error: "the command may not contain newlines or NUL bytes" };
  const steps: string[][] = [];
  let cur: string[] = [];
  let word = "";
  let has = false;
  let quote: "'" | '"' | null = null;
  const endWord = (): void => { if (has) cur.push(word); word = ""; has = false; };
  for (let i = 0; i < line.length; i++) {
    const c = line[i] as string;
    if (quote) {
      if (c === quote) { quote = null; continue; }
      if (quote === '"' && (c === "$" || c === "`")) return { error: `"${c}" inside double quotes needs a shell to expand` };
      if (quote === '"' && c === "\\" && i + 1 < line.length) { word += line[++i]; continue; }
      word += c; has = true;
      continue;
    }
    if (c === "'" || c === '"') { quote = c; has = true; continue; }
    if (c === "\\") { if (i + 1 >= line.length) return { error: "the command ends with a backslash" }; word += line[++i]; has = true; continue; }
    if (c === " " || c === "\t") { endWord(); continue; }
    if (c === "&") {
      if (line[i + 1] !== "&") return { error: "a lone & (background job) is not allowed" };
      endWord();
      if (cur.length === 0) return { error: "&& needs a command before it" };
      steps.push(cur); cur = []; i++;
      continue;
    }
    if (c === "|") return { error: "pipes are not allowed; run the steps separately" };
    if (c === ";") return { error: "; is not allowed; chain with && instead" };
    if (c === "<" || c === ">") return { error: "redirection is not allowed; use the write_file tool to change files" };
    if (c === "`" || c === "$") return { error: `"${c}" (substitution or variables) needs a shell` };
    if (c === "(" || c === ")" || c === "{" || c === "}") return { error: `"${c}" needs a shell` };
    if (c === "#" && !has) return { error: "comments are not allowed" };
    word += c; has = true;
  }
  if (quote) return { error: "unterminated quote" };
  endWord();
  if (cur.length === 0) return { error: line.trimEnd().endsWith("&&") ? "&& needs a command after it" : "the command is empty" };
  steps.push(cur);
  if (steps.length > MAX_STEPS) return { error: `at most ${MAX_STEPS} commands can be chained` };
  return { steps };
}

/** An argument that names a place outside the project, or something the shell would have expanded to one. */
function pathProblem(arg: string): string | null {
  const value = arg.startsWith("-") ? (arg.includes("=") ? arg.slice(arg.indexOf("=") + 1) : "") : arg;
  if (value === "") return null;
  const v = value.replace(/\\/g, "/");
  if (v.startsWith("/") || /^[A-Za-z]:/.test(v) || v.startsWith("~") || v.startsWith("//")) return `"${arg}" names a place outside the project`;
  if (v.split("/").includes("..")) return `"${arg}" leaves the project folder`;
  return null;
}

const INSPECT = new Set(["ls", "cat", "head", "tail", "wc", "grep", "rg", "diff", "sort", "uniq", "echo", "pwd", "find", "tree", "stat", "basename", "dirname"]);
const GIT_READ = new Set(["status", "diff", "log", "show", "branch", "rev-parse", "ls-files", "blame", "shortlog"]);
const PKG = new Set(["npm", "pnpm", "yarn", "bun"]);
const PKG_RUN = new Set(["test", "run", "run-script", "t", "tst", "start", "ci", "install", "i", "add", "exec"]);
const PKG_NETWORK = new Set(["ci", "install", "i", "add"]);
const RUNNERS = new Set(["node", "python", "python3", "pytest", "tsc", "vitest", "jest", "eslint", "prettier", "make", "ruff", "mypy", "deno"]);
const GO = new Set(["test", "build", "vet", "fmt", "run", "mod"]);
const CARGO = new Set(["test", "build", "check", "fmt", "clippy", "run"]);
/** Flags that turn a harmless program into a general executor or a writer. */
const DANGEROUS_FLAGS: Record<string, RegExp> = {
  find: /^-(exec|execdir|ok|okdir|delete|fprint|fprintf|fls)$/,
  sed: /^-i/,
  sort: /^-o/,
  tree: /^-o$/,
  rg: /^--(pre|pre-glob)/,
  grep: /^--(include-from|exclude-from)$/,
  git: /^(--exec-path|-c|--git-dir|--work-tree|--output|-o)(=|$)/,
  diff: /^--to-file$/,
};

function classify(argv: string[]): CommandStep | string {
  const [program, ...args] = argv as [string, ...string[]];
  if (/[\\/]/.test(program)) return `"${program}" must be a plain program name, not a path`;
  const name = program.toLowerCase().replace(/\.(exe|cmd|bat)$/, "");
  // Judge the program before its arguments, so the reason names the real problem.
  if (!INSPECT.has(name) && name !== "git" && !PKG.has(name) && name !== "go" && name !== "cargo" && !RUNNERS.has(name) && name !== "npx") return `"${program}" is not on the list of programs the agent may run`;
  const bad = args.map(pathProblem).find((p) => p !== null);
  if (bad) return bad;
  const flagBad = DANGEROUS_FLAGS[name];
  if (flagBad) {
    const f = args.find((a) => flagBad.test(a));
    if (f) return `"${f}" is not allowed with ${name}`;
  }
  if (INSPECT.has(name)) return { argv, risk: "inspect", network: false };
  if (name === "git") {
    const sub = args.find((a) => !a.startsWith("-"));
    if (!sub || !GIT_READ.has(sub)) return `git ${sub ?? ""} is not allowed; the agent may only read repository state (${[...GIT_READ].join(", ")})`;
    if (sub === "branch" && args.some((a) => /^(-d|-D|-m|-M|-c|-C|--delete|--move|--copy|--force)$/.test(a))) return "git branch may only list branches";
    return { argv, risk: "inspect", network: false };
  }
  if (PKG.has(name)) {
    const sub = args.find((a) => !a.startsWith("-"));
    if (!sub || !PKG_RUN.has(sub)) return `${name} ${sub ?? ""} is not allowed; use test, run, ci or install`;
    return { argv, risk: "project", network: PKG_NETWORK.has(sub) };
  }
  if (name === "go") { const sub = args[0]; return sub && GO.has(sub) ? { argv, risk: "project", network: sub === "mod" } : `go ${sub ?? ""} is not allowed`; }
  if (name === "cargo") { const sub = args[0]; return sub && CARGO.has(sub) ? { argv, risk: "project", network: false } : `cargo ${sub ?? ""} is not allowed`; }
  if (RUNNERS.has(name)) return { argv, risk: "project", network: false };
  if (name === "npx") return "npx downloads and runs arbitrary packages; add the tool as a dev dependency and use `npm run` instead";
  return `"${program}" is not on the list of programs the agent may run`;
}

export function planCommand(line: string): CommandPlan {
  const t = tokenize(line);
  if ("error" in t) return deny(t.error);
  const steps: CommandStep[] = [];
  for (const argv of t.steps) {
    const c = classify(argv);
    if (typeof c === "string") return deny(c);
    steps.push(c);
  }
  return { ok: true, steps, requiresApproval: steps.some((s) => s.risk === "project" || s.network) };
}
