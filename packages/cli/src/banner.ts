/**
 * The welcome screen of the interactive session: the name in large block letters, then where you are and what to type.
 * Colour is used only on a terminal that asked for it (not when piped, not with NO_COLOR); the text is the same either way.
 */

const GLYPHS: Record<string, string[]> = {
  C: [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"],
  L: ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"],
  U: ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
  T: ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
  O: [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
  R: ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
};

/** The six rows of "CLUTOR" in block letters. */
export const WORDMARK: string[] = Array.from({ length: 6 }, (_, row) => [..."CLUTOR"].map((ch) => GLYPHS[ch]?.[row] ?? "").join(""));
export const WORDMARK_WIDTH = Math.max(...WORDMARK.map((r) => r.length));

/** Top to bottom: cyan, through blue, to violet (xterm 256-colour codes). */
const GRADIENT = [51, 45, 39, 33, 63, 99];

export interface BannerOptions {
  version: string;
  folder: string;
  model: string;
  /** Width of the terminal in columns; a narrower terminal than the block letters gets a one-line title. */
  columns?: number;
  color?: boolean;
}

export function useColor(env: NodeJS.ProcessEnv = process.env, isTTY: boolean = !!process.stdout.isTTY): boolean {
  return isTTY && !("NO_COLOR" in env) && env["TERM"] !== "dumb";
}

export function banner(o: BannerOptions): string {
  const paint = (code: number | string, s: string): string => (o.color ? `\x1b[${typeof code === "number" ? `38;5;${code}` : code}m${s}\x1b[0m` : s);
  const dim = (s: string): string => paint("2", s);
  const wide = (o.columns ?? 80) >= WORDMARK_WIDTH + 2;
  const lines: string[] = [""];
  if (wide) lines.push(...WORDMARK.map((r, i) => paint(GRADIENT[i] ?? 99, r)));
  else lines.push(paint(45, "C L U T O R"));
  lines.push("", `${paint("1", "clutor interactive session")} ${dim(`v${o.version}`)}`, dim("big models on small machines, with honest numbers"), "");
  lines.push(`  ${dim("folder")}  ${o.folder}`, `  ${dim("model")}   ${o.model}`, "");
  lines.push(`  ${dim("tips")}    type a task in plain words and the agent works on this folder`);
  lines.push(`          its changes are held aside; nothing is written without your approval`);
  lines.push(`          /help lists commands · /plan <model> · /cache · /exit`, "");
  return lines.join("\n");
}
