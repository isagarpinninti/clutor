import { type Style, wrap } from "./ansi";

/** The start screen, in the app's style: the logo (six hexagons) and name, the model and where it runs, a few quiet tips, recent conversations. No box. */

export interface WelcomeInfo {
  version: string;
  cwd: string;
  /** Empty while there is no model. */
  model: string;
  detail: string;
  recent: string[];
}

/** Cuts to `n` columns with an ellipsis; a path keeps its end, where the useful part is. */
const cut = (s: string, n: number, keepEnd = false): string => (s.length <= n ? s : keepEnd ? `…${s.slice(s.length - (n - 1))}` : `${s.slice(0, n - 1)}…`);

/** The logo (assets/clutor-logo.svg: six hexagons, the one on the right missing and the middle one faded), rasterized by scripts/gen-terminal-logo.py and drawn two pixel rows to a line with half blocks. */
const LOGO = [
  "......##.......#...",
  ".....#####...#####.",
  "....#######.#######",
  "....#######.#######",
  "....#######.#######",
  "....#######.#######",
  "......###.....###..",
  "..##......oo.......",
  ".#####...ooooo.....",
  "######..ooooooo....",
  "######..ooooooo....",
  "######..ooooooo....",
  "######..ooooooo....",
  "..###.....ooo......",
  "......##.......#...",
  ".....#####...#####.",
  "....#######.#######",
  "....#######.#######",
  "....#######.#######",
  "....#######.#######",
  "......###.....###..",
  "..................."
];
const SOLID = 63, FADED = 17;   // the app's blue, and the same blue at the faded hexagon's opacity on a dark background
function markLines(st: Style): string[] {
  const paint = (ch: string): number | undefined => (ch === "#" ? SOLID : ch === "o" ? FADED : undefined);
  const out: string[] = [];
  for (let r = 0; r < LOGO.length; r += 2) {
    let line = "";
    const top = LOGO[r] as string, bot = LOGO[r + 1] as string;
    for (let c = 0; c < top.length; c++) {
      const t = paint(top[c] as string), b = paint(bot[c] as string);
      if (t === undefined && b === undefined) line += " ";
      else if (t === b) line += st.fg(t as number, "█");
      else if (b === undefined) line += st.fg(t as number, "▀");
      else if (t === undefined) line += st.fg(b, "▄");
      else line += st.bg(b, st.fg(t, "▀"));
    }
    out.push(line);
  }
  return out;
}

export function welcomeLines(st: Style, cols: number, i: WelcomeInfo): string[] {
  const w = Math.min(Math.max(cols, 40), 100) - 4;
  const home = process.env["HOME"] ?? "";
  const dir = home && i.cwd.startsWith(home) ? `~${i.cwd.slice(home.length)}` : i.cwd;
  const mark = markLines(st);
  const text = [
    `${st.bold("Clutor")} ${st.dim(`v${i.version}`)}`,
    i.model ? cut(i.model, w - 8) : st.warn("no model loaded (use /model)"),
    ...(i.model ? [st.dim(cut(i.detail, w - 8))] : []),
    st.dim(cut(dir, w - 8, true)),
  ];
  const lines: string[] = [""];
  const top = 3;   // the text sits beside the middle of the mark
  for (let r = 0; r < mark.length; r++) lines.push(`  ${mark[r]}   ${text[r - top] ?? ""}`);
  lines.push("");
  const tips: [string, string][] = [
    ["Type to chat", "your messages stay on this machine"],
    ["shift+tab", "switch to the coding agent"],
    ["/", "commands · @file attaches a file · esc stops a reply"],
  ];
  const kw = Math.max(...tips.map(([k]) => k.length)) + 2;
  for (const [k, v] of tips) lines.push(...wrap(`  ${st.accent(k.padEnd(kw))}${st.dim(v)}`, w, kw + 2));
  if (i.recent.length) {
    lines.push("", `  ${st.dim("Recent")}`);
    for (const r of i.recent.slice(0, 3)) lines.push(`  ${st.dim("·")} ${cut(r, w - 4)}`);
  }
  lines.push("");
  return lines;
}
