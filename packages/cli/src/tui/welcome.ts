import { type Style, width, wrap } from "./ansi";

/** The start screen: a bordered panel with a welcome and the model on the left, tips and recent conversations on the right. */

export interface WelcomeInfo {
  version: string;
  cwd: string;
  /** Empty while there is no model. */
  model: string;
  detail: string;
  recent: string[];
}

const centre = (s: string, w: number): string => {
  const n = Math.max(0, w - width(s));
  return " ".repeat(Math.floor(n / 2)) + s + " ".repeat(Math.ceil(n / 2));
};
/** Cuts to `n` columns with an ellipsis; a path keeps its end, where the useful part is. */
const cut = (s: string, n: number, keepEnd = false): string => (width(s) <= n ? s : keepEnd ? `…${s.slice(s.length - (n - 1))}` : `${s.slice(0, n - 1)}…`);
const pad = (s: string, w: number): string => s + " ".repeat(Math.max(0, w - width(s)));

export function welcomeLines(st: Style, cols: number, i: WelcomeInfo): string[] {
  const w = Math.min(Math.max(cols, 40), 104);
  const home = process.env["HOME"] ?? "";
  const dir = home && i.cwd.startsWith(home) ? `~${i.cwd.slice(home.length)}` : i.cwd;
  const b = st.accent;
  const title = ` Clutor v${i.version} `;
  const top = `${b("╭───")}${st.bold(b(title))}${b(`${"─".repeat(Math.max(0, w - 5 - title.length))}╮`)}`;
  const bottom = b(`╰${"─".repeat(w - 2)}╯`);

  const tips = [
    ["Type to chat", "your messages stay on this machine"],
    ["shift+tab", "switch to the coding agent"],
    ["/", "commands · @file attaches a file · esc stops a reply"],
  ];

  if (w < 76) {
    const inner = w - 4;
    const rows = [
      centre(`${st.accent("✻")} ${st.bold("Welcome")}`, inner),
      ...(i.model ? [centre(st.dim(i.model), inner)] : []),
      st.dim(cut(dir, inner, true)),
      ...tips.flatMap(([k, v]) => wrap(`${st.accent(k as string)} ${st.dim(v as string)}`, inner, 2)),
    ];
    return [top, ...rows.map((r) => `${b("│")} ${pad(r, inner)} ${b("│")}`), bottom];
  }

  const leftW = Math.floor((w - 3) * 0.42);
  const rightW = w - 3 - leftW;
  const left = [
    "",
    centre(st.bold("Welcome back!"), leftW),
    "",
    centre(st.accent("✻"), leftW),
    centre(st.accent("C L U T O R"), leftW),
    "",
    ...(i.model ? [centre(cut(i.model, leftW - 2), leftW), centre(st.dim(cut(i.detail, leftW - 2)), leftW)] : [centre(st.warn("no model loaded"), leftW)]),
    centre(st.dim(cut(dir, leftW - 2, true)), leftW),
  ];
  const rule = st.dim("─".repeat(rightW - 2));
  const right = [
    st.bold(st.accent("Tips for getting started")),
    ...tips.flatMap(([k, v]) => wrap(`${st.accent(k as string)}  ${st.dim(v as string)}`, rightW - 2, 2)),
    rule,
    st.bold(st.accent("Recent conversations")),
    ...(i.recent.length ? i.recent.slice(0, 3).map((r) => st.dim(cut(`· ${r}`, rightW - 2))) : [st.dim("nothing yet")]),
  ];
  const n = Math.max(left.length, right.length);
  const rows: string[] = [];
  for (let r = 0; r < n; r++) rows.push(`${b("│")}${pad(left[r] ?? "", leftW)}${b("│")} ${pad(right[r] ?? "", rightW - 1)}${b("│")}`);
  return [top, ...rows, bottom];
}
