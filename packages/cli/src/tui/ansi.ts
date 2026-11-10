/**
 * Terminal primitives for the interactive session: styles that switch off with NO_COLOR, display width, ANSI-aware wrapping, and a
 * redrawable region at the bottom of the screen (the part that holds the input box, the spinner and the reply being typed).
 */

const ESC = "\x1b";
const CSI_OR_SGR = /^\x1b\[[0-9;?]*[ -/]*[@-~]/;

export function stripAnsi(s: string): string { return s.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, ""); }

/** Columns a code point takes: 0 for combining and control characters, 2 for East Asian wide characters and emoji, else 1. */
export function charWidth(cp: number): number {
  if (cp < 32 || (cp >= 0x7f && cp < 0xa0)) return 0;
  if ((cp >= 0x300 && cp <= 0x36f) || (cp >= 0x200b && cp <= 0x200f) || (cp >= 0xfe00 && cp <= 0xfe0f) || cp === 0x2060) return 0;
  if ((cp >= 0x1100 && cp <= 0x115f) || (cp >= 0x2e80 && cp <= 0xa4cf) || (cp >= 0xac00 && cp <= 0xd7a3) || (cp >= 0xf900 && cp <= 0xfaff) ||
      (cp >= 0xfe30 && cp <= 0xfe6f) || (cp >= 0xff00 && cp <= 0xff60) || (cp >= 0xffe0 && cp <= 0xffe6) || (cp >= 0x1f300 && cp <= 0x1f64f) ||
      (cp >= 0x1f900 && cp <= 0x1f9ff) || (cp >= 0x20000 && cp <= 0x3fffd)) return 2;
  return 1;
}

export function width(s: string): number {
  let w = 0;
  for (const ch of stripAnsi(s)) w += charWidth(ch.codePointAt(0) as number);
  return w;
}

/** Cuts `s` to at most `max` visible columns, keeping escape sequences, and closes any open style. */
export function fit(s: string, max: number): string {
  if (width(s) <= max) return s;
  let out = "", w = 0;
  for (let i = 0; i < s.length;) {
    const m = CSI_OR_SGR.exec(s.slice(i, i + 24));
    if (s[i] === ESC && m) { out += m[0]; i += m[0].length; continue; }
    const cp = s.codePointAt(i) as number;
    const ch = String.fromCodePoint(cp);
    const cw = charWidth(cp);
    if (w + cw > max) break;
    out += ch; w += cw; i += ch.length;
  }
  return `${out}${ESC}[0m`;
}

/**
 * Wraps one logical line to `cols` columns at spaces (a word longer than a row is split). Escape sequences take no room. Continuation
 * rows are indented by `hang` spaces. Styles stay open across the break, which a terminal carries over to the next row.
 */
export function wrap(line: string, cols: number, hang = 0): string[] {
  cols = Math.max(8, cols);
  if (width(line) <= cols) return [line];
  const rows: string[] = [];
  let row = "", rw = 0;
  const pad = " ".repeat(hang);
  const flush = (): void => { rows.push(row.replace(/ +$/, "")); row = pad; rw = hang; };
  // Tokens: runs of spaces, escape sequences, and words.
  const tokens = line.match(/\x1b\[[0-9;?]*[ -/]*[@-~]| +|[^ \x1b]+/g) ?? [];
  for (const t of tokens) {
    if (t[0] === ESC) { row += t; continue; }
    if (t[0] === " ") { if (rw + t.length <= cols && rw > hang) { row += t; rw += t.length; } else if (rw > hang) flush(); continue; }
    const tw = width(t);
    if (rw + tw <= cols) { row += t; rw += tw; continue; }
    if (rw > hang) flush();
    if (tw <= cols - hang) { row += t; rw += tw; continue; }
    for (const ch of t) {
      const cw = charWidth(ch.codePointAt(0) as number);
      if (rw + cw > cols) flush();
      row += ch; rw += cw;
    }
  }
  rows.push(row.replace(/ +$/, ""));
  return rows;
}

export type ThemeName = "dark" | "light" | "ocean" | "mono";
export const THEMES: ThemeName[] = ["dark", "light", "ocean", "mono"];

/** xterm 256-colour numbers. `mono` has none: emphasis only. */
export interface Palette { accent: number; hi: number; ok: number; warn: number; bad: number; code: number; kw: number; str: number; num: number; com: number; fn: number; ty: number }

const PALETTES: Record<Exclude<ThemeName, "mono">, Palette> = {
  dark: { accent: 69, hi: 111, ok: 114, warn: 221, bad: 203, code: 180, kw: 176, str: 114, num: 215, com: 244, fn: 111, ty: 80 },
  light: { accent: 166, hi: 208, ok: 28, warn: 130, bad: 160, code: 94, kw: 90, str: 28, num: 130, com: 245, fn: 25, ty: 30 },
  ocean: { accent: 75, hi: 159, ok: 79, warn: 222, bad: 204, code: 117, kw: 141, str: 114, num: 215, com: 244, fn: 111, ty: 80 },
};

export interface Style {
  color: boolean;
  theme: ThemeName;
  pal: Palette;
  bold(s: string): string; dim(s: string): string; italic(s: string): string; underline(s: string): string; inverse(s: string): string;
  /** xterm 256-colour foreground. */
  fg(n: number, s: string): string;
  bg(n: number, s: string): string;
  accent(s: string): string; hi(s: string): string; ok(s: string): string; warn(s: string): string; bad(s: string): string; code(s: string): string;
  /** Syntax colours: keyword, string, number, comment, function, type. */
  tok(kind: "kw" | "str" | "num" | "com" | "fn" | "ty", s: string): string;
}

export function makeStyle(color: boolean, theme: ThemeName = "dark"): Style {
  const w = (open: string, close: string) => (s: string): string => (color ? `${ESC}[${open}m${s}${ESC}[${close}m` : s);
  const mono = theme === "mono";
  const pal = PALETTES[mono ? "dark" : theme];
  const fg = (n: number, s: string): string => (color && !mono ? `${ESC}[38;5;${n}m${s}${ESC}[39m` : s);
  const bold = w("1", "22"), italic = w("3", "23");
  const faint = w("2", "22");
  /** Quiet text: a mid grey like the app's (works in every terminal, unlike the faint attribute); emphasis-only themes keep the attribute. */
  const dim = (s: string): string => (color && !mono ? `${ESC}[38;5;245m${s}${ESC}[39m` : faint(s));
  const hue = (n: number, emphasis: (s: string) => string = (s) => s) => (s: string): string => (mono ? emphasis(s) : fg(n, s));
  return {
    color, theme, pal,
    bold, dim, italic, underline: w("4", "24"), inverse: w("7", "27"),
    fg, bg: (n, s) => (color && !mono ? `${ESC}[48;5;${n}m${s}${ESC}[49m` : s),
    accent: hue(pal.accent, bold), hi: hue(pal.hi, bold), ok: hue(pal.ok), warn: hue(pal.warn, bold), bad: hue(pal.bad, bold), code: hue(pal.code),
    tok: (kind, s) => (mono ? (kind === "com" ? dim(s) : kind === "kw" ? bold(s) : s) : kind === "com" ? fg(pal.com, s) : fg(pal[kind], s)),
  };
}

/**
 * A block of lines at the bottom of the screen that can be redrawn in place. `draw` erases what it drew last time and writes the new
 * lines, so a spinner can tick and the input box can follow the cursor without scrolling the screen. Every line must fit in one row
 * (use `wrap` and `fit`), otherwise the terminal's own wrapping makes the row count wrong.
 */
export class Region {
  private cursorRow = 0;
  private lastRow = 0;
  constructor(private readonly out: { write(s: string): unknown; columns?: number }) {}
  get cols(): number { return this.out.columns && this.out.columns > 0 ? this.out.columns : 80; }

  draw(lines: string[], cursor?: { row: number; col: number }): void {
    let s = `${ESC}[?2026h${ESC}[?25l${this.up()}\r${ESC}[J${lines.join("\n")}${ESC}[0m`;
    let row = lines.length - 1;
    if (cursor) {
      if (row > cursor.row) s += `${ESC}[${row - cursor.row}A`;
      else if (row < cursor.row) s += `${ESC}[${cursor.row - row}B`;
      s += "\r";
      if (cursor.col > 0) s += `${ESC}[${cursor.col}C`;
      row = cursor.row;
      s += `${ESC}[?25h`;
    }
    this.lastRow = lines.length - 1;
    this.cursorRow = row;
    this.out.write(`${s}${ESC}[?2026l`);
  }

  /** Erases the region and leaves the cursor where it began. */
  clear(): void {
    this.out.write(`${this.up()}\r${ESC}[J`);
    this.cursorRow = 0;
  }

  /** Forgets the region but leaves what it drew on screen, with the cursor on the line after it. */
  freeze(): void {
    const down = this.lastRow - this.cursorRow;
    this.out.write(`${down > 0 ? `${ESC}[${down}B` : ""}${ESC}[?25h\n`);
    this.cursorRow = 0;
    this.lastRow = 0;
  }

  private up(): string { return this.cursorRow > 0 ? `${ESC}[${this.cursorRow}A` : ""; }
}
