import { type Style, charWidth, fit, width } from "./ansi";
import type { Key } from "./keys";

/** The prompt box: text with a cursor, history, multi-line input, and the slash-command menu. No I/O here; `view` says what to draw. */

export interface Cmd { name: string; args?: string; help: string }

export type Action =
  | { type: "submit"; text: string }
  | { type: "exit" }
  | { type: "clear-screen" }
  | { type: "notice"; text: string };

interface Seg { start: number; end: number }

/** Splits text into rows of at most `cols` columns, breaking at newlines and wherever a row fills up. */
export function layout(text: string, cols: number): Seg[] {
  const segs: Seg[] = [];
  let start = 0, w = 0, i = 0;
  while (i < text.length) {
    const cp = text.codePointAt(i) as number;
    const len = cp > 0xffff ? 2 : 1;
    if (cp === 10) { segs.push({ start, end: i }); i += 1; start = i; w = 0; continue; }
    const cw = charWidth(cp);
    if (w + cw > cols) { segs.push({ start, end: i }); start = i; w = 0; }
    w += cw; i += len;
  }
  segs.push({ start, end: text.length });
  return segs;
}

export class Editor {
  text = "";
  pos = 0;
  menuIdx = 0;
  showShortcuts = false;
  private hIdx = -1;
  private draft = "";
  private armedExit = false;

  constructor(readonly history: string[], private readonly cmds: Cmd[]) {}

  /** Commands matching what has been typed after the slash, best first; empty when the menu is not showing. */
  matches(): Cmd[] {
    if (!/^\/[^\s]*$/.test(this.text)) return [];
    const q = this.text.slice(1).toLowerCase();
    return [...this.cmds.filter((c) => c.name.startsWith(q)), ...this.cmds.filter((c) => !c.name.startsWith(q) && c.name.includes(q))];
  }

  private set(text: string, pos = text.length): void { this.text = text; this.pos = pos; this.menuIdx = 0; }
  private insert(s: string): void { this.set(this.text.slice(0, this.pos) + s + this.text.slice(this.pos), this.pos + s.length); this.hIdx = -1; this.showShortcuts = false; }
  private lineStart(p = this.pos): number { return this.text.lastIndexOf("\n", p - 1) + 1; }
  private lineEnd(p = this.pos): number { const e = this.text.indexOf("\n", p); return e < 0 ? this.text.length : e; }
  private wordLeft(): number { let p = this.pos; while (p > 0 && /\s/.test(this.text[p - 1] as string)) p--; while (p > 0 && !/\s/.test(this.text[p - 1] as string)) p--; return p; }
  private wordRight(): number { let p = this.pos; while (p < this.text.length && /\s/.test(this.text[p] as string)) p++; while (p < this.text.length && !/\s/.test(this.text[p] as string)) p++; return p; }

  handle(k: Key): Action | undefined {
    if (k.name !== "ctrl-c" && k.name !== "ctrl-d") this.armedExit = false;
    const menu = this.matches();
    switch (k.name) {
      case "char": if (this.text === "" && k.text === "?") { this.showShortcuts = !this.showShortcuts; return; } this.insert(k.text ?? ""); return;
      case "paste": this.insert(k.text ?? ""); return;
      case "enter": {
        if (this.text.endsWith("\\")) { this.set(`${this.text.slice(0, -1)}\n`); return; }
        if (menu.length > 0) {
          const pick = menu[Math.min(this.menuIdx, menu.length - 1)] as Cmd;
          if (this.text !== `/${pick.name}`) {
            if (pick.args) { this.set(`/${pick.name} `); return; }
            this.set(`/${pick.name}`);
          }
        }
        const text = this.text.trim();
        if (!text) return;
        if (this.history[this.history.length - 1] !== text) this.history.push(text);
        this.hIdx = -1;
        this.set("");
        return { type: "submit", text };
      }
      case "newline": this.insert("\n"); return;
      case "tab": case "backtab": {
        if (menu.length > 0) { const pick = menu[Math.min(this.menuIdx, menu.length - 1)] as Cmd; this.set(`/${pick.name}${pick.args ? " " : ""}`); }
        return;
      }
      case "backspace":
        if (k.alt) { const p = this.wordLeft(); this.set(this.text.slice(0, p) + this.text.slice(this.pos), p); }
        else if (this.pos > 0) { const prev = this.pos - ((this.text.codePointAt(this.pos - 2) ?? 0) > 0xffff ? 2 : 1); this.set(this.text.slice(0, prev) + this.text.slice(this.pos), prev); }
        return;
      case "delete": if (this.pos < this.text.length) this.set(this.text.slice(0, this.pos) + this.text.slice(this.pos + 1), this.pos); return;
      case "left": this.pos = k.alt || k.ctrl ? this.wordLeft() : Math.max(0, this.pos - 1); return;
      case "right": this.pos = k.alt || k.ctrl ? this.wordRight() : Math.min(this.text.length, this.pos + 1); return;
      case "home": case "ctrl-a": this.pos = this.lineStart(); return;
      case "end": case "ctrl-e": this.pos = this.lineEnd(); return;
      case "ctrl-u": this.set(this.text.slice(this.pos), 0); return;
      case "ctrl-k": this.set(this.text.slice(0, this.pos), this.pos); return;
      case "ctrl-w": { const p = this.wordLeft(); this.set(this.text.slice(0, p) + this.text.slice(this.pos), p); return; }
      case "ctrl-l": return { type: "clear-screen" };
      case "escape": if (this.showShortcuts) this.showShortcuts = false; else if (this.text) this.set(""); return;
      case "up":
        if (menu.length > 0) { this.menuIdx = (this.menuIdx + menu.length - 1) % menu.length; return; }
        if (this.text.includes("\n") && this.lineStart() > 0) { this.moveLine(-1); return; }
        this.recall(-1); return;
      case "down":
        if (menu.length > 0) { this.menuIdx = (this.menuIdx + 1) % menu.length; return; }
        if (this.text.includes("\n") && this.lineEnd() < this.text.length) { this.moveLine(1); return; }
        this.recall(1); return;
      case "ctrl-c":
        if (this.text) { this.set(""); return; }
        if (this.armedExit) return { type: "exit" };
        this.armedExit = true;
        return { type: "notice", text: "Press Ctrl-C again to exit" };
      case "ctrl-d":
        if (this.text === "") return { type: "exit" };
        if (this.pos < this.text.length) this.set(this.text.slice(0, this.pos) + this.text.slice(this.pos + 1), this.pos);
        return;
      default: return;
    }
  }

  private moveLine(dir: -1 | 1): void {
    const col = this.pos - this.lineStart();
    if (dir < 0) { const prevStart = this.lineStart(this.lineStart() - 1); this.pos = Math.min(prevStart + col, this.lineStart() - 1); }
    else { const next = this.lineEnd() + 1; this.pos = Math.min(next + col, this.lineEnd(next)); }
  }

  private recall(dir: -1 | 1): void {
    if (this.history.length === 0) return;
    if (this.hIdx === -1) { if (dir > 0) return; this.draft = this.text; this.hIdx = this.history.length; }
    this.hIdx += dir;
    if (this.hIdx >= this.history.length) { this.hIdx = -1; this.set(this.draft); return; }
    this.hIdx = Math.max(0, this.hIdx);
    this.set(this.history[this.hIdx] as string);
  }

  /**
   * What to draw: a rule, the text (the first row after `❯`), a rule, then either the command menu, the shortcut list, or the status line.
   * `left` and `right` are the status line's two ends.
   */
  view(st: Style, cols: number, status: { left: string; right: string }, notice?: string): { lines: string[]; cursor: { row: number; col: number } } {
    const inner = Math.max(10, cols - 3);
    const segs = layout(this.text, inner);
    const rule = st.dim("─".repeat(cols));
    const lines: string[] = [rule];
    segs.forEach((s, i) => lines.push(`${i === 0 ? st.accent("❯") : " "} ${this.text.slice(s.start, s.end)}`));
    lines.push(rule);
    let row = 1, col = 2;
    for (let i = 0; i < segs.length; i++) {
      const s = segs[i] as Seg;
      const last = i === segs.length - 1;
      if (this.pos >= s.start && (this.pos < s.end || (this.pos === s.end && (last || this.text[s.end] === "\n")))) {
        row = 1 + i; col = 2 + width(this.text.slice(s.start, this.pos)); break;
      }
    }
    const menu = this.matches();
    if (menu.length > 0) {
      const shown = menu.slice(Math.max(0, Math.min(this.menuIdx - 3, menu.length - 8)), Math.max(0, Math.min(this.menuIdx - 3, menu.length - 8)) + 8);
      const nameW = Math.max(...shown.map((c) => c.name.length + (c.args ? c.args.length + 1 : 0))) + 3;
      for (const c of shown) {
        const on = c === menu[Math.min(this.menuIdx, menu.length - 1)];
        const label = `/${c.name}${c.args ? ` ${c.args}` : ""}`.padEnd(nameW + 1);
        lines.push(fit(on ? `  ${st.accent(label)}${st.accent(c.help)}` : `  ${st.dim(label)}${st.dim(c.help)}`, cols));
      }
    } else if (this.showShortcuts) {
      for (const l of ["/ for commands       enter to send        \\ then enter for a new line", "esc to clear         ctrl-c to cancel     ctrl-d to exit (empty prompt)", "↑ ↓ for history     ctrl-a / ctrl-e line start / end      ctrl-w delete word", "@file to attach a file's text        while the model replies: esc to stop"]) lines.push(`  ${st.dim(l)}`);
    } else {
      const left = notice ? st.warn(notice) : st.dim(status.left);
      const gap = Math.max(1, cols - 2 - width(left) - width(status.right));
      lines.push(fit(`  ${left}${" ".repeat(gap)}${st.dim(status.right)}`, cols));
    }
    return { lines: lines.map((l) => fit(l, cols)), cursor: { row, col } };
  }
}
