import { type Style, width, wrap } from "./ansi";

/**
 * Renders a model's Markdown reply for a terminal while it is still arriving. Complete lines are rendered once and become permanent;
 * the line being typed is rendered on every update (inline styles only), so the reply appears word by word. Handled: headings,
 * bullets, numbered lists, quotes, rules, fenced code, tables, and inline code, bold, italic and links.
 */

export interface Rendered {
  /** Rows that are final: print them once. */
  done: string[];
  /** Rows for the unfinished line: redraw them on every update. */
  live: string[];
}

export class MarkdownStream {
  private buf = "";
  private inCode = false;
  private table: string[] = [];

  constructor(private readonly st: Style, private readonly cols: () => number) {}

  push(text: string): Rendered {
    this.buf += text;
    const done: string[] = [];
    let i: number;
    while ((i = this.buf.indexOf("\n")) >= 0) {
      const line = this.buf.slice(0, i).replace(/\r$/, "");
      this.buf = this.buf.slice(i + 1);
      done.push(...this.block(line));
    }
    return { done, live: this.liveRows() };
  }

  /** Call when the reply is over: renders what is left of the last line and any table still being collected. */
  end(): Rendered {
    const done: string[] = [];
    if (this.buf) { done.push(...this.block(this.buf.replace(/\r$/, ""))); this.buf = ""; }
    done.push(...this.flushTable());
    if (this.inCode) { this.inCode = false; done.push(this.st.dim("╰─")); }
    return { done, live: [] };
  }

  private liveRows(): string[] {
    if (!this.buf) return [];
    if (this.inCode) return wrap(this.st.dim("│ ") + this.st.code(this.buf), this.cols(), 2);
    if (this.buf.trimStart().startsWith("|")) return wrap(this.st.dim(this.buf), this.cols());
    return this.format(this.buf);
  }

  private block(line: string): string[] {
    const st = this.st;
    const fence = /^\s*```\s*([\w+#.-]*)/.exec(line);
    if (fence && (this.inCode ? /^\s*```\s*$/.test(line) : true)) {
      const out = this.flushTable();
      if (this.inCode) { this.inCode = false; out.push(st.dim("╰─")); }
      else { this.inCode = true; out.push(st.dim(`╭─${fence[1] ? ` ${fence[1]} ` : ""}`)); }
      return out;
    }
    if (this.inCode) return wrap(st.dim("│ ") + st.code(line), this.cols(), 2);
    if (line.trimStart().startsWith("|")) { this.table.push(line); return []; }
    const out = this.flushTable();
    out.push(...this.format(line));
    return out;
  }

  private format(line: string): string[] {
    const st = this.st, cols = this.cols();
    if (line.trim() === "") return [""];
    let m = /^(#{1,6})\s+(.*)$/.exec(line);
    if (m) {
      const text = this.inline(m[2] as string);
      return wrap(m[1] === "#" ? st.bold(st.underline(st.accent(text))) : st.bold(st.accent(text)), cols);
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) return [st.dim("─".repeat(Math.min(cols, 60)))];
    m = /^(\s*)[-*+]\s+(.*)$/.exec(line);
    if (m) { const ind = (m[1] as string).length; return wrap(`${" ".repeat(ind)}${st.accent("•")} ${this.inline(m[2] as string)}`, cols, ind + 2); }
    m = /^(\s*)(\d+)[.)]\s+(.*)$/.exec(line);
    if (m) {
      const ind = (m[1] as string).length, num = `${m[2]}.`;
      return wrap(`${" ".repeat(ind)}${st.accent(num)} ${this.inline(m[3] as string)}`, cols, ind + num.length + 1);
    }
    m = /^>\s?(.*)$/.exec(line);
    if (m) return wrap(`${st.dim("▎")} ${st.italic(this.inline(m[1] as string))}`, cols, 2);
    return wrap(this.inline(line), cols);
  }

  /** Inline styles. An unclosed marker is left as typed, so a half-arrived `**bold` does not flicker. */
  inline(s: string): string {
    const st = this.st;
    const parts = s.split(/(`[^`\n]+`)/g);
    return parts.map((p, i) => {
      if (i % 2 === 1) return st.code(p.slice(1, -1));
      return p
        .replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g, (_m, t: string, u: string) => `${st.underline(t)} ${st.dim(`(${u})`)}`)
        .replace(/\*\*([^*\n]+?)\*\*/g, (_m, t: string) => st.bold(t))
        .replace(/(?<![*\w])\*(?![\s*])([^*\n]+?)(?<![\s*])\*(?![*\w])/g, (_m, t: string) => st.italic(t))
        .replace(/(?<![_\w])_(?![\s_])([^_\n]+?)(?<![\s_])_(?![_\w])/g, (_m, t: string) => st.italic(t));
    }).join("");
  }

  private flushTable(): string[] {
    if (this.table.length === 0) return [];
    const st = this.st;
    const raw = this.table;
    this.table = [];
    const cells = (l: string): string[] => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    const isSep = (l: string): boolean => cells(l).every((c) => /^:?-{1,}:?$/.test(c));
    const rows = raw.filter((l) => !isSep(l)).map((l) => cells(l).map((c) => this.inline(c)));
    const n = Math.max(...rows.map((r) => r.length));
    const w = Array.from({ length: n }, (_, c) => Math.max(...rows.map((r) => width(r[c] ?? ""))));
    if (w.reduce((a, b) => a + b, 0) + 3 * (n - 1) + 2 > this.cols()) return raw.flatMap((l) => wrap(st.dim(l), this.cols()));
    const line = (r: string[], head: boolean): string => {
      const body = w.map((cw, c) => { const cell = r[c] ?? ""; return cell + " ".repeat(cw - width(cell)); }).join(st.dim(" │ "));
      return ` ${head ? st.bold(body) : body}`;
    };
    const out = [line(rows[0] as string[], true), st.dim(` ${w.map((cw) => "─".repeat(cw)).join("─┼─")}`)];
    for (const r of rows.slice(1)) out.push(line(r, false));
    return out;
  }
}
