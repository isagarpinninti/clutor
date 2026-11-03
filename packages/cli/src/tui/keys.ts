/** Turns the bytes a terminal sends in raw mode into key events. Pure, so it can be tested without a terminal. */

export interface Key {
  /** "char" for typed text (in `text`), "paste" for a bracketed paste, else the key's name: up, down, left, right, home, end, enter, newline, tab, backtab, backspace, delete, escape, ctrl-a ... */
  name: string;
  text?: string;
  ctrl?: boolean;
  alt?: boolean;
}

const CSI: Record<string, string> = { A: "up", B: "down", C: "right", D: "left", H: "home", F: "end", Z: "backtab" };
const TILDE: Record<string, string> = { "1": "home", "3": "delete", "4": "end", "7": "home", "8": "end", "5": "pageup", "6": "pagedown" };

export class KeyParser {
  private pending = "";
  private paste: string | undefined;

  feed(chunk: string): Key[] {
    const keys: Key[] = [];
    let s = this.pending + chunk;
    this.pending = "";
    while (s.length > 0) {
      if (this.paste !== undefined) {
        const end = s.indexOf("\x1b[201~");
        if (end < 0) { this.paste += s; break; }
        keys.push({ name: "paste", text: (this.paste + s.slice(0, end)).replace(/\r\n?/g, "\n") });
        this.paste = undefined;
        s = s.slice(end + 6);
        continue;
      }
      if (s.startsWith("\x1b[200~")) { this.paste = ""; s = s.slice(6); continue; }
      const c = s[0] as string;
      if (c === "\x1b") {
        if (s.length === 1) { keys.push({ name: "escape" }); break; }
        const m = /^\x1b\[([0-9;]*)([A-Za-z~])/.exec(s);
        if (m) {
          s = s.slice(m[0].length);
          const params = m[1] as string, final = m[2] as string;
          const mod = Number((params.split(";")[1] ?? "1")) - 1;
          const key: Key = { name: "" };
          if (final === "~") key.name = TILDE[params.split(";")[0] as string] ?? "unknown";
          else if (final === "u") {
            const code = Number(params.split(";")[0]);
            key.name = code === 13 ? (mod & 1 || mod & 2 ? "newline" : "enter") : code === 27 ? "escape" : code === 9 ? "tab" : code === 127 ? "backspace" : "unknown";
          } else key.name = CSI[final] ?? "unknown";
          if (mod & 4) key.ctrl = true;
          if (mod & 2) key.alt = true;
          if (mod & 1 && key.name === "enter") key.name = "newline";
          keys.push(key);
          continue;
        }
        if (/^\x1b\[[0-9;]*$/.test(s)) { this.pending = s; break; }
        const o = /^\x1bO([A-Z])/.exec(s);
        if (o) { keys.push({ name: CSI[o[1] as string] ?? "unknown" }); s = s.slice(3); continue; }
        if (s[1] === "\r" || s[1] === "\n") { keys.push({ name: "newline" }); s = s.slice(2); continue; }
        if (s[1] === "\x7f") { keys.push({ name: "backspace", alt: true }); s = s.slice(2); continue; }
        if (s[1] === "b" || s[1] === "f" || s[1] === "d") { keys.push({ name: s[1] === "b" ? "left" : s[1] === "f" ? "right" : "delete", alt: true }); s = s.slice(2); continue; }
        keys.push({ name: "escape" });
        s = s.slice(1);
        continue;
      }
      s = s.slice(c.length);
      if (c === "\r" || c === "\n") keys.push({ name: "enter" });
      else if (c === "\t") keys.push({ name: "tab" });
      else if (c === "\x7f" || c === "\b") keys.push({ name: "backspace" });
      else if (c.charCodeAt(0) < 32) keys.push({ name: `ctrl-${String.fromCharCode(c.charCodeAt(0) + 96)}`, ctrl: true });
      else {
        // Collect a run of ordinary characters (a fast typist or a paste without brackets) into one event.
        let text = c;
        while (s.length > 0 && s[0] !== "\x1b" && s.charCodeAt(0) >= 32 && s[0] !== "\x7f") { text += s[0]; s = s.slice(1); }
        keys.push({ name: "char", text });
      }
    }
    return keys;
  }
}
