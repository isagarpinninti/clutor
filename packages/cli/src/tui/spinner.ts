import type { Style } from "./ansi";

/** The animated status line shown while the model works: a pulsing glyph, a verb with a light sweeping across it, and live numbers. */

const FRAMES = ["·", "✢", "✳", "∗", "✻", "✽", "✻", "∗", "✳", "✢"];
const VERBS = ["Pondering", "Reasoning", "Mulling", "Weaving", "Brewing", "Musing", "Percolating", "Untangling", "Simmering", "Noodling", "Composing", "Distilling"];

export function verbFor(seed: number): string { return VERBS[Math.abs(seed) % VERBS.length] as string; }

export function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

/** One frame. `t` is milliseconds since the spinner started; `label` is the verb (or a fixed phrase such as "Loading model"). */
export function spinnerLine(st: Style, t: number, label: string, details: string[] = []): string {
  const glyph = st.accent(FRAMES[Math.floor(t / 110) % FRAMES.length] as string);
  const chars = [...label];
  const lit = (Math.floor(t / 70) % (chars.length + 8)) - 4;
  const word = chars.map((ch, i) => (Math.abs(i - lit) <= 1 ? st.hi(ch) : st.accent(ch))).join("");
  const tail = details.length ? ` ${st.dim(`(${details.join(" · ")})`)}` : "";
  return `${glyph} ${word}${st.accent("…")}${tail}`;
}
