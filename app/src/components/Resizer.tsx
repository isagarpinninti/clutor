import { useEffect, useRef } from "react";

const read = (key: string): number | undefined => { try { const v = Number(localStorage.getItem(key)); return Number.isFinite(v) && v > 0 ? v : undefined; } catch { return undefined; } };
const write = (key: string, v: number): void => { try { localStorage.setItem(key, String(Math.round(v))); } catch { /* the width just is not remembered */ } };
const clamp = (n: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, n));

/**
 * A drag handle on one edge of a panel. It sets a CSS variable on the page (so the panel and anything lined up with it follow), remembers the width, and also works from the keyboard
 * (arrow keys) and resets on a double click. `side` is the edge of the panel the handle sits on: "right" for a panel on the left of the window, "left" for one on the right.
 */
export function Resizer({ side, cssVar, storageKey, min, max, fallback, label }: { side: "left" | "right"; cssVar: string; storageKey: string; min: number; max: number; fallback: number; label: string }) {
  const width = useRef(fallback);
  const apply = (w: number): void => { width.current = clamp(w, min, max); document.documentElement.style.setProperty(cssVar, `${width.current}px`); };
  useEffect(() => { apply(read(storageKey) ?? fallback); }, []);
  const down = (e: React.PointerEvent<HTMLDivElement>): void => {
    e.preventDefault();
    const startX = e.clientX, start = width.current;
    const move = (ev: PointerEvent): void => apply(start + (side === "right" ? ev.clientX - startX : startX - ev.clientX));
    const up = (): void => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); document.body.classList.remove("resizing"); write(storageKey, width.current); };
    document.body.classList.add("resizing");
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  const key = (e: React.KeyboardEvent): void => {
    const d = e.key === "ArrowLeft" ? -16 : e.key === "ArrowRight" ? 16 : 0;
    if (!d) return;
    e.preventDefault();
    apply(width.current + (side === "right" ? d : -d));
    write(storageKey, width.current);
  };
  return <div className={`resizer ${side}`} role="separator" aria-orientation="vertical" aria-label={label} tabIndex={0} onPointerDown={down} onKeyDown={key} onDoubleClick={() => { apply(fallback); write(storageKey, fallback); }} />;
}
