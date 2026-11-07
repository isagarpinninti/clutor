/**
 * The Clutor mark is a cluster of seven cells, the experts of a mixture-of-experts model. Five are lit, in the shape of a "C" open to the right: the experts that are loaded and working;
 * the dim one in the middle is the rest of the model, which stays on disk. It is how Clutor runs a model much bigger than the machine: only the cells it needs are lit.
 * The mark is still when nothing is happening. While the model thinks the cells light up one after another, which is the router choosing experts; while a reply streams they breathe together.
 */
export type LogoState = "idle" | "thinking" | "speaking";

const R = 9;       // a cell's radius
const D = 19.4;    // the distance from the middle to each outer cell
const hexPoints = (cx: number, cy: number): string =>
  Array.from({ length: 6 }, (_, k) => { const a = ((30 + 60 * k) * Math.PI) / 180; return `${(cx + R * Math.cos(a)).toFixed(2)},${(cy - R * Math.sin(a)).toFixed(2)}`; }).join(" ");

/** The five lit cells, going round from the top right; the gap on the right is the opening of the C. */
const CELLS = [60, 120, 180, 240, 300].map((deg) => hexPoints(32 + D * Math.cos((deg * Math.PI) / 180), 32 - D * Math.sin((deg * Math.PI) / 180)));
const CORE = hexPoints(32, 32);

/** `tile` puts it on a solid blue square with white cells (the app icon); otherwise it is just the blue cells, so it sits on any background. */
export function Logo({ size = 24, state = "idle", tile = false }: { size?: number; state?: LogoState; tile?: boolean }) {
  const scale = tile ? 0.74 : 1;
  return (
    <svg className={`logo ${state}${tile ? " tile" : ""}`} width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="Clutor">
      {tile ? <rect width="64" height="64" rx="16" fill="var(--brand, #3150e0)" /> : null}
      <g fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" transform={`translate(32 32) scale(${scale}) translate(-32 -32)`}>
        <polygon className="logo-core" points={CORE} />
        {CELLS.map((p, i) => <polygon key={i} className="logo-cell" points={p} style={{ ["--i" as string]: i }} />)}
      </g>
    </svg>
  );
}
