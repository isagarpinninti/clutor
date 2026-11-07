/**
 * The Clutor mark: a bold "C" and a dot, flat, in the theme's blue. The C is the part of a huge model that is loaded and working on a small machine; the dot is the next token
 * arriving at its opening.
 * It is a solid stroke when nothing is happening. While the model thinks the C breaks into eight segments, one for each expert a mixture-of-experts model can pick from, and they light
 * up one after another round the ring, which is what the engine is doing; while a reply streams the segments breathe together.
 */
export type LogoState = "idle" | "thinking" | "speaking";

const N = 8;
const R = 16;
const START = 135;   // degrees clockwise from twelve o'clock where the first segment sits: the ring runs round the bottom, left and top, and the gap is centred on the right
const SPAN = 270;

const pt = (deg: number): string => { const a = (deg * Math.PI) / 180; return `${(32 + R * Math.sin(a)).toFixed(2)} ${(32 - R * Math.cos(a)).toFixed(2)}`; };

function arc(i: number): string {
  const step = SPAN / (N - 1);
  const half = 6;   // each segment spans 12 degrees; with the round caps there is a hair of space between them
  return `M${pt(START + i * step - half)}A${R} ${R} 0 0 1 ${pt(START + i * step + half)}`;
}

const PATHS = Array.from({ length: N }, (_, i) => arc(i));
const FULL = `M${pt(START)}A${R} ${R} 0 1 1 ${pt(START + SPAN)}`;

/** `tile` puts it on a solid blue square with white strokes (the app icon); otherwise it is just the blue shape, so it sits on any background. */
export function Logo({ size = 24, state = "idle", tile = false }: { size?: number; state?: LogoState; tile?: boolean }) {
  return (
    <svg className={`logo ${state}${tile ? " tile" : ""}`} width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="Clutor">
      {tile ? <rect width="64" height="64" rx="16" fill="var(--brand, #3b6fe3)" /> : null}
      {state === "idle"
        ? <path d={FULL} fill="none" stroke="currentColor" strokeWidth="9" strokeLinecap="round" />
        : (
          <g fill="none" stroke="currentColor" strokeWidth="6.4" strokeLinecap="round">
            {PATHS.map((d, i) => <path key={i} d={d} className="logo-seg" style={{ ["--i" as string]: i, opacity: 0.38 + (0.62 * i) / (N - 1) }} />)}
          </g>
        )}
      <circle className="logo-spark" cx="51" cy="32" r="4.4" fill="currentColor" />
    </svg>
  );
}
