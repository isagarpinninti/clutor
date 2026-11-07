import { useId } from "react";

/**
 * The Clutor mark says what Clutor is for: a model far bigger than the machine it runs on. The big faint rings are the whole model, larger than the tile that holds it and cut off by
 * its edges; the bright "C" with a spark at its opening is the part that is loaded and working right now, with the next token coming in.
 * The C is a solid stroke when nothing is happening. While the model thinks it breaks into eight segments, one for each expert a mixture-of-experts model can pick from, and they light
 * up one after another round the ring, which is what the engine is doing; while a reply streams the segments breathe together.
 */
export type LogoState = "idle" | "thinking" | "speaking";

const N = 8;
const R = 13.5;
const START = 135;   // degrees clockwise from twelve o'clock where the first segment sits: the ring runs round the bottom, left and top, and the gap is centred on the right
const SPAN = 270;

const pt = (deg: number, r = R): [number, number] => { const a = (deg * Math.PI) / 180; return [32 + r * Math.sin(a), 32 - r * Math.cos(a)]; };
const at = (deg: number): string => pt(deg).map((v) => v.toFixed(2)).join(" ");

function arc(i: number): string {
  const step = SPAN / (N - 1);
  const half = 11;   // each segment spans 22 degrees; with the round caps there is a hair of space between them
  return `M${at(START + i * step - half)}A${R} ${R} 0 0 1 ${at(START + i * step + half)}`;
}

const PATHS = Array.from({ length: N }, (_, i) => arc(i));
const FULL = `M${at(START)}A${R} ${R} 0 1 1 ${at(START + SPAN)}`;

export function Logo({ size = 24, state = "idle", tile = true }: { size?: number; state?: LogoState; tile?: boolean }) {
  const id = useId();
  const ink = tile ? "#fff" : `url(#${id}t)`;
  return (
    <svg className={`logo ${state}`} width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="Clutor">
      <defs>
        <linearGradient id={`${id}t`} x1="6" y1="4" x2="58" y2="60" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#4facfe" /><stop offset=".52" stopColor="#4d6bfe" /><stop offset="1" stopColor="#7c5cff" />
        </linearGradient>
        <linearGradient id={`${id}f`} x1="32" y1="47" x2="32" y2="17" gradientUnits="userSpaceOnUse"><stop offset="0" stopColor="#fff" stopOpacity=".34" /><stop offset="1" stopColor="#fff" /></linearGradient>
        <clipPath id={`${id}c`}><rect width="64" height="64" rx="18" /></clipPath>
      </defs>
      {tile ? <rect width="64" height="64" rx="18" fill={`url(#${id}t)`} /> : null}
      <g clipPath={`url(#${id}c)`} fill="none" stroke={ink}>
        <circle className="logo-model" cx="32" cy="32" r="36" strokeWidth="2.2" opacity=".4" />
        <circle className="logo-model" cx="32" cy="32" r="25.5" strokeWidth="1.8" opacity=".3" />
      </g>
      {state === "idle"
        ? <path d={FULL} fill="none" strokeWidth="5.6" strokeLinecap="round" stroke={tile ? `url(#${id}f)` : `url(#${id}t)`} />
        : (
          <g fill="none" strokeWidth="4.6" strokeLinecap="round" stroke={ink}>
            {PATHS.map((d, i) => <path key={i} d={d} className="logo-seg" style={{ ["--i" as string]: i, opacity: 0.38 + (0.62 * i) / (N - 1) }} />)}
          </g>
        )}
      <circle className="logo-spark" cx="45.6" cy="32" r="3" fill={ink} />
    </svg>
  );
}
