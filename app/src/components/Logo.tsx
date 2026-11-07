import { useId } from "react";

/**
 * The Clutor mark: a "C" that fades in from the bottom and ends at a spark, the token that comes in at its opening.
 * It is a solid stroke when nothing is happening. While the model thinks it breaks into eight segments, one for each expert a mixture-of-experts model can pick from, and they light
 * up one after another round the ring, which is what the engine is doing; while a reply streams the segments breathe together.
 */
export type LogoState = "idle" | "thinking" | "speaking";

const N = 8;
const R = 19;
const START = 135;   // degrees clockwise from twelve o'clock where the first segment sits: the ring runs round the bottom, left and top, and the gap is centred on the right
const SPAN = 270;

function arc(i: number): string {
  const step = SPAN / (N - 1);
  const half = 9;   // each segment spans 18 degrees; with the round caps they just do not touch
  const at = (deg: number): [number, number] => { const a = ((START + i * step + deg) * Math.PI) / 180; return [32 + R * Math.sin(a), 32 - R * Math.cos(a)]; };
  const [x1, y1] = at(-half), [x2, y2] = at(half);
  return `M${x1.toFixed(2)} ${y1.toFixed(2)}A${R} ${R} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

const PATHS = Array.from({ length: N }, (_, i) => arc(i));
const pt = (deg: number): string => { const a = (deg * Math.PI) / 180; return `${(32 + R * Math.sin(a)).toFixed(2)} ${(32 - R * Math.cos(a)).toFixed(2)}`; };
const FULL = `M${pt(START)}A${R} ${R} 0 1 1 ${pt(START + SPAN)}`;

export function Logo({ size = 24, state = "idle", tile = true }: { size?: number; state?: LogoState; tile?: boolean }) {
  const id = useId();
  return (
    <svg className={`logo ${state}`} width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="Clutor">
      <defs>
        <linearGradient id={`${id}t`} x1="6" y1="4" x2="58" y2="60" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#2dd4bf" /><stop offset=".55" stopColor="#5b7cfa" /><stop offset="1" stopColor="#9a6bff" />
        </linearGradient>
        <linearGradient id={`${id}f`} x1="32" y1="52" x2="32" y2="12" gradientUnits="userSpaceOnUse"><stop offset="0" stopColor="#fff" stopOpacity=".32" /><stop offset="1" stopColor="#fff" /></linearGradient>
      </defs>
      {tile ? <rect width="64" height="64" rx="18" fill={`url(#${id}t)`} /> : null}
      {state === "idle"
        ? <path d={FULL} fill="none" strokeWidth="6" strokeLinecap="round" stroke={tile ? `url(#${id}f)` : `url(#${id}t)`} />
        : (
          <g fill="none" strokeWidth="5" strokeLinecap="round" stroke={tile ? "#fff" : `url(#${id}t)`}>
            {PATHS.map((d, i) => <path key={i} d={d} className="logo-seg" style={{ ["--i" as string]: i, opacity: 0.38 + (0.62 * i) / (N - 1) }} />)}
          </g>
        )}
      <circle className="logo-spark" cx="50.5" cy="32" r="3.4" fill={tile ? "#fff" : `url(#${id}t)`} />
    </svg>
  );
}
