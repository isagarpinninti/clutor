import { useId } from "react";

/** The Clutor mark: a white "C" with a token (a block of weights) streaming into its opening, on a blue tile. */
export function Logo({ size = 24 }: { size?: number }) {
  const id = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="Clutor">
      <defs>
        <linearGradient id={id} x1="8" y1="2" x2="56" y2="64" gradientUnits="userSpaceOnUse"><stop offset="0" stopColor="#4c8dff" /><stop offset="1" stopColor="#1d4ed8" /></linearGradient>
      </defs>
      <rect width="64" height="64" rx="16" fill={`url(#${id})`} />
      <path d="M41.5 22.4A15 15 0 1 0 41.5 41.6" fill="none" stroke="#fff" strokeWidth="7" strokeLinecap="round" />
      <rect x="44" y="28" width="8" height="8" rx="2" fill="#bfd6ff" />
    </svg>
  );
}
