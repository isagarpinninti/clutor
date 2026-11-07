import { useEffect, useState } from "react";

const VERBS = ["Thinking", "Reasoning", "Working it out", "Putting it together", "Almost there"];

/** A small orb that breathes, with a ripple that fades outward. Calm on purpose: it says "working", not "loading". */
export function Spark({ size = 22 }: { size?: number }) {
  return <span className="orb" style={{ width: size, height: size }} aria-hidden="true"><i /><i /></span>;
}

/** The wait indicator: the asterisk, a shimmering word (cycling through a few, or the label you give it), and how long it has been. */
export function Thinking({ label, orb = true }: { label?: string; orb?: boolean }) {
  const [verb, setVerb] = useState(0);
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const clock = setInterval(() => setSecs(Math.floor((Date.now() - t0) / 1000)), 500);
    const words = label ? undefined : setInterval(() => setVerb((v) => (v + 1) % VERBS.length), 3400);
    return () => { clearInterval(clock); if (words) clearInterval(words); };
  }, [label]);
  return (
    <div className="thinking" role="status" aria-live="polite">
      {orb ? <Spark /> : null}
      <span className="shimmer" key={label ?? verb}>{label ?? VERBS[verb]}…</span>
      {secs >= 2 ? <span className="elapsed">{secs}s</span> : null}
    </div>
  );
}
