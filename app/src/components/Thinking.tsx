import { useEffect, useState } from "react";

const VERBS = ["Thinking", "Reasoning", "Working it out", "Putting it together", "Almost there"];

/** A small asterisk whose eight rays pulse in turn while it slowly turns. */
export function Spark({ size = 22 }: { size?: number }) {
  return (
    <svg className="spark" width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <line x1="12" y1="3" x2="12" y2="21" /><line x1="3" y1="12" x2="21" y2="12" />
      <line x1="5.6" y1="5.6" x2="18.4" y2="18.4" /><line x1="18.4" y1="5.6" x2="5.6" y2="18.4" />
    </svg>
  );
}

/** The wait indicator: the asterisk, a shimmering word (cycling through a few, or the label you give it), and how long it has been. */
export function Thinking({ label }: { label?: string }) {
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
      <Spark />
      <span className="shimmer" key={label ?? verb}>{label ?? VERBS[verb]}…</span>
      {secs >= 2 ? <span className="elapsed">{secs}s</span> : null}
    </div>
  );
}
