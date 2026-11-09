import { useEffect, useState } from "react";
import { bridge } from "../bridge";
import type { Msg } from "../store";
import { Spark } from "./Thinking";
import { fmtElapsed } from "./RunProgress";

const cache = new Map<string, string>();

/** A picture the image model made: loaded from the app's pictures folder, with Save. */
export function GeneratedImage({ image }: { image: NonNullable<Msg["image"]> }) {
  const [src, setSrc] = useState<string | undefined>(cache.get(image.path) ? `data:image/png;base64,${cache.get(image.path)}` : undefined);
  const [error, setError] = useState<string>();
  useEffect(() => {
    if (src) return;
    let live = true;
    bridge.image.read(image.path).then((b) => { cache.set(image.path, b); if (live) setSrc(`data:image/png;base64,${b}`); }, (e: unknown) => { if (live) setError(e instanceof Error ? e.message : String(e)); });
    return () => { live = false; };
  }, [image.path, src]);
  return (
    <figure className="gen-image">
      {src ? <img src={src} alt={image.prompt} width={image.w} height={image.h} /> : <div className="gen-skel" style={{ aspectRatio: `${image.w} / ${image.h}` }}>{error ?? "Loading the picture…"}</div>}
      <figcaption>
        <span>{image.w} × {image.h}{image.seconds ? ` · made in ${fmtElapsed(image.seconds)}` : ""}</span>
        <button type="button" className="link-btn" onClick={() => void bridge.image.saveAs(image.path, "clutor-picture.png")}>Save as…</button>
      </figcaption>
    </figure>
  );
}

/** Where a picture is: loading the model, step n of N with the time left, or finishing. */
export function MakingImage({ making }: { making: NonNullable<Msg["making"]> }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(id); }, []);
  const elapsed = (now - making.startedAt) / 1000;
  const left = making.stage === "sampling" && making.secsPerStep ? (making.steps - making.step) * making.secsPerStep + 15 : undefined;
  const pct = making.stage === "finishing" ? 95 : making.stage === "sampling" ? Math.round((making.step / making.steps) * 90) : 3;
  return (
    <div className="making" role="status" aria-live="polite">
      <div className="making-head"><Spark size={18} /><b>{making.stage === "loading" ? "Loading the image model and reading your prompt…" : making.stage === "finishing" ? "Finishing the picture…" : `Step ${making.step} of ${making.steps}`}</b><small>{fmtElapsed(elapsed)}{left ? ` · about ${fmtElapsed(left)} left` : ""}</small></div>
      <span className="bar"><i style={{ width: `${pct}%` }} /></span>
      <p className="making-note">The chat model is unloaded while a picture is made, to leave the memory to the image model. It loads again when you chat.</p>
    </div>
  );
}
