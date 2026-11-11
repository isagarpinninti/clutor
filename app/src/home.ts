import { memoryAdvice } from "./memory";
import type { Auto } from "./library";
import type { SystemInfo } from "./bridge";

/**
 * What the home screen says about this computer and the model it would use for each task. Everything is read from the file sizes and the memory reported by the system;
 * no speed is claimed here (the speed is on the model's own card, labelled estimated or measured).
 */
export type Fit = "memory" | "tight" | "streams" | "unknown";
export interface HomeRow { task: "Chat" | "Code" | "Vision" | "Image"; model: string | undefined; bytes: number; fit: Fit; reason: string }

const bytesOf = (c: Auto[keyof Auto]["choice"]): { name: string; bytes: number } | undefined => {
  if (!c) return undefined;
  if ("model" in c) return { name: c.model.name, bytes: c.model.bytes };
  if ("diffusion" in c) return { name: c.diffusion.name, bytes: c.diffusion.bytes };
  return { name: c.name, bytes: c.bytes };
};

export function homeRows(auto: Auto, system: SystemInfo | undefined, ctx: number): HomeRow[] {
  const row = (task: HomeRow["task"], a: Auto[keyof Auto]): HomeRow => {
    const b = bytesOf(a.choice);
    if (!b) return { task, model: undefined, bytes: 0, fit: "unknown", reason: a.reason };
    const name = b.name.replace(/\.(gguf|safetensors)$/i, "");
    if (!system) return { task, model: name, bytes: b.bytes, fit: "unknown", reason: a.reason };
    const advice = task === "Image" ? undefined : memoryAdvice(b.bytes, ctx, system.ramBytes, undefined);
    // the library already calls a model "tight" above 70% of memory (the operating system and the context need the rest); say the same here
    const fit: Fit = advice?.level === "over" ? "streams" : task !== "Image" && b.bytes > system.ramBytes * 0.7 ? "tight" : "memory";
    return { task, model: name, bytes: b.bytes, fit, reason: a.reason };
  };
  return [row("Chat", auto.chat), row("Code", auto.code), row("Vision", auto.vision), row("Image", auto.image)];
}

export const fitLabel = (f: Fit): string => (f === "memory" ? "Fits in memory" : f === "tight" ? "Tight in memory" : f === "streams" ? "Streams from SSD" : "Not checked");
