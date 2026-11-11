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

/** The picker words its reason as "<model>: <why>"; the row already shows the model, so keep the why. */
export const whyOf = (reason: string): string => { const i = reason.indexOf(": "); return i >= 0 ? reason.slice(i + 2) : reason; };

export function homeRows(auto: Auto, system: SystemInfo | undefined, ctx: number): HomeRow[] {
  const row = (task: HomeRow["task"], a: Auto[keyof Auto]): HomeRow => {
    const b = bytesOf(a.choice);
    if (!b) return { task, model: undefined, bytes: 0, fit: "unknown", reason: whyOf(a.reason) };
    const name = b.name.replace(/\.(gguf|safetensors)$/i, "");
    if (!system) return { task, model: name, bytes: b.bytes, fit: "unknown", reason: whyOf(a.reason) };
    const reason = whyOf(a.reason);
    const advice = task === "Image" ? undefined : memoryAdvice(b.bytes, ctx, system.ramBytes, undefined);
    // the library already calls a model "tight" above 55% of memory (the operating system and the context need the rest); say the same here
    const fit: Fit = advice?.level === "over" ? "streams" : task !== "Image" && b.bytes > system.ramBytes * 0.55 ? "tight" : "memory";
    return { task, model: name, bytes: b.bytes, fit, reason };
  };
  return [row("Chat", auto.chat), row("Code", auto.code), row("Vision", auto.vision), row("Image", auto.image)];
}

export const fitLabel = (f: Fit): string => (f === "memory" ? "Fits in memory" : f === "tight" ? "Tight in memory" : f === "streams" ? "Streams from SSD" : "Not checked");

/** What the second first-run screen says: the machine, how many model files were found, which one Clutor would start with, and a plain note when that one must stream from the SSD. */
export interface FirstRun { machine: string; found: string; recommended?: { task: HomeRow["task"]; model: string; fit: Fit; note: string }; streamNote?: string }

export function firstRun(system: SystemInfo | undefined, rows: HomeRow[], files: number): FirstRun {
  const machine = system ? `${system.os} ${system.arch}, ${Math.round(system.ramBytes / 2 ** 30)} GB memory, ${system.cores} cores` : "this computer (details not available yet)";
  const found = files === 0 ? "No model files found yet." : `${files} model file${files === 1 ? "" : "s"} found.`;
  const chat = rows.find((r) => r.task === "Chat" && r.model);
  const out: FirstRun = { machine, found };
  if (chat) {
    out.recommended = { task: "Chat", model: chat.model as string, fit: chat.fit, note: chat.reason };
    if (chat.fit === "streams") out.streamNote = "This model is bigger than your memory. Clutor can still run it by reading the parts it needs from your SSD, but it will be much slower than a model that fits. The model card shows an estimate, and your own speed is measured once you chat.";
    else if (chat.fit === "tight") out.streamNote = "This model nearly fills your memory. It should run, but close other large apps first.";
  }
  return out;
}
