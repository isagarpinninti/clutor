import { bitsPerWeight, metaNumber, metaString, parseGguf, type ByteReader, type GgufFile } from "../gguf/parse";

/**
 * A folder of model files, understood: what each file is (read from its header, not guessed from its name alone), which files belong together (a picture-reading model and its
 * projector; an image generator with its text encoder and VAE), and which model suits which task on a given machine. The user gives a folder; this decides the rest.
 */

export type FileKind = "chat-model" | "vision-model" | "projector" | "image-model" | "text-encoder" | "vae" | "other";

export interface Entry { path: string; name: string; bytes: number }

export interface Classified extends Entry {
  kind: FileKind;
  /** GGUF architecture, or the family of an image model ("qwen-image", "flux", ...). */
  arch?: string;
  moe?: boolean;
  /** Size in billions of parameters (from the file's own label when it has one, else estimated from its bytes). */
  paramsB?: number;
  quant?: string;
  /** A distilled few-step image model (fast, one pass per step). */
  turbo?: boolean;
  coder?: boolean;
  /** Why a file is "other", or anything the person should know. */
  note?: string;
}

/** The architectures Clutor's engine can run (c/src/forward.c). */
export const ENGINE_ARCHS = new Set(["olmoe", "qwen3moe", "qwen3vlmoe", "granitemoe", "qwen2", "llama", "qwen35"]);
const MOE_ARCHS = new Set(["olmoe", "qwen3moe", "qwen3vlmoe", "granitemoe"]);
const IMAGE_ARCHS = new Set(["flux", "flux2", "sd", "sd1", "sd2", "sdxl", "sd3", "qwen_image", "qwen-image", "wan", "hunyuan_video", "z_image"]);

const lower = (s: string): string => s.toLowerCase();
const isTurbo = (name: string): boolean => /turbo|lightning|schnell|lcm|distill|hyper|sdxs|flash/i.test(name);
const isCoder = (name: string): boolean => /coder|code(?!x?l)/i.test(name);

/** Billions of parameters from a label such as "30B-A3B", "7B", "1.5B" or a name; undefined when none is stated. */
export function paramsFromLabel(label: string | undefined): number | undefined {
  const m = label ? /(\d+(?:\.\d+)?)\s*([BM])\b/i.exec(label) : null;
  if (!m) return undefined;
  const v = Number(m[1]);
  return (m[2] as string).toUpperCase() === "B" ? v : v / 1000;
}

function dominantQuant(f: GgufFile): { name: string; bits: number } | undefined {
  const bytes = new Map<string, { n: number; type: number }>();
  for (const t of f.tensors) {
    if (t.dims.length < 2 || t.nbytes === undefined) continue;
    const cur = bytes.get(t.typeName) ?? { n: 0, type: t.type };
    cur.n += t.nbytes;
    bytes.set(t.typeName, cur);
  }
  let best: { name: string; n: number; type: number } | undefined;
  for (const [name, v] of bytes) if (!best || v.n > best.n) best = { name, ...v };
  const bits = best ? bitsPerWeight(best.type) : undefined;
  return best && bits ? { name: best.name, bits } : undefined;
}

function classifyGgufFile(e: Entry, f: GgufFile): Classified {
  const base: Classified = { ...e, kind: "other" };
  const arch = metaString(f.metadata, "general.architecture");
  const nm = `${e.name} ${metaString(f.metadata, "general.name") ?? ""}`;
  if (arch === "clip" || metaString(f.metadata, "general.type") === "mmproj") {
    return { ...base, kind: "projector", arch: metaString(f.metadata, "clip.projector_type") ?? "clip" };
  }
  const q = dominantQuant(f);
  const label = paramsFromLabel(metaString(f.metadata, "general.size_label")) ?? paramsFromLabel(nm);
  const params = label ?? (q ? Math.round(((e.bytes * 8) / q.bits / 1e9) * 10) / 10 : undefined);
  const common = { ...(q ? { quant: q.name } : {}), ...(params !== undefined ? { paramsB: params } : {}), ...(isTurbo(nm) ? { turbo: true } : {}), ...(isCoder(nm) ? { coder: true } : {}) };
  if (arch && ENGINE_ARCHS.has(arch)) {
    const experts = metaNumber(f.metadata, `${arch}.expert_count`) ?? 0;
    return { ...base, ...common, kind: arch === "qwen3vlmoe" ? "vision-model" : "chat-model", arch, ...(MOE_ARCHS.has(arch) && experts > 0 ? { moe: true } : {}) };
  }
  // An image generator: a ComfyUI-style file has no "general.architecture" at all, only tensors named after the network.
  const names = f.tensors.map((t) => t.name);
  const has = (s: string): boolean => names.some((n) => n.includes(s));
  if (!arch || IMAGE_ARCHS.has(arch)) {
    const family = arch && IMAGE_ARCHS.has(arch) ? arch.replace("_", "-")
      : has("time_text_embed") && has("img_mlp") ? "qwen-image"
      : has("double_blocks") && has("single_blocks") ? "flux"
      : has("input_blocks") && has("output_blocks") ? "stable-diffusion"
      : has("model.diffusion_model.") || has("transformer_blocks") ? "diffusion" : undefined;
    if (family) return { ...base, ...common, kind: "image-model", arch: family };
    if (!arch) return { ...base, note: "a GGUF file that states no architecture and does not look like an image model" };
  }
  return { ...base, ...common, arch, note: `the architecture "${arch}" is not supported by the engine yet` };
}

/** Reads a safetensors file's header: an 8-byte length, then JSON naming every tensor. */
export async function readSafetensorsNames(reader: ByteReader): Promise<string[]> {
  const len = new DataView((await reader.read(0, 8)).slice().buffer).getBigUint64(0, true);
  if (len <= 0n || len > 100_000_000n || Number(len) + 8 > reader.size) throw new Error("not a safetensors header");
  const json = JSON.parse(new TextDecoder().decode(await reader.read(8, Number(len)))) as Record<string, unknown>;
  return Object.keys(json).filter((k) => k !== "__metadata__");
}

function classifySafetensors(e: Entry, names: string[] | undefined): Classified {
  const base: Classified = { ...e, kind: "other" };
  const n = lower(e.name);
  const has = (s: string): boolean => !!names?.some((x) => x.includes(s));
  const quant = /int8|w4a8|nvfp4|fp8|bf16|fp16|mlx-\d+bit/.exec(n)?.[0];
  const q = quant ? { quant } : {};
  const vaeName = /(^|[_.-])vae([_.-]|$)/.test(n) && e.bytes < 4 * 2 ** 30;
  if (has("encoder.conv_in") || has("decoder.conv_in") || has("post_quant_conv") || (has("decoder.conv1") && has("encoder.conv1")) || (has("decoder.middle") && has("decoder.head")) || (vaeName && !has("transformer_blocks") && !has("self_attn.q_proj"))) return { ...base, ...q, kind: "vae" };
  if ((has("embed_tokens") && has(".self_attn.q_proj")) || (!names && /text_encoder|qwen3vl|qwen3\.5|t5|umt5|clip_l/.test(n))) {
    return { ...base, ...q, kind: "text-encoder", arch: /qwen3vl|qwen_3_vl/.test(n) ? "qwen3vl" : undefined, ...(/_pe_|prompt_enh/.test(n) ? { note: "a prompt-enhancing variant" } : {}) } as Classified;
  }
  if (has("transformer_blocks") || has("diffusion_model.") || has("double_blocks")) return { ...base, ...q, kind: "image-model", arch: has("time_text_embed") && has("img_mlp") ? "qwen-image" : "diffusion", ...(isTurbo(n) ? { turbo: true } : {}) };
  return { ...base, note: "a safetensors file the library does not recognise" };
}

/** Works out what one file is. Never throws: a file that cannot be read is "other" with the reason. */
export async function classifyFile(e: Entry, reader: ByteReader): Promise<Classified> {
  const n = lower(e.name);
  try {
    if (n.endsWith(".gguf")) return classifyGgufFile(e, await parseGguf(reader));
    if (n.endsWith(".safetensors")) return classifySafetensors(e, await readSafetensorsNames(reader).catch(() => undefined));
  } catch (err) {
    return { ...e, kind: "other", note: err instanceof Error ? err.message : String(err) };
  }
  return { ...e, kind: "other", note: "not a model file" };
}

export interface VisionModel { model: Classified; projector: Classified }
export interface ImagePipeline {
  diffusion: Classified;
  textEncoder?: Classified;
  vae?: Classified;
  complete: boolean;
  missing: string[];
  family: string;
}
export interface Library {
  chat: Classified[];
  vision: VisionModel[];
  image: ImagePipeline[];
  other: Classified[];
}

const dirOf = (p: string): string => p.replace(/[\\/][^\\/]*$/, "");
/** The part of a file name before its quantisation or variant: files of one model share it. */
const stem = (name: string): string => lower(name).replace(/\.(gguf|safetensors)$/, "").replace(/[.\-_](i1|imat)(?=[.\-_]|$)/g, "").split(/[.\-_](?=(?:q\d|iq\d|f16|bf16|mmproj|fp\d|int\d|nvfp4|mlx)\w*)/)[0] ?? "";
const sameFolder = (a: Entry, b: Entry): boolean => dirOf(a.path) === dirOf(b.path);
const nearby = (a: Entry, b: Entry): boolean => sameFolder(a, b) || dirOf(a.path).startsWith(dirOf(b.path)) || dirOf(b.path).startsWith(dirOf(a.path));

/** Groups classified files into what can be used: chat models, picture-reading models with their projector, and image generators with their text encoder and VAE. */
export function buildLibrary(files: Classified[]): Library {
  const projectors = files.filter((f) => f.kind === "projector");
  const lib: Library = { chat: [], vision: [], image: [], other: [] };
  for (const f of files) {
    if (f.kind === "chat-model") lib.chat.push(f);
    else if (f.kind === "vision-model") {
      const mine = projectors.filter((p) => sameFolder(p, f));
      const pick = mine.find((p) => stem(p.name) === stem(f.name)) ?? (mine.length === 1 ? mine[0] : undefined);
      if (pick) lib.vision.push({ model: f, projector: pick });
      lib.chat.push(f);   // a picture-reading model also chats (and without its projector, only chats)
    } else if (f.kind === "image-model") {
      const near = (k: FileKind): Classified[] => files.filter((c) => c.kind === k && nearby(c, f));
      const rank = (c: Classified): number => (c.quant === "int8" ? 0 : c.quant === "w4a8" ? 1 : c.quant === "fp8" ? 2 : 3) * 1e12 + c.bytes;   // the encoder variants that are tested first, then the smaller
      const te = near("text-encoder").filter((c) => !c.note).sort((a, b) => rank(a) - rank(b))[0];
      const vae = near("vae").sort((a, b) => a.bytes - b.bytes)[0];
      const missing = [...(te ? [] : ["text encoder"]), ...(vae ? [] : ["VAE"])];
      lib.image.push({ diffusion: f, ...(te ? { textEncoder: te } : {}), ...(vae ? { vae } : {}), complete: missing.length === 0, missing, family: f.arch ?? "diffusion" });
    } else if (f.kind === "other") lib.other.push(f);
  }
  return lib;
}

export type Task = "chat" | "code" | "vision" | "image";
export interface Machine {
  ramBytes: number;
  /** A model's measured generation speed on this computer, tokens per second, when it has been run. */
  measuredTps?: (name: string) => number | undefined;
}
export interface Choice<T> { choice: T | undefined; reason: string; ranked: Array<{ item: T; score: number; why: string }> }

const GiB = 2 ** 30;
function scoreLlm(m: Classified, task: Task, mach: Machine): { score: number; why: string } | undefined {
  const why: string[] = [];
  const ram = mach.ramBytes || 8 * GiB;
  const fitsResident = m.bytes <= ram * 0.7;
  const streams = !fitsResident && m.moe && m.bytes <= ram * 4;
  if (!fitsResident && !streams) return undefined;   // would not run at all
  let score = Math.log2(Math.max(0.5, m.paramsB ?? m.bytes / 0.6e9));
  why.push(`${m.paramsB ? `${m.paramsB}B` : "size unknown"}${m.moe ? " mixture-of-experts" : ""}`);
  if (m.moe) { score += 1.2; why.push("only a fraction of it is used per token"); }   // a 30B mixture-of-experts reads about what a 3B dense model does
  if (streams) { score -= 1.5; why.push("larger than memory: streams from disk"); }
  else if (m.bytes > ram * 0.55) { score -= 0.6; why.push("tight on memory"); }
  if (task === "code") { if (m.coder) { score += 1.5; why.push("tuned for code"); } }
  else if (m.coder) { score -= 0.4; why.push("a code model, used for chat"); }
  const tps = mach.measuredTps?.(m.name);
  if (tps !== undefined) { why.push(`${tps.toFixed(1)} tokens/s measured here`); if (tps < 1) score -= 4; else if (tps < 3) score -= 1; else score += 0.3; /* under one token a second is not a usable chat */ }
  return { score, why: why.join("; ") };
}

/** The model for a text task on this machine: the most capable one that runs comfortably. */
export function pickLlm(lib: Library, task: "chat" | "code", mach: Machine): Choice<Classified> {
  const ranked = lib.chat.flatMap((m) => { const s = scoreLlm(m, task, mach); return s ? [{ item: m, score: s.score, why: s.why }] : []; }).sort((a, b) => b.score - a.score);
  if (ranked.length) return { choice: ranked[0]!.item, reason: `${ranked[0]!.item.name}: ${ranked[0]!.why}`, ranked };
  const smallest = [...lib.chat].sort((a, b) => a.bytes - b.bytes)[0];
  return smallest ? { choice: smallest, reason: `${smallest.name}: the smallest model there is, and it may not fit this computer's memory`, ranked: [] } : { choice: undefined, reason: "no chat model in the folder", ranked: [] };
}

/** The model for a question about a picture: a picture-reading model with its projector. */
export function pickVision(lib: Library, mach: Machine): Choice<VisionModel> {
  const ranked = lib.vision.flatMap((v) => { const s = scoreLlm(v.model, "vision", mach); return s ? [{ item: v, score: s.score, why: s.why }] : []; }).sort((a, b) => b.score - a.score);
  if (ranked.length) return { choice: ranked[0]!.item, reason: `${ranked[0]!.item.model.name} with its projector: ${ranked[0]!.why}`, ranked };
  return { choice: undefined, reason: lib.vision.length ? "the picture-reading models here are too large for this computer" : "no picture-reading model (with its projector) in the folder", ranked: [] };
}

/** The image generator: one that has its text encoder and VAE; a few-step model on a computer with little memory, the full one when there is plenty. */
export function pickImage(lib: Library, mach: Machine): Choice<ImagePipeline> {
  const ram = mach.ramBytes || 8 * GiB;
  const ranked = lib.image.filter((p) => p.complete).map((p) => {
    let score = 0; const why: string[] = [];
    const weights = p.diffusion.bytes + (p.textEncoder?.bytes ?? 0) + (p.vae?.bytes ?? 0);
    if (p.diffusion.turbo) { score += ram >= 48 * GiB ? 0 : 2; why.push("few steps, one pass each (fast)"); } else if (ram >= 48 * GiB) { score += 1; why.push("full quality"); } else { why.push("many steps: slow on this computer"); }
    if (weights > ram * 0.9) { score -= 3; why.push("its files together are larger than memory"); }
    if (p.textEncoder?.quant === "int8") { score += 0.3; why.push("int8 text encoder"); }
    return { item: p, score, why: why.join("; ") };
  }).sort((a, b) => b.score - a.score);
  if (ranked.length) return { choice: ranked[0]!.item, reason: `${ranked[0]!.item.diffusion.name}: ${ranked[0]!.why}`, ranked };
  const partial = lib.image[0];
  return { choice: undefined, reason: partial ? `${partial.diffusion.name} is missing its ${partial.missing.join(" and ")}` : "no image model in the folder", ranked: [] };
}

/** A few lines saying what the folder holds, for the screen. */
export function describeLibrary(lib: Library): string[] {
  const out: string[] = [];
  if (lib.chat.length) out.push(`${lib.chat.length} chat model${lib.chat.length === 1 ? "" : "s"}`);
  if (lib.vision.length) out.push(`${lib.vision.length} that read${lib.vision.length === 1 ? "s" : ""} pictures`);
  const done = lib.image.filter((p) => p.complete).length;
  if (lib.image.length) out.push(`${done} image generator${done === 1 ? "" : "s"} ready${lib.image.length > done ? `, ${lib.image.length - done} missing parts` : ""}`);
  if (lib.other.length) out.push(`${lib.other.length} file${lib.other.length === 1 ? "" : "s"} not usable`);
  return out;
}

/** Classifies every file with a limit on how many are read at once. */
export async function scanFiles(entries: Entry[], readerFor: (e: Entry) => ByteReader, concurrency = 4): Promise<Classified[]> {
  const out: Classified[] = new Array(entries.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, entries.length) }, async () => {
    for (;;) { const i = next++; if (i >= entries.length) return; out[i] = await classifyFile(entries[i] as Entry, readerFor(entries[i] as Entry)); }
  }));
  return out;
}
