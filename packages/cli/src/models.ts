import { createWriteStream } from "node:fs";
import { mkdir, readdir, rename, rm, stat, statfs } from "node:fs/promises";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { basename, extname, isAbsolute, join, resolve, sep } from "node:path";
import { CATALOG, QUANTS, approximateLayout, buildModelLayout, parseGguf, type GgufFile, type ModelLayout } from "@clutor/core";
import { fileReader } from "@clutor/core/node";

export interface ResolvedModel {
  id: string;
  name: string;
  source: "catalog" | "file";
  path?: string;
  layout: ModelLayout;
  gguf?: GgufFile;
  quant?: string;
}

export class ModelError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelError";
  }
}

/** `deepseek-v3`, `deepseek-v3@Q4_K_M`, or the path of a .gguf file. */
export async function resolveModel(ref: string, opts: { modelsDir?: string; allowFiles?: boolean } = {}): Promise<ResolvedModel> {
  const [idPart, quantPart] = ref.split("@");
  const spec = CATALOG.find((s) => s.id === idPart);
  if (spec && !ref.toLowerCase().endsWith(".gguf")) {
    const quant = quantPart ?? "Q4_K_M";
    const bpw = QUANTS[quant];
    if (bpw === undefined) throw new ModelError(`unknown quantisation "${quant}"; choose one of ${Object.keys(QUANTS).join(", ")}`);
    return { id: `${spec.id}@${quant}`, name: spec.name, source: "catalog", layout: approximateLayout(spec, bpw), quant };
  }
  if (!ref.toLowerCase().endsWith(".gguf")) throw new ModelError(`unknown model "${ref}"; use one of ${CATALOG.map((s) => s.id).join(", ")}, or the path of a .gguf file`);
  if (opts.allowFiles === false) throw new ModelError("reading a model file needs the access token");
  const path = isAbsolute(ref) ? ref : resolve(opts.modelsDir ?? ".", ref);
  try { await stat(path); } catch { throw new ModelError(`${path} does not exist`); }
  const r = await fileReader(path);
  try {
    const gguf = await parseGguf(r);
    const layout = buildModelLayout(gguf);
    const dominant = Object.entries(layout.quantisations).sort((a, b) => b[1] - a[1])[0]?.[0];
    return { id: basename(path), name: layout.name || basename(path), source: "file", path, layout, gguf, ...(dominant ? { quant: dominant } : {}) };
  } finally { await r.close(); }
}

export interface LocalModel { file: string; path: string; bytes: number }

export async function listLocalModels(dir: string): Promise<LocalModel[]> {
  const out: LocalModel[] = [];
  const walk = async (d: string, depth: number): Promise<void> => {
    let entries;
    try { entries = await readdir(d, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const p = join(d, e.name);
      if (e.isDirectory() && depth < 3) await walk(p, depth + 1);
      else if (e.isFile() && extname(e.name).toLowerCase() === ".gguf") out.push({ file: p.slice(resolve(dir).length + 1).split(sep).join("/"), path: p, bytes: (await stat(p)).size });
    }
  };
  await walk(resolve(dir), 0);
  return out.sort((a, b) => a.file.localeCompare(b.file));
}

/**
 * Real models small enough to download and test with. Sizes are the file sizes Hugging Face reported on 2026-10-30; the download
 * is checked against the size the server announces, and nothing is fetched unless you run `clutor pull`.
 */
export interface Download { id: string; catalogId: string; repo: string; file: string; bytes: number; note: string }
const GiB_ = 2 ** 30;
export const DOWNLOADS: Download[] = [
  { id: "olmoe-1b-7b-q4_k_m", catalogId: "olmoe-1b-7b", repo: "bartowski/OLMoE-1B-7B-0924-Instruct-GGUF", file: "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", bytes: Math.round(3.92 * GiB_), note: "a real mixture-of-experts model (64 experts, 8 used): the one to test expert streaming with" },
  { id: "qwen2.5-coder-7b-q4_k_m", catalogId: "qwen2.5-coder-7b", repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", file: "qwen2.5-coder-7b-instruct-q4_k_m.gguf", bytes: Math.round(4.36 * GiB_), note: "a dense coding model for the agent (not a mixture of experts)" },
];
export const downloadUrl = (d: Download): string => `https://huggingface.co/${d.repo}/resolve/main/${d.file}`;

export class DownloadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DownloadError";
  }
}

/** Downloads `url` to `dest` through a `.part` file, refusing to start without room for it and refusing a size the server did not announce. */
export async function downloadFile(url: string, dest: string, opts: { fetch?: typeof fetch; expectedBytes?: number; onProgress?: (done: number, total: number) => void } = {}): Promise<number> {
  const f = opts.fetch ?? globalThis.fetch;
  await mkdir(resolve(dest, ".."), { recursive: true });
  const res = await f(url, { redirect: "follow" });
  if (!res.ok || !res.body) throw new DownloadError(`the server answered ${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!total) throw new DownloadError("the server did not say how large the file is, so it cannot be checked");
  if (opts.expectedBytes && Math.abs(total - opts.expectedBytes) / opts.expectedBytes > 0.05) throw new DownloadError(`expected about ${opts.expectedBytes} bytes but the server offers ${total}; not downloading`);
  const fs = await statfs(resolve(dest, ".."));
  const free = Number(fs.bavail) * Number(fs.bsize);
  if (free < total * 1.1) throw new DownloadError(`not enough free disk: need ${(total / GiB_).toFixed(2)} GiB, have ${(free / GiB_).toFixed(2)} GiB`);
  const part = `${dest}.part`;
  let done = 0;
  const body = Readable.fromWeb(res.body as never);
  body.on("data", (c: Buffer) => { done += c.length; opts.onProgress?.(done, total); });
  try {
    await pipeline(body, createWriteStream(part));
    if (done !== total) throw new DownloadError(`the download stopped at ${done} of ${total} bytes`);
    await rename(part, dest);
  } catch (e) {
    await rm(part, { force: true });
    throw e;
  }
  return total;
}
