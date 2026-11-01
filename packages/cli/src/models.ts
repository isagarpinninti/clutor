import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
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
 * Real models to download and test with. Sizes and SHA-256 hashes are the ones Hugging Face publishes for each file (queried 2026-10-31); a download is
 * checked against both, and nothing is fetched unless you run `clutor pull`. The OLMoE and Qwen3 files are the ones the native engine was validated on.
 */
const GiB_ = 2 ** 30;
export interface Download { id: string; catalogId: string; repo: string; file: string; bytes: number; sha256: string; note: string }
export const DOWNLOADS: Download[] = [
  { id: "olmoe-1b-7b-q4_k_m", catalogId: "olmoe-1b-7b", repo: "allenai/OLMoE-1B-7B-0924-Instruct-GGUF", file: "olmoe-1b-7b-0924-instruct-q4_k_m.gguf", bytes: 4213512672, sha256: "8c310f1435a1222338fd2d3d974975be9cd908180b644bab0c2a94da1ac32f3f", note: "a real mixture-of-experts model (64 experts, 8 used) that fits in memory: the one `clutor generate` was validated on" },
  { id: "granite-3.0-3b-a800m-q4_k_m", catalogId: "granite-3.0-3b-a800m", repo: "bartowski/granite-3.0-3b-a800m-instruct-GGUF", file: "granite-3.0-3b-a800m-instruct-Q4_K_M.gguf", bytes: 2059356000, sha256: "aa6287bafd182d06a6d11b0d001458deec902e8c0522875fc20cde2f561ebf6d", note: "IBM Granite MoE (40 experts, 8 used, 1.9 GiB): the third architecture the engine can run with `clutor generate`" },
  { id: "qwen3-30b-a3b-q4_k_m", catalogId: "qwen3-30b-a3b", repo: "Qwen/Qwen3-30B-A3B-GGUF", file: "Qwen3-30B-A3B-Q4_K_M.gguf", bytes: 18556685824, sha256: "0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5", note: "a mixture-of-experts model (128 experts, 8 used) LARGER than 16 GiB of RAM: run it with `clutor generate ... --cache-mb 4096`" },
  { id: "qwen2.5-1.5b-q4_k_m", catalogId: "qwen2.5-1.5b", repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF", file: "qwen2.5-1.5b-instruct-q4_k_m.gguf", bytes: 1117320736, sha256: "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e", note: "a small dense model (1.0 GiB): the fourth architecture the engine can run with `clutor generate`" },
  { id: "qwen2.5-coder-7b-q4_k_m", catalogId: "qwen2.5-coder-7b", repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", file: "qwen2.5-coder-7b-instruct-q4_k_m.gguf", bytes: 4683073536, sha256: "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c", note: "a dense coding model for the agent (not a mixture of experts)" },
];
export const downloadUrl = (d: Download): string => `https://huggingface.co/${d.repo}/resolve/main/${d.file}`;

export class DownloadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DownloadError";
  }
}

/**
 * Downloads `url` to `dest` through a `.part` file. An interrupted download keeps its `.part` and the next call resumes it with a Range request. It refuses
 * to start without room for the rest, refuses a size the server did not announce, and (with `expectedSha256`) deletes the file and fails unless the whole
 * file hashes to the published value.
 */
export async function downloadFile(url: string, dest: string, opts: { fetch?: typeof fetch; expectedBytes?: number; expectedSha256?: string; onProgress?: (done: number, total: number) => void } = {}): Promise<number> {
  const f = opts.fetch ?? globalThis.fetch;
  await mkdir(resolve(dest, ".."), { recursive: true });
  const part = `${dest}.part`;
  const have = await stat(part).then((s) => s.size).catch(() => 0);
  let res = await f(url, { redirect: "follow", ...(have > 0 ? { headers: { Range: `bytes=${have}-` } } : {}) });
  if (res.status === 416) { await rm(part, { force: true }); res = await f(url, { redirect: "follow" }); }   // the partial file does not fit the file on the server: start again
  if (!res.ok || !res.body) throw new DownloadError(`the server answered ${res.status} for ${url}`);
  const resumed = res.status === 206;
  let start = 0, total: number;
  if (resumed) {
    const m = /^bytes (\d+)-\d+\/(\d+)$/.exec(res.headers.get("content-range") ?? "");
    if (!m || Number(m[1]) !== have) throw new DownloadError("the server's answer to the resume request does not continue where the partial file stops");
    start = have; total = Number(m[2]);
  } else {
    total = Number(res.headers.get("content-length") ?? 0);
  }
  if (!total) throw new DownloadError("the server did not say how large the file is, so it cannot be checked");
  if (opts.expectedBytes && Math.abs(total - opts.expectedBytes) / opts.expectedBytes > 0.05) throw new DownloadError(`expected about ${opts.expectedBytes} bytes but the server offers ${total}; not downloading`);
  const fs = await statfs(resolve(dest, ".."));
  const free = Number(fs.bavail) * Number(fs.bsize);
  if (free < (total - start) * 1.1) throw new DownloadError(`not enough free disk: need ${((total - start) / GiB_).toFixed(2)} GiB, have ${(free / GiB_).toFixed(2)} GiB`);
  const hash = opts.expectedSha256 ? createHash("sha256") : undefined;
  if (hash && start > 0) for await (const chunk of createReadStream(part)) hash.update(chunk as Buffer);   // the bytes already on disk count towards the hash
  let done = start;
  const body = Readable.fromWeb(res.body as never);
  body.on("data", (c: Buffer) => { done += c.length; hash?.update(c); opts.onProgress?.(done, total); });
  try {
    await pipeline(body, createWriteStream(part, { flags: resumed ? "a" : "w" }));
  } catch (e) {
    throw new DownloadError(`the download was interrupted at ${done} of ${total} bytes (${e instanceof Error ? e.message : String(e)}); run the same command again to resume`);
  }
  if (done !== total) throw new DownloadError(`the download stopped at ${done} of ${total} bytes; run the same command again to resume`);
  if (hash && hash.digest("hex") !== opts.expectedSha256?.toLowerCase()) {
    await rm(part, { force: true });
    throw new DownloadError("the downloaded file does not match the SHA-256 the publisher lists; it was deleted");
  }
  await rename(part, dest);
  return total;
}
