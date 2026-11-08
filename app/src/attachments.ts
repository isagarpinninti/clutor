/**
 * Files attached to a chat message. The model reads text, so only text files are accepted (code, notes, logs, data); images, PDFs and other binary files are refused with a
 * reason instead of being sent as noise. What is attached is cut to fit the model's context, and the cut is shown.
 */

/** A picture is an attachment too: `text` is empty, `image` says how big it is and carries a small preview; its pixels are kept in memory (see imageRgb), not in the saved chat. */
export interface Attachment { id: string; name: string; bytes: number; text: string; truncated: boolean; image?: { w: number; h: number; thumb: string } }

const PIXELS = new Map<string, string>();
/** The base64 of a picture's 8-bit RGB pixels, while the app has been open since it was attached (undefined after a restart). */
export const imageRgb = (id: string): string | undefined => PIXELS.get(id);

/** The size a picture is resized to: both sides multiples of 32, at most `maxTokens` model tokens (one per 32 x 32 pixels), the same rule as the engine's. */
export function fitImage(w: number, h: number, maxTokens = 256): { w: number; h: number } {
  const cap = maxTokens * 1024;
  const s = w * h > cap ? Math.sqrt(cap / (w * h)) : 1;
  let nw = Math.max(64, Math.round((w * s) / 32) * 32), nh = Math.max(64, Math.round((h * s) / 32) * 32);
  while (nw * nh > cap && (nw > 64 || nh > 64)) { if (nw >= nh && nw > 64) nw -= 32; else if (nh > 64) nh -= 32; else break; }
  return { w: nw, h: nh };
}

/** Reads a picture file for a model that takes pictures: decoded in the page, resized, kept as RGB. */
export async function readImage(file: File, id: string): Promise<Read> {
  if (file.size > 20_000_000) return { ok: false, name: file.name, reason: "larger than 20 MB" };
  let bmp: ImageBitmap;
  try { bmp = await createImageBitmap(file); } catch { return { ok: false, name: file.name, reason: "not a picture the app can open (PNG, JPEG, GIF, WebP)" }; }
  const { w, h } = fitImage(bmp.width, bmp.height);
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const cx = cv.getContext("2d", { willReadFrequently: true });
  if (!cx) return { ok: false, name: file.name, reason: "no drawing surface" };
  cx.fillStyle = "#fff"; cx.fillRect(0, 0, w, h);   // transparent areas become white
  cx.imageSmoothingQuality = "high";
  cx.drawImage(bmp, 0, 0, w, h);
  const px = cx.getImageData(0, 0, w, h).data;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < px.length; i += 4) { rgb[j++] = px[i] as number; rgb[j++] = px[i + 1] as number; rgb[j++] = px[i + 2] as number; }
  let bin = "";
  for (let i = 0; i < rgb.length; i += 0x8000) bin += String.fromCharCode(...rgb.subarray(i, i + 0x8000));
  PIXELS.set(id, btoa(bin));
  const t = document.createElement("canvas"); const k = 72 / Math.max(w, h);
  t.width = Math.round(w * k); t.height = Math.round(h * k);
  t.getContext("2d")?.drawImage(cv, 0, 0, t.width, t.height);
  return { ok: true, file: { id, name: file.name, bytes: file.size, text: "", truncated: false, image: { w, h, thumb: t.toDataURL("image/jpeg", 0.7) } } };
}

export const MAX_FILE_BYTES = 1_000_000;
export const MAX_FILES = 5;

export type Read = { ok: true; file: Attachment } | { ok: false; name: string; reason: string };

const bytesOf = (file: File): Promise<ArrayBuffer> => typeof file.arrayBuffer === "function" ? file.arrayBuffer() : new Promise((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(r.result as ArrayBuffer); r.onerror = () => reject(r.error); r.readAsArrayBuffer(file); });

export async function readAttachment(file: File, id: string): Promise<Read> {
  if (file.size > MAX_FILE_BYTES) return { ok: false, name: file.name, reason: `larger than ${Math.round(MAX_FILE_BYTES / 1e6)} MB` };
  const buf = new Uint8Array(await bytesOf(file));
  if (buf.subarray(0, 8000).includes(0)) return { ok: false, name: file.name, reason: "not a text file (images, PDFs and other binary files cannot be read by the model)" };
  let text: string;
  try { text = new TextDecoder("utf-8", { fatal: true }).decode(buf); } catch { return { ok: false, name: file.name, reason: "not UTF-8 text" }; }
  return { ok: true, file: { id, name: file.name, bytes: file.size, text, truncated: false } };
}

/** Cuts the files so together they take at most about half of the model's context (three characters a token is a cautious guess), the longest files first. */
export function fitAttachments(files: Attachment[], contextTokens: number): Attachment[] {
  const budget = Math.max(1500, Math.floor(contextTokens * 1.5));
  const total = files.reduce((n, f) => n + f.text.length, 0);
  if (total <= budget) return files;
  const share = Math.max(500, Math.floor(budget / files.length));
  return files.map((f) => (f.text.length > share ? { ...f, text: f.text.slice(0, share), truncated: true } : f));
}

/** The message as the model sees it: your words, then each file in a fenced block. */
export function withAttachments(text: string, files: Array<Pick<Attachment, "name" | "text" | "truncated"> & { image?: unknown }> | undefined): string {
  if (!files?.length) return text;
  files = files.filter((f) => !("image" in f && f.image));
  if (!files.length) return text;
  const blocks = files.map((f) => `File: ${f.name}${f.truncated ? " (cut off to fit)" : ""}\n\`\`\`\n${f.text.replace(/```/g, "'''")}\n\`\`\``);
  return `${text}\n\n${blocks.join("\n\n")}`;
}
