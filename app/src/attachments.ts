/**
 * Files attached to a chat message. The model reads text, so only text files are accepted (code, notes, logs, data); images, PDFs and other binary files are refused with a
 * reason instead of being sent as noise. What is attached is cut to fit the model's context, and the cut is shown.
 */

export interface Attachment { id: string; name: string; bytes: number; text: string; truncated: boolean }

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
export function withAttachments(text: string, files: Array<Pick<Attachment, "name" | "text" | "truncated">> | undefined): string {
  if (!files?.length) return text;
  const blocks = files.map((f) => `File: ${f.name}${f.truncated ? " (cut off to fit)" : ""}\n\`\`\`\n${f.text.replace(/```/g, "'''")}\n\`\`\``);
  return `${text}\n\n${blocks.join("\n\n")}`;
}
