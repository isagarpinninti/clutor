/** Will this model fit in the memory this computer has to give right now? Checked before the model is opened, while the file's own pages are not yet counted as free. */

export interface MemoryAdvice { level: "tight" | "over"; text: string }

const GB = 2 ** 30;
const gb = (n: number): string => `${(n / GB).toFixed(n < 10 * GB ? 1 : 0)} GB`;

/**
 * `ctx` is the context length in tokens. The estimate is the model file, about half a gigabyte for the engine's buffers, and the attention cache at a generous 32 KB a token (at most 2 GB).
 * Returns nothing when there is room (or when nothing is known).
 */
export function memoryAdvice(modelBytes: number, ctx: number, totalBytes: number, availBytes: number | undefined): MemoryAdvice | undefined {
  if (!(modelBytes > 0) || !(totalBytes > 0)) return undefined;
  const need = modelBytes + GB / 2 + Math.min(ctx * 32768, 2 * GB);
  if (need > totalBytes * 0.9) {
    return { level: "over", text: `This model needs about ${gb(need)} and this computer has ${gb(totalBytes)}. It will read from disk while it answers, so expect it to be slow.` };
  }
  if (availBytes && availBytes > 0 && need > availBytes * 1.05) {
    return { level: "tight", text: `Only ${gb(availBytes)} of memory is free and this model needs about ${gb(need)}. Other apps are holding the rest, so answers will be slow until you close some.` };
  }
  return undefined;
}
