/**
 * Prompt-lookup drafting: guess the next tokens by finding the last few tokens earlier in the context and copying what followed.
 * Editing code repeats itself (a function signature, an import block, a test that mirrors the code it tests), so the guess is often
 * right, and a model can check several guessed tokens in one pass instead of generating them one by one.
 */

export interface DraftOptions {
  /** Longest suffix to look for (default 4). */
  maxNgram?: number;
  /** Shortest suffix worth trusting (default 2). */
  minNgram?: number;
  /** Most tokens to propose (default 8). */
  maxDraft?: number;
}

/** An incrementally built index of every n-gram seen so far, so lookup does not rescan the context. */
export class PromptLookupIndex {
  private readonly tokens: number[] = [];
  private readonly grams = new Map<string, number[]>();
  private readonly maxN: number;
  private readonly minN: number;
  private readonly maxDraft: number;

  constructor(opts: DraftOptions = {}) {
    this.maxN = opts.maxNgram ?? 4;
    this.minN = opts.minNgram ?? 2;
    this.maxDraft = opts.maxDraft ?? 8;
    if (this.minN < 1 || this.maxN < this.minN || this.maxDraft < 1) throw new RangeError("invalid drafting options");
  }

  get length(): number { return this.tokens.length; }

  push(...ts: number[]): void {
    for (const t of ts) {
      this.tokens.push(t);
      const end = this.tokens.length;
      // Index the n-grams that END at the previous token, keyed to the position of the token that followed them.
      for (let n = this.minN; n <= this.maxN; n++) {
        const start = end - 1 - n;
        if (start < 0) break;
        const k = this.tokens.slice(start, end - 1).join(",");
        const list = this.grams.get(k);
        if (list) list.push(end - 1);
        else this.grams.set(k, [end - 1]);
      }
    }
  }

  /** The tokens that most recently followed the longest matching suffix, up to `maxDraft`. Empty if nothing matches. */
  draft(): number[] {
    const len = this.tokens.length;
    for (let n = Math.min(this.maxN, len - 1); n >= this.minN; n--) {
      const k = this.tokens.slice(len - n).join(",");
      const list = this.grams.get(k);
      if (!list) continue;
      // The latest earlier occurrence; positions are of the token that followed it, and the suffix itself is not an "earlier" occurrence.
      for (let i = list.length - 1; i >= 0; i--) {
        const at = list[i] as number;
        if (at >= len) continue;
        const out = this.tokens.slice(at, Math.min(len, at + this.maxDraft));
        if (out.length) return out;
      }
    }
    return [];
  }
}

export interface DraftingReport {
  tokens: number;
  /** Model verification passes needed when each pass checks a draft and then adds one token of its own. */
  passes: number;
  drafted: number;
  accepted: number;
  /** Average tokens produced per pass: 1 means drafting never helped. */
  tokensPerPass: number;
  acceptanceRate: number;
}

/**
 * How well would drafting do on this token stream, if the model's output were exactly the stream? Each pass drafts from the text so
 * far, accepts the longest correct prefix, and then the model adds one token itself. This is an upper bound on a real model
 * (a real model may disagree with the text), measured on real text.
 */
export function evaluateDrafting(stream: number[], opts: DraftOptions & { warmup?: number } = {}): DraftingReport {
  const warm = Math.min(opts.warmup ?? 16, stream.length);
  const idx = new PromptLookupIndex(opts);
  idx.push(...stream.slice(0, warm));
  let pos = warm, passes = 0, drafted = 0, accepted = 0;
  while (pos < stream.length) {
    const d = idx.draft();
    let ok = 0;
    while (ok < d.length && pos + ok < stream.length && d[ok] === stream[pos + ok]) ok++;
    drafted += d.length;
    accepted += ok;
    const step = Math.min(ok + 1, stream.length - pos);
    idx.push(...stream.slice(pos, pos + step));
    pos += step;
    passes++;
  }
  const produced = stream.length - warm;
  return { tokens: produced, passes, drafted, accepted, tokensPerPass: passes ? produced / passes : 1, acceptanceRate: drafted ? accepted / drafted : 0 };
}

/**
 * A rough tokenizer for code: identifiers, numbers, whitespace runs and single punctuation marks. Real models use BPE, whose tokens
 * are similar in granularity but not identical, so figures from this are indicative only.
 */
export function roughCodeTokens(text: string, vocab: Map<string, number> = new Map()): number[] {
  const out: number[] = [];
  for (const m of text.matchAll(/[A-Za-z_][A-Za-z0-9_]*|\d+|\s+|[^\sA-Za-z0-9_]/g)) {
    const t = m[0];
    let id = vocab.get(t);
    if (id === undefined) { id = vocab.size; vocab.set(t, id); }
    out.push(id);
  }
  return out;
}
