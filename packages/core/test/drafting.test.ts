import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { PromptLookupIndex, evaluateDrafting, mulberry32, roughCodeTokens } from "../src";

describe("prompt-lookup drafting", () => {
  it("copies what followed the last occurrence of the current suffix", () => {
    const i = new PromptLookupIndex({ maxNgram: 3, minNgram: 2, maxDraft: 4 });
    i.push(1, 2, 3, 4, 5, 6, 9, 9, 2, 3);
    expect(i.draft()).toEqual([4, 5, 6, 9]);
  });
  it("prefers the longest matching suffix, then the most recent occurrence", () => {
    const i = new PromptLookupIndex({ maxNgram: 3, minNgram: 2, maxDraft: 2 });
    i.push(7, 1, 2, 3, 100, 101, 8, 1, 2, 3, 200, 201, 9, 1, 2, 3);
    expect(i.draft()).toEqual([200, 201]);
    const j = new PromptLookupIndex({ maxNgram: 3, minNgram: 2, maxDraft: 2 });
    j.push(5, 2, 3, 40, 41, 6, 1, 2, 3, 50, 51, 6, 2, 3);
    expect(j.draft()[0]).toBe(50); // the 2-gram "2,3" last continued with 50
  });
  it("proposes nothing when the suffix is new, and never more than maxDraft", () => {
    const i = new PromptLookupIndex({ maxDraft: 3 });
    i.push(1, 2, 3, 4, 5);
    expect(i.draft()).toEqual([]);
    i.push(1, 2, 3, 4);
    expect(i.draft().length).toBeLessThanOrEqual(3);
  });
  it("never drafts from the suffix itself, so it cannot predict a token that has not been seen", () => {
    fc.assert(fc.property(fc.array(fc.integer({ min: 0, max: 5 }), { minLength: 1, maxLength: 60 }), (xs) => {
      const i = new PromptLookupIndex();
      i.push(...xs);
      const d = i.draft();
      expect(d.length).toBeLessThanOrEqual(8);
      // whatever it proposes is something that really followed an earlier copy of the suffix
      if (d.length) {
        let found = false;
        for (let n = 2; n <= 4 && !found; n++) for (let s = 0; s + n < xs.length; s++) {
          if (xs.slice(s, s + n).join() === xs.slice(xs.length - n).join() && xs.slice(s + n, s + n + d.length).join() === d.join()) found = true;
        }
        expect(found).toBe(true);
      }
    }));
  });
  it("rejects nonsense options", () => {
    expect(() => new PromptLookupIndex({ minNgram: 0 })).toThrow(RangeError);
    expect(() => new PromptLookupIndex({ minNgram: 5, maxNgram: 3 })).toThrow(RangeError);
  });
});

describe("drafting evaluation", () => {
  const code = `
export function add(a: number, b: number): number { return a + b; }
export function sub(a: number, b: number): number { return a - b; }
export function mul(a: number, b: number): number { return a * b; }
export function div(a: number, b: number): number { return a / b; }
describe("math", () => {
  it("adds", () => { expect(add(1, 2)).toBe(3); });
  it("subtracts", () => { expect(sub(3, 2)).toBe(1); });
  it("multiplies", () => { expect(mul(2, 3)).toBe(6); });
  it("divides", () => { expect(div(6, 3)).toBe(2); });
});`.repeat(3);
  it("accepts several tokens per pass on repetitive code, and about one on random tokens", () => {
    const real = evaluateDrafting(roughCodeTokens(code));
    expect(real.tokensPerPass).toBeGreaterThan(2);
    expect(real.acceptanceRate).toBeGreaterThan(0.5);
    const r = mulberry32(9);
    const noise = Array.from({ length: 3000 }, () => Math.floor(r() * 5000));
    const rand = evaluateDrafting(noise);
    expect(rand.tokensPerPass).toBeLessThan(1.1);
  });
  it("keeps its books: every produced token came from exactly one pass, and accepted never exceeds drafted", () => {
    fc.assert(fc.property(fc.array(fc.integer({ min: 0, max: 8 }), { minLength: 20, maxLength: 300 }), (xs) => {
      const r = evaluateDrafting(xs, { warmup: 10 });
      expect(r.accepted).toBeLessThanOrEqual(r.drafted);
      expect(r.tokens).toBe(xs.length - 10);
      expect(r.passes).toBeLessThanOrEqual(r.tokens);
      expect(r.tokensPerPass).toBeGreaterThanOrEqual(1);
    }));
  });
  it("tokenises code losslessly into reusable ids", () => {
    const vocab = new Map<string, number>();
    const ids = roughCodeTokens("const x = foo(1);\nconst y = foo(2);", vocab);
    const back = new Map([...vocab].map(([k, v]) => [v, k]));
    expect(ids.map((i) => back.get(i)).join("")).toBe("const x = foo(1);\nconst y = foo(2);");
  });
});
