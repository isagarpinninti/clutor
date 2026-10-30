import { describe, expect, it } from "vitest";
import { banner, useColor, WORDMARK, WORDMARK_WIDTH } from "../src/banner";

const opts = { version: "9.9.9", folder: "/work/app", model: "none" };

describe("welcome banner", () => {
  it("draws CLUTOR in six aligned rows", () => {
    expect(WORDMARK).toHaveLength(6);
    expect(new Set(WORDMARK.map((r) => r.length)).size).toBe(1);
    expect(WORDMARK_WIDTH).toBe(WORDMARK[0]?.length);
  });
  it("shows the version, folder and model, without escape codes unless colour is on", () => {
    const plain = banner(opts);
    expect(plain).toContain(WORDMARK[0]);
    expect(plain).toMatch(/clutor interactive session v9\.9\.9/);
    expect(plain).toContain("/work/app");
    expect(plain).not.toContain("\x1b[");
    expect(banner({ ...opts, color: true })).toContain("\x1b[");
  });
  it("falls back to a one-line title on a narrow terminal", () => {
    const t = banner({ ...opts, columns: 40 });
    expect(t).not.toContain("██");
    expect(t).toContain("C L U T O R");
  });
  it("uses colour only on a terminal that allows it", () => {
    expect(useColor({}, true)).toBe(true);
    expect(useColor({}, false)).toBe(false);
    expect(useColor({ NO_COLOR: "1" }, true)).toBe(false);
    expect(useColor({ TERM: "dumb" }, true)).toBe(false);
  });
});
