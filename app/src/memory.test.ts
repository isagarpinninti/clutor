import { describe, expect, it } from "vitest";
import { memoryAdvice } from "./memory";

const GB = 2 ** 30;
describe("memoryAdvice", () => {
  it("is quiet when the model fits in the memory that is free", () => {
    expect(memoryAdvice(4 * GB, 8192, 16 * GB, 10 * GB)).toBeUndefined();
  });
  it("says memory is tight when other apps hold what the model needs", () => {
    const a = memoryAdvice(10.5 * GB, 8192, 16 * GB, 0.3 * GB);
    expect(a?.level).toBe("tight");
    expect(a?.text).toContain("0.3 GB");
  });
  it("says a model as big as the computer's memory will be slow", () => {
    const a = memoryAdvice(20 * GB, 8192, 16 * GB, 12 * GB);
    expect(a?.level).toBe("over");
    expect(a?.text).toContain("slow");
  });
  it("gives no advice when nothing is known", () => {
    expect(memoryAdvice(4 * GB, 8192, 0, undefined)).toBeUndefined();
    expect(memoryAdvice(0, 8192, 16 * GB, 1 * GB)).toBeUndefined();
    expect(memoryAdvice(10 * GB, 8192, 16 * GB, undefined)).toBeUndefined();
  });
});
