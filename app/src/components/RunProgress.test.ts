import { describe, expect, it } from "vitest";
import { fmtElapsed } from "./RunProgress";

describe("fmtElapsed", () => {
  it("shows seconds, then minutes and seconds, then hours and minutes", () => {
    expect(fmtElapsed(0)).toBe("0s");
    expect(fmtElapsed(7.9)).toBe("7s");
    expect(fmtElapsed(59)).toBe("59s");
    expect(fmtElapsed(60)).toBe("1m 00s");
    expect(fmtElapsed(125)).toBe("2m 05s");
    expect(fmtElapsed(3725)).toBe("1h 02m");
    expect(fmtElapsed(-3)).toBe("0s");
  });
});
