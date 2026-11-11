import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadPolicy, PolicyError } from "../src/node/policy";

const dir = (text?: string): string => { const d = mkdtempSync(join(tmpdir(), "clutor-pol-")); if (text !== undefined) writeFileSync(join(d, "clutor.policy.json"), text); return d; };

describe("policy file", () => {
  it("is off when there is no file", () => expect(loadPolicy(dir(), {})).toEqual({ requireSandbox: false, forbidAutoApprove: false, source: null }));
  it("reads the two switches", () => {
    const p = loadPolicy(dir('{"requireSandbox":true,"forbidAutoApprove":true}'), {});
    expect(p.requireSandbox && p.forbidAutoApprove).toBe(true);
  });
  it("refuses a typo, a wrong type and bad JSON instead of ignoring them", () => {
    expect(() => loadPolicy(dir('{"requireSandbx":true}'), {})).toThrow(PolicyError);
    expect(() => loadPolicy(dir('{"requireSandbox":"yes"}'), {})).toThrow(/true or false/);
    expect(() => loadPolicy(dir("{"), {})).toThrow(PolicyError);
    expect(() => loadPolicy(dir("[]"), {})).toThrow(/object/);
  });
  it("can come from CLUTOR_POLICY, and a missing named file is an error", () => {
    const f = join(dir('{"requireSandbox":true}'), "clutor.policy.json");
    expect(loadPolicy(dir(), { CLUTOR_POLICY: f }).requireSandbox).toBe(true);
    expect(() => loadPolicy(dir(), { CLUTOR_POLICY: "/nonexistent/p.json" })).toThrow(/does not exist/);
  });
});
