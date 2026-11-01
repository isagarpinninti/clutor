import { afterEach, describe, expect, it, vi } from "vitest";
import type * as Sessions from "../src/sessions";
import { cleanup, start, tmp, type Running } from "./helpers";

// The project roots are filled in asynchronously while the server starts. Slow that step down, so a request that arrives straight after
// the server begins listening is guaranteed to beat it (on a loaded CI runner it sometimes did, by chance).
vi.mock("../src/sessions", async (orig) => {
  const real = await orig<typeof Sessions>();
  return { ...real, normaliseRoot: async (r: string) => { await new Promise((done) => setTimeout(done, 250)); return real.normaliseRoot(r); } };
});

let s: Running | undefined;
afterEach(async () => { await s?.stop(); s = undefined; cleanup(); });

describe("a request that arrives before the project roots are ready", () => {
  it("still gets its decision signed, instead of an unsigned answer with receipt null", async () => {
    const root = tmp();
    s = await start({}, [{ content: "B", logprobs: { content: [{ token: "B", logprob: -0.35, top_logprobs: [{ token: "B", logprob: -0.35 }, { token: "A", logprob: -1.6 }, { token: "C", logprob: -3 }, { token: "D", logprob: -3 }] }] } }], { roots: [root] });
    const r = await s.post("/v1/decide", { prompt: "PR #42 changes auth with no tests", choices: ["approve", "request_changes", "block"] });
    expect(r.status).toBe(200);
    expect(r.body.receipt, "the decision was returned without a signed receipt").not.toBeNull();
    expect(r.body.receipt.tag).toMatch(/^clt_/);
  });
});
