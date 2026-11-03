import { describe, expect, it } from "vitest";
import type { StreamEvent } from "@engine/stream";
import { JobQueue, etaMs, recover, type Job, type JobSession, type JobStore } from "./jobs";

const memory = (initial: Job[] = []): JobStore & { saved: Job[] } => {
  const s = { saved: initial, load: () => s.saved, save: (j: Job[]) => { s.saved = JSON.parse(JSON.stringify(j)) as Job[]; } };
  return s;
};
const tick = (ms = 5): Promise<void> => new Promise((r) => setTimeout(r, ms));
async function until(f: () => boolean, ms = 2000): Promise<void> { const t = Date.now(); while (!f()) { if (Date.now() - t > ms) throw new Error("timed out"); await tick(); } }

/** A session that writes the given words one by one, honours cancel, and records what it was asked. */
function fake(words: string[], opts: { delay?: number; failAt?: number } = {}) {
  const calls: Array<{ messages: unknown; prefill?: string; max?: number }> = [];
  const session: JobSession = {
    async *stream(messages, o): AsyncGenerator<StreamEvent> {
      calls.push({ messages, ...(o?.prefill !== undefined ? { prefill: o.prefill } : {}), ...(o?.max !== undefined ? { max: o.max } : {}) });
      let i = 0, seen = 0;
      const skip = (o?.prefill ?? "").length;   // a real engine continues after the text it was given
      for (const w of words) {
        seen += w.length;
        if (seen <= skip) { i++; continue; }
        if (o?.signal?.aborted) { yield { type: "done", finish: "cancel" }; return; }
        if (opts.failAt === i) throw new Error("the engine stopped");
        await tick(opts.delay ?? 2);
        yield { type: "content", text: w };
        i++;
      }
      yield { type: "done", finish: "stop", completionTokens: words.length, tokensPerSecond: 7 };
    },
  };
  return { session, calls };
}

const base = (n: string) => ({ id: n, title: n, conversationId: "c", messageId: `m${n}`, messages: [{ role: "user" as const, content: n }], thinking: false, temperature: 0.7, maxTokens: 100 });

describe("JobQueue", () => {
  it("runs jobs one at a time, in the order they were queued, and reports each as done", async () => {
    const f = fake(["a ", "b ", "c"]);
    const finished: string[] = [];
    const q = new JobQueue(memory(), { onFinish: (j) => finished.push(j.id) });
    q.enqueue(base("1")); q.enqueue(base("2"));
    expect(q.position("1")).toBe(1); expect(q.position("2")).toBe(2);
    q.attach(f.session);
    await until(() => q.list().every((j) => j.status === "done"));
    expect(finished).toEqual(["1", "2"]);
    expect(q.get("1")?.content).toBe("a b c");
    expect(q.get("1")?.tokens).toBe(3);
    expect(q.get("1")?.tps).toBe(7);
  });

  it("waits for a model: nothing runs until an engine is attached", async () => {
    const q = new JobQueue(memory());
    q.enqueue(base("1"));
    await tick(30);
    expect(q.get("1")?.status).toBe("queued");
    q.attach(fake(["x"]).session);
    await until(() => q.get("1")?.status === "done");
  });

  it("pauses on request, keeping the partial reply, and resumes after the text it wrote", async () => {
    const f = fake(["one ", "two ", "three ", "four ", "five"], { delay: 25 });
    const q = new JobQueue(memory());
    q.attach(f.session);
    q.enqueue(base("1"));
    await until(() => (q.get("1")?.content.length ?? 0) >= 8);
    q.pause("1");
    await until(() => q.get("1")?.status === "paused");
    const partial = q.get("1")?.content ?? "";
    expect(partial.length).toBeGreaterThan(0);
    expect(q.get("1")?.pausedBy).toBe("user");
    q.resume("1");
    await until(() => q.get("1")?.status === "done");
    expect(q.get("1")?.content).toBe("one two three four five");     // the whole reply, with nothing repeated
    expect(f.calls[1]?.prefill).toBe(partial);                       // the second run was told what had been written
    expect(f.calls[1]?.max).toBeLessThan(100);                       // and only gets the rest of the budget
  });

  it("a chat message takes the engine: the job pauses, then continues by itself afterwards", async () => {
    const f = fake(["a ", "b ", "c ", "d ", "e ", "f"], { delay: 25 });
    const q = new JobQueue(memory());
    q.attach(f.session);
    q.enqueue(base("1"));
    await until(() => (q.get("1")?.content.length ?? 0) >= 4);
    const release = await q.holdForChat();
    expect(q.running).toBeUndefined();                                // the engine is free for the chat
    expect(["paused", "queued"]).toContain(q.get("1")?.status);
    await tick(60);
    expect(q.get("1")?.status).not.toBe("running");                  // held while the chat has it
    release();
    await until(() => q.get("1")?.status === "done");
    expect(q.get("1")?.content).toBe("a b c d e f");
  });

  it("cancels a running job and a waiting one", async () => {
    const f = fake(["a ", "b ", "c ", "d"], { delay: 30 });
    const q = new JobQueue(memory());
    q.attach(f.session);
    q.enqueue(base("1")); q.enqueue(base("2"));
    await until(() => q.get("1")?.status === "running");
    q.cancel("2");
    expect(q.get("2")?.status).toBe("cancelled");
    q.cancel("1");
    await until(() => q.get("1")?.status === "cancelled");
    expect(q.get("1")?.finished).toBeDefined();
  });

  it("marks a job failed with the reason, and retry runs it again from the start", async () => {
    let fail = true;
    const session: JobSession = {
      async *stream(): AsyncGenerator<StreamEvent> { if (fail) throw new Error("context is full"); yield { type: "content", text: "ok" }; yield { type: "done", finish: "stop", completionTokens: 1 }; },
    };
    const q = new JobQueue(memory());
    q.attach(session);
    q.enqueue(base("1"));
    await until(() => q.get("1")?.status === "failed");
    expect(q.get("1")?.error).toBe("context is full");
    fail = false;
    q.retry("1");
    await until(() => q.get("1")?.status === "done");
    expect(q.get("1")?.content).toBe("ok");
    expect(q.get("1")?.error).toBeUndefined();
  });

  it("survives a restart: a job that was running comes back paused with its partial reply, a queued one is still queued", () => {
    const saved: Job[] = [
      { ...base("1"), status: "running", content: "half an ans", reasoning: "", tokens: 4, elapsedMs: 9000, created: 1 },
      { ...base("2"), status: "queued", content: "", reasoning: "", tokens: 0, elapsedMs: 0, created: 2 },
      { ...base("3"), status: "done", content: "fin", reasoning: "", tokens: 1, elapsedMs: 10, created: 3 },
    ];
    const r = recover(saved);
    expect(r[0]).toMatchObject({ status: "paused", pausedBy: "interrupted", content: "half an ans" });
    expect(r[1]?.status).toBe("queued");
    expect(r[2]?.status).toBe("done");
    const q = new JobQueue(memory(saved));
    expect(q.get("1")?.status).toBe("paused");
  });

  it("saves its state as it changes, so nothing is lost if the app is killed", async () => {
    const store = memory();
    const q = new JobQueue(store);
    q.attach(fake(["a ", "b"]).session);
    q.enqueue(base("1"));
    await until(() => q.get("1")?.status === "done");
    expect(store.saved.find((j) => j.id === "1")?.status).toBe("done");
    expect(store.saved.find((j) => j.id === "1")?.content).toBe("a b");
  });

  it("losing the model under a running job pauses it as interrupted instead of failing it", async () => {
    const f = fake(["a ", "b ", "c ", "d ", "e"], { delay: 30 });
    const q = new JobQueue(memory());
    q.attach(f.session);
    q.enqueue(base("1"));
    await until(() => q.get("1")?.status === "running");
    q.attach(null);
    await until(() => q.get("1")?.status === "paused");
    expect(q.get("1")?.pausedBy).toBe("interrupted");
  });

  it("estimates the time left as an upper bound from the measured speed", () => {
    const j: Job = { ...base("1"), status: "running", content: "", reasoning: "", tokens: 20, elapsedMs: 1, created: 1, tps: 4 };
    expect(etaMs(j)).toBe(((100 - 20) / 4) * 1000);
    expect(etaMs({ ...j, tps: undefined })).toBeUndefined();
    expect(etaMs({ ...j, status: "queued" })).toBeUndefined();
  });

  it("removing a finished job forgets it", async () => {
    const q = new JobQueue(memory());
    q.attach(fake(["x"]).session);
    q.enqueue(base("1"));
    await until(() => q.get("1")?.status === "done");
    q.remove("1");
    expect(q.list()).toHaveLength(0);
  });
});

import { shouldPin } from "./store";
describe("shouldPin", () => {
  it("pins models that are comfortable for the machine and skips large ones", () => {
    expect(shouldPin(true, 5.6e9, 16e9)).toBe(true);
    expect(shouldPin(true, 8e9, 16e9)).toBe(false);
    expect(shouldPin(false, 1e9, 16e9)).toBe(false);
    expect(shouldPin(true, 5e9, undefined)).toBe(true);
  });
});
