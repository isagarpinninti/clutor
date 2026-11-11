import { describe, expect, it } from "vitest";
import { firstRun, fitLabel, homeRows, whyOf } from "./home";
import type { Auto } from "./library";

const GiB = 2 ** 30;
const c = (name: string, bytes: number) => ({ name, bytes, path: `/m/${name}`, kind: "llm" }) as never;
const auto = (chat?: ReturnType<typeof c>): Auto => ({ chat: { choice: chat, reason: "r", ranked: [] }, code: { choice: chat, reason: "r", ranked: [] }, vision: { choice: undefined, reason: "none", ranked: [] }, image: { choice: undefined, reason: "none", ranked: [] } }) as Auto;
const sys = (ram: number) => ({ os: "macOS", arch: "arm64", ramBytes: ram * GiB, cores: 8, modelsDir: "/m" });

describe("home rows", () => {
  it("marks a model that fits and one that must stream", () => {
    expect(homeRows(auto(c("small.gguf", 2 * GiB)), sys(16), 8192)[0]).toMatchObject({ task: "Chat", model: "small", fit: "memory" });
    expect(homeRows(auto(c("big.gguf", 17 * GiB)), sys(16), 8192)[0]).toMatchObject({ fit: "streams" });
  });
  it("calls a model tight above 55% of memory, as the library does", () => {
    expect(homeRows(auto(c("mid.gguf", 9 * GiB)), sys(16), 8192)[0]?.fit).toBe("tight");
  });
  it("says not checked when the system is unknown, and leaves a missing model empty", () => {
    const r = homeRows(auto(c("a.gguf", GiB)), undefined, 8192);
    expect(r[0]?.fit).toBe("unknown");
    expect(r[2]?.model).toBeUndefined();
    expect(fitLabel("unknown")).toBe("Not checked");
  });
});

describe("first run", () => {
  it("names the machine, the files found, and warns when the model must stream", () => {
    const big = firstRun(sys(16), homeRows(auto(c("big.gguf", 17 * GiB)), sys(16), 8192), 3);
    expect(big.machine).toBe("macOS arm64, 16 GB memory, 8 cores");
    expect(big.found).toBe("3 model files found.");
    expect(big.recommended).toMatchObject({ model: "big", fit: "streams" });
    expect(big.streamNote).toMatch(/bigger than your memory/);
    expect(firstRun(sys(16), homeRows(auto(c("s.gguf", 2 * GiB)), sys(16), 8192), 1).streamNote).toBeUndefined();
  });
  it("says so when there is nothing yet, and does not invent a recommendation", () => {
    const r = firstRun(undefined, homeRows(auto(undefined), undefined, 8192), 0);
    expect(r.found).toBe("No model files found yet.");
    expect(r.recommended).toBeUndefined();
    expect(r.machine).toMatch(/not available/);
  });
});

describe("why", () => {
  it("keeps the reason and drops the model name the row already shows", () => {
    expect(whyOf("Qwen3-30B.gguf: 30B mixture-of-experts; tight on memory")).toBe("30B mixture-of-experts; tight on memory");
    expect(whyOf("no chat model in the folder")).toBe("no chat model in the folder");
  });
});
