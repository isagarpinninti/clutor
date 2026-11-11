import { describe, expect, it } from "vitest";
import { fitLabel, homeRows } from "./home";
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
  it("calls a model tight above 70% of memory, as the library does", () => {
    expect(homeRows(auto(c("mid.gguf", 11.5 * GiB)), sys(16), 8192)[0]?.fit).toBe("tight");
  });
  it("says not checked when the system is unknown, and leaves a missing model empty", () => {
    const r = homeRows(auto(c("a.gguf", GiB)), undefined, 8192);
    expect(r[0]?.fit).toBe("unknown");
    expect(r[2]?.model).toBeUndefined();
    expect(fitLabel("unknown")).toBe("Not checked");
  });
});
