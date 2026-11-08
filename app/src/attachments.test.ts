import { describe, expect, it } from "vitest";
import { fitAttachments, readAttachment, withAttachments, type Attachment } from "./attachments";

const att = (name: string, n: number): Attachment => ({ id: name, name, bytes: n, text: "x".repeat(n), truncated: false });
const file = (parts: BlobPart[], name: string): File => new File(parts, name);

describe("readAttachment", () => {
  it("reads a text file", async () => {
    const r = await readAttachment(file(["hello\nworld"], "a.txt"), "1");
    expect(r).toMatchObject({ ok: true, file: { name: "a.txt", text: "hello\nworld" } });
  });
  it("refuses binary files, invalid UTF-8 and very large files with a reason", async () => {
    expect(await readAttachment(file([new Uint8Array([137, 80, 78, 71, 0, 1])], "p.png"), "1")).toMatchObject({ ok: false, reason: expect.stringMatching(/not a text file/) });
    expect(await readAttachment(file([new Uint8Array([0xff, 0xfe, 0x41])], "l.txt"), "1")).toMatchObject({ ok: false, reason: expect.stringMatching(/UTF-8/) });
    expect(await readAttachment(file([new Uint8Array(1_100_000).fill(65)], "big.txt"), "1")).toMatchObject({ ok: false, reason: expect.stringMatching(/larger/) });
  });
});

describe("fitAttachments", () => {
  it("leaves small attachments alone and cuts large ones to the context, saying so", () => {
    const small = [att("a", 100)];
    expect(fitAttachments(small, 4096)).toBe(small);
    const cut = fitAttachments([att("a", 20000), att("b", 20000)], 4096);
    expect(cut.every((f) => f.truncated)).toBe(true);
    expect(cut.reduce((n, f) => n + f.text.length, 0)).toBeLessThanOrEqual(6144);
  });
});

describe("withAttachments", () => {
  it("appends each file in a fenced block and neutralises fences inside the file", () => {
    const out = withAttachments("Summarise", [{ name: "n.md", text: "a ``` b", truncated: true }]);
    expect(out).toContain("File: n.md (cut off to fit)");
    expect(out.match(/```/g)).toHaveLength(2);
    expect(withAttachments("hi", undefined)).toBe("hi");
  });
});

describe("pictures", () => {
  it("are resized to multiples of 32 with at most 256 model tokens, keeping their shape", async () => {
    const { fitImage } = await import("./attachments");
    expect(fitImage(448, 448)).toEqual({ w: 448, h: 448 });
    const big = fitImage(4000, 3000);
    expect(big.w % 32 === 0 && big.h % 32 === 0 && big.w * big.h <= 256 * 1024 && big.w > big.h).toBe(true);
    expect(fitImage(10, 10)).toEqual({ w: 64, h: 64 });
  });
  it("are not sent to the model as text", async () => {
    const { withAttachments } = await import("./attachments");
    expect(withAttachments("what is this?", [{ name: "a.png", text: "", truncated: false, image: { w: 64, h: 64, thumb: "" } }])).toBe("what is this?");
  });
});
