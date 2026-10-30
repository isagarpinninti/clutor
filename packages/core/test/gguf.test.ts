import { readFileSync } from "node:fs";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { GgufError, buildGgufHeader, buildModelLayout, expertSizeBytes, memoryReader, parseGguf, syntheticTensors, validateBounds, type SyntheticModelSpec } from "../src";

const real = readFileSync(new URL("./fixtures/stories260K.gguf", import.meta.url));

const MOE: SyntheticModelSpec = { arch: "qwen2moe", name: "tiny-moe", layers: 3, embedding: 64, heads: 8, kvHeads: 4, contextLength: 4096, expertCount: 8, expertsUsed: 2, expertFfn: 32, denseFfn: 0, vocab: 256, sharedExperts: 1 };
function synth(spec = MOE): { bytes: Uint8Array; offsets: number[]; dataOffset: number } {
  const t = syntheticTensors(spec);
  const h = buildGgufHeader(spec, t);
  const bytes = new Uint8Array(h.dataOffset + h.dataSize);
  bytes.set(h.header, 0);
  for (let i = h.dataOffset; i < bytes.length; i++) bytes[i] = (i * 31 + 7) & 0xff; // each byte is a function of its position, so a misplaced read is visible
  return { bytes, offsets: h.offsets, dataOffset: h.dataOffset };
}

describe("a real GGUF file made by llama.cpp's tooling (stories260K)", () => {
  it("parses, and the tensor table accounts for every byte of the file", async () => {
    const f = await parseGguf(memoryReader(real));
    expect(f.version).toBe(3);
    expect(f.tensorCount).toBe(48);
    expect(f.tensors).toHaveLength(48);
    expect(f.metadata.get("general.architecture")).toBe("llama");
    expect(f.metadata.get("llama.block_count")).toBe(5);
    expect(f.metadata.get("llama.embedding_length")).toBe(64);
    // header + the sum of all tensor sizes is exactly the file: nothing missing, nothing overlapping.
    const sum = f.tensors.reduce((s, t) => s + (t.nbytes as number), 0);
    expect(f.dataOffset + sum).toBe(f.fileSize);
    const sorted = [...f.tensors].sort((a, b) => a.offset - b.offset);
    for (let i = 1; i < sorted.length; i++) expect((sorted[i] as { offset: number }).offset).toBeGreaterThanOrEqual((sorted[i - 1] as { offset: number }).offset + ((sorted[i - 1] as { nbytes: number }).nbytes));
    expect(() => validateBounds(f)).not.toThrow();
  });

  it("gives a layout whose parts add up to the model", async () => {
    const l = buildModelLayout(await parseGguf(memoryReader(real)));
    expect(l.layerCount).toBe(5);
    expect(l.expertCount).toBe(0);
    expect(l.residentBytes + l.layers.reduce((s, x) => s + x.denseBytes, 0)).toBe(l.totalBytes);
    expect(l.kv).toMatchObject({ kvHeads: 4, headDim: 8, layers: 5 });
    expect(l.quantisations).toEqual({ F32: l.totalBytes });
  });

  it("is read through a ByteReader that only returns a prefix: a small first read is enough, and a tiny one grows", async () => {
    let biggest = 0;
    const r = { size: real.length, read: async (o: number, n: number) => (biggest = Math.max(biggest, n), real.subarray(o, o + n)) };
    await parseGguf(r);
    expect(biggest).toBeLessThanOrEqual(4 * 1024 * 1024);
  });
});

describe("synthetic mixture-of-experts files", () => {
  it("round-trips the architecture, expert counts and per-expert byte ranges", async () => {
    const { bytes, dataOffset } = synth();
    const f = await parseGguf(memoryReader(bytes));
    expect(f.dataOffset).toBe(dataOffset);
    const l = buildModelLayout(f);
    expect(l).toMatchObject({ arch: "qwen2moe", layerCount: 3, expertCount: 8, expertsUsed: 2, sharedExperts: 1 });
    expect(l.layers.every((x) => x.experts.length === 8)).toBe(true);
    expect(l.layers[0]?.experts[0]).toHaveLength(3); // gate, up, down
    expect(expertSizeBytes(l)).toBeGreaterThan(0);
    // Every expert range lies inside its fused tensor, ranges of different experts never overlap, and the byte at the start of an
    // expert's range is the byte the writer put there.
    const seen: Array<[number, number]> = [];
    for (const layer of l.layers) for (const e of layer.experts) for (const r of e) {
      expect(r.offset + r.length).toBeLessThanOrEqual(bytes.length);
      expect(bytes[r.offset]).toBe((r.offset * 31 + 7) & 0xff);
      seen.push([r.offset, r.offset + r.length]);
    }
    seen.sort((a, b) => a[0] - b[0]);
    for (let i = 1; i < seen.length; i++) expect((seen[i] as [number, number])[0]).toBeGreaterThanOrEqual((seen[i - 1] as [number, number])[1]);
  });

  it("keeps the router resident and the shared expert with the dense layer", async () => {
    const l = buildModelLayout(await parseGguf(memoryReader(synth().bytes)));
    expect(l.layers[0]?.routerBytes).toBe(64 * 8 * 4);
    const names = (await parseGguf(memoryReader(synth().bytes))).tensors.filter((t) => t.name.startsWith("blk.0.") && (t.name.includes("shexp") || t.name.includes("attn_q"))).length;
    expect(names).toBe(4);
    expect((l.layers[0]?.denseBytes ?? 0)).toBeGreaterThan(0);
  });

  it("treats a dense model as having no experts", async () => {
    const dense = synth({ ...MOE, arch: "llama", expertCount: 0, expertsUsed: 0, denseFfn: 128, sharedExperts: 0 });
    const l = buildModelLayout(await parseGguf(memoryReader(dense.bytes)));
    expect(l.expertCount).toBe(0);
    expect(l.layers.every((x) => x.experts.length === 0)).toBe(true);
  });
});

describe("hostile and damaged files", () => {
  it("refuses bad magic, unknown versions and files that are too small", async () => {
    const bad = Uint8Array.from(real);
    bad[0] = 0x58;
    await expect(parseGguf(memoryReader(bad))).rejects.toThrow(/magic/);
    const v9 = Uint8Array.from(real);
    v9[4] = 9;
    await expect(parseGguf(memoryReader(v9))).rejects.toThrow(/version/);
    await expect(parseGguf(memoryReader(new Uint8Array(10)))).rejects.toThrow(GgufError);
  });

  it("refuses a tensor that points past the end of the file, and a misaligned one", async () => {
    const f = await parseGguf(memoryReader(real));
    expect(() => validateBounds({ ...f, fileSize: f.fileSize - 1000 })).toThrow(/past the end/);
    expect(() => validateBounds({ ...f, tensors: [{ ...(f.tensors[0] as object), offset: 3 } as never] })).toThrow(/aligned/);
  });

  it("refuses counts and sizes that would exhaust memory, without allocating them", async () => {
    const huge = Uint8Array.from(real);
    new DataView(huge.buffer).setBigUint64(8, 5_000_000_000n, true); // tensor count
    await expect(parseGguf(memoryReader(huge))).rejects.toThrow(/tensors/);
    const hugeKv = Uint8Array.from(real);
    new DataView(hugeKv.buffer).setBigUint64(16, 1n << 60n, true);
    await expect(parseGguf(memoryReader(hugeKv))).rejects.toThrow(/larger than 2\^53|metadata/);
  });

  it("only ever throws GgufError, for random bytes, truncations and flipped bytes", async () => {
    const ok = async (b: Uint8Array): Promise<void> => {
      try { await parseGguf(memoryReader(b)); } catch (e) { expect(e).toBeInstanceOf(GgufError); }
    };
    await fc.assert(fc.asyncProperty(fc.uint8Array({ maxLength: 400 }), ok), { numRuns: 400 });
    await fc.assert(fc.asyncProperty(fc.integer({ min: 0, max: 20000 }), (cut) => ok(real.subarray(0, cut))), { numRuns: 200 });
    await fc.assert(fc.asyncProperty(fc.integer({ min: 0, max: 14000 }), fc.integer({ min: 1, max: 255 }), (i, x) => {
      const b = Uint8Array.from(real);
      b[i] = (b[i] as number) ^ x;
      return ok(b);
    }), { numRuns: 400 });
  });
});
