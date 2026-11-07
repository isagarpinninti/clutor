import { describe, expect, it } from "vitest";
import { buildGgufHeader, type SyntheticModelSpec } from "@core/gguf/write";
import type { ByteReader } from "@core/gguf/parse";
import { GiB } from "@core/planner/plan";
import { checkReader, diskFromBench } from "./plan";

function reader(spec: SyntheticModelSpec): ByteReader {
  const { header, dataOffset, dataSize } = buildGgufHeader(spec);
  return { size: dataOffset + dataSize, read: async (o, l) => header.subarray(o, Math.min(header.length, o + l)) };
}
const SMALL: SyntheticModelSpec = { arch: "qwen2moe", name: "small-moe", layers: 6, embedding: 256, heads: 8, kvHeads: 4, contextLength: 2048, expertCount: 32, expertsUsed: 4, expertFfn: 256, denseFfn: 0, vocab: 1024, sharedExperts: 0 };
const BIG: SyntheticModelSpec = { arch: "qwen2moe", name: "big-moe", layers: 78, embedding: 6144, heads: 48, kvHeads: 8, contextLength: 131072, expertCount: 256, expertsUsed: 8, expertFfn: 2048, denseFfn: 0, vocab: 150000, sharedExperts: 1, quantType: 12 };

describe("the 'will it run' check", () => {
  it("says a small model fits", async () => {
    const c = await checkReader(reader(SMALL), { hostRamBytes: 16 * GiB, contextTokens: 4096 });
    expect(c.verdict).toBe("fits");
    expect(c.headline).toMatch(/^Fits in memory/);
    expect(c.experts).toBe(32);
  });

  it("says a model far bigger than memory streams its experts, and gives computed bounds, not a promise", async () => {
    const c = await checkReader(reader(BIG), { hostRamBytes: 16 * GiB, contextTokens: 4096 });
    expect(c.fileBytes).toBeGreaterThan(100 * GiB);
    expect(["streams", "tight"]).toContain(c.verdict);
    expect(c.speed).toMatch(/upper bounds|at most/);
    expect(c.notes.join(" ")).toMatch(/assumed/);
  });

  it("uses a measured disk speed and says so, and a faster disk gives a higher bound", async () => {
    const slow = await checkReader(reader(BIG), { hostRamBytes: 16 * GiB, contextTokens: 4096, disk: { bytesPerSec: 0.5 * GiB, measured: true } });
    const fast = await checkReader(reader(BIG), { hostRamBytes: 16 * GiB, contextTokens: 4096, disk: { bytesPerSec: 6 * GiB, measured: true } });
    expect(slow.notes.join(" ")).toMatch(/measured on this computer/);
    const at = (c: typeof slow) => c.plan.speed.find((s) => s.hitRate === 0.95)!.maxTokensPerSec;
    expect(at(fast)).toBeGreaterThan(at(slow) * 10);
  });

  it("says too big when even streaming cannot fit", async () => {
    const c = await checkReader(reader(BIG), { hostRamBytes: 2 * GiB, contextTokens: 4096 });
    expect(c.verdict).toBe("too-big");
  });

  it("more memory never makes a model less able to run", async () => {
    const order = ["too-big", "tight", "streams", "fits"];
    let prev = -1;
    for (const gb of [2, 8, 16, 64, 512]) {
      const v = (await checkReader(reader(BIG), { hostRamBytes: gb * GiB, contextTokens: 4096 })).verdict;
      expect(order.indexOf(v)).toBeGreaterThanOrEqual(prev);
      prev = order.indexOf(v);
    }
  });
});

describe("disk measurement", () => {
  it("uses the scattered rate, and flags a figure too fast for a disk", () => {
    expect(diskFromBench({ sequentialBytesPerSec: 3e9, scatteredBytesPerSec: 1e9 })).toEqual({ disk: { bytesPerSec: 1e9, measured: true } });
    expect(diskFromBench({ sequentialBytesPerSec: 30e9, scatteredBytesPerSec: 20e9 }).warning).toMatch(/memory/);
  });
});
