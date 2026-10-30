import { createWriteStream } from "node:fs";
import { open, stat } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { buildGgufHeader, syntheticTensors, type SyntheticModelSpec } from "../gguf/write";
import type { ByteReader } from "../gguf/parse";
import type { ModelLayout } from "../gguf/layout";
import { mulberry32 } from "../engine/trace";
import type { DiskModel } from "../engine/streamer";

/** A ByteReader over a file on disk: only the requested bytes are read. */
export async function fileReader(path: string): Promise<ByteReader & { close(): Promise<void> }> {
  const fh = await open(path, "r");
  const size = (await stat(path)).size;
  return {
    size,
    async read(offset, length) {
      const n = Math.min(length, size - offset);
      const buf = Buffer.alloc(Math.max(0, n));
      const { bytesRead } = await fh.read(buf, 0, buf.length, offset);
      return buf.subarray(0, bytesRead);
    },
    close: () => fh.close(),
  };
}

/** Writes a valid GGUF file with the spec's tensors and deterministic pseudo-random data, streaming, so a multi-gigabyte file needs no memory. */
export async function writeSyntheticGguf(path: string, spec: SyntheticModelSpec, seed = 1): Promise<{ bytes: number; dataOffset: number }> {
  const { header, dataSize, dataOffset } = buildGgufHeader(spec, syntheticTensors(spec));
  const out = createWriteStream(path);
  const rnd = mulberry32(seed);
  const done = new Promise<void>((res, rej) => { out.on("finish", res); out.on("error", rej); });
  const write = (b: Uint8Array): Promise<void> => new Promise((res) => { if (out.write(b)) res(); else out.once("drain", res); });
  await write(header);
  const chunk = new Uint8Array(1 << 20);
  for (let left = dataSize; left > 0;) {
    const n = Math.min(left, chunk.length);
    for (let i = 0; i < n; i += 4) { const v = (rnd() * 4294967296) >>> 0; chunk[i] = v; if (i + 1 < n) chunk[i + 1] = v >>> 8; if (i + 2 < n) chunk[i + 2] = v >>> 16; if (i + 3 < n) chunk[i + 3] = v >>> 24; }
    await write(chunk.subarray(0, n));
    left -= n;
  }
  out.end();
  await done;
  return { bytes: header.length + dataSize, dataOffset };
}

export interface ReadMeasurement {
  samples: number;
  bytes: number;
  seconds: number;
  bytesPerSec: number;
  /** Mean time of one expert read, seconds. */
  meanReadSec: number;
  p95ReadSec: number;
  /** True when the file is large enough relative to RAM that most of these reads cannot have come from the page cache. */
  likelyCold: boolean;
  fileBytes: number;
}

/**
 * Measures real random reads of expert-sized ranges from a model file. Reads go through the operating system's page cache, which
 * Node cannot bypass, so a file smaller than RAM may be served from memory after the first touch; `likelyCold` says whether the file
 * is big enough (more than about 1.5x RAM) for the figure to mean "from the SSD".
 */
export async function measureExpertReads(path: string, layout: ModelLayout, totalRamBytes: number, samples = 48, seed = 3): Promise<ReadMeasurement> {
  const fh = await open(path, "r");
  const fileBytes = (await stat(path)).size;
  try {
    const rnd = mulberry32(seed);
    const experts = layout.layers.flatMap((l) => l.experts).filter((e) => e.length > 0);
    if (experts.length === 0) throw new Error("this model has no experts to read");
    const times: number[] = [];
    let bytes = 0;
    const t0 = performance.now();
    for (let i = 0; i < samples; i++) {
      const ranges = experts[Math.floor(rnd() * experts.length)] as ModelLayout["layers"][number]["experts"][number];
      const s = performance.now();
      for (const r of ranges) {
        const buf = Buffer.allocUnsafe(r.length);
        const { bytesRead } = await fh.read(buf, 0, r.length, r.offset);
        bytes += bytesRead;
      }
      times.push((performance.now() - s) / 1000);
    }
    const seconds = (performance.now() - t0) / 1000;
    times.sort((a, b) => a - b);
    return { samples, bytes, seconds, bytesPerSec: bytes / seconds, meanReadSec: times.reduce((a, b) => a + b, 0) / times.length, p95ReadSec: times[Math.floor(times.length * 0.95)] as number, likelyCold: fileBytes > 1.5 * totalRamBytes, fileBytes };
  } finally {
    await fh.close();
  }
}

/** A disk model for the streaming simulator from a real measurement. */
export function diskModelFrom(m: ReadMeasurement, expertBytes: number, rangesPerExpert: number): DiskModel {
  // Split the mean expert read into a per-request cost and a transfer cost, so smaller or merged requests are modelled sensibly.
  const transfer = expertBytes / m.bytesPerSec;
  const latency = Math.max(0, (m.meanReadSec - transfer) / Math.max(1, rangesPerExpert));
  return { bytesPerSec: m.bytesPerSec, requestLatencySec: latency };
}
