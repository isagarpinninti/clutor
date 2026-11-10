import { buildModelLayout, type ModelLayout } from "@core/gguf/layout";
import { parseGguf, type ByteReader } from "@core/gguf/parse";
import { GiB, planMemory, type MemoryPlan } from "@core/planner/plan";
import type { InspectApi } from "./bridge";

/**
 * "Will it run?": reads the model file's header (a few megabytes, not the model), works out where every byte would go on this computer, and states the fastest the disk could
 * possibly let it generate. Everything here is computed from the file and the numbers given; the disk speed is either assumed or the one the user measured, and says which.
 */

export interface DiskInfo { bytesPerSec: number; measured: boolean }

export interface ModelCheck {
  name: string;
  arch: string;
  fileBytes: number;
  layers: number;
  experts: number;
  expertsUsed: number;
  plan: MemoryPlan;
  verdict: "fits" | "streams" | "tight" | "too-big";
  headline: string;
  speed: string;
  notes: string[];
}

/** The reader the planner needs, over a file on this computer. */
export function fileReader(inspect: InspectApi, path: string, size: number): ByteReader {
  return { size, read: (offset, length) => inspect.read(path, offset, length) };
}

const gib = (b: number): string => `${(b / GiB).toFixed(b >= 10 * GiB ? 0 : 1)} GB`;
const tps = (v: number): string => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

export function describe(layout: ModelLayout, plan: MemoryPlan, hostRam: number, disk: DiskInfo): Omit<ModelCheck, "name" | "arch" | "fileBytes" | "layers" | "experts" | "expertsUsed" | "plan"> {
  // The planner words its own caveat about an assumed disk; this check says it once, below.
  const notes = plan.warnings.filter((w) => !/^Disk speed is an assumption/.test(w));
  const at = (h: number) => plan.speed.find((s) => Math.abs(s.hitRate - h) < 1e-9)?.maxTokensPerSec;
  const low = at(0.5), high = at(0.95);
  const range = low !== undefined && high !== undefined && Number.isFinite(low) && Number.isFinite(high) ? `at most ${tps(low)} to ${tps(high)} tokens per second` : undefined;
  const diskNote = disk.measured ? `Disk speed used: ${(disk.bytesPerSec / 1e9).toFixed(1)} GB/s, measured on this computer.` : `Disk speed used: ${(disk.bytesPerSec / 1e9).toFixed(1)} GB/s, assumed. Measure it for a real figure.`;
  if (plan.strategy === "fully-resident") {
    return { verdict: "fits", headline: `Fits in memory (${gib(plan.usedBytes)} of ${gib(hostRam)}).`, speed: "Speed is limited by the processor, not the disk. Run it to see the real speed.", notes };
  }
  if (plan.strategy === "does-not-fit") {
    return { verdict: "too-big", headline: `Too big for this computer: it needs ${gib(plan.usedBytes + plan.shortfallBytes)} even with streaming, and about ${gib(plan.ceilingBytes)} is available.`, speed: "It cannot be loaded.", notes };
  }
  const tight = plan.strategy === "dense-ring-experts-streamed";
  notes.push(diskNote);
  return {
    verdict: tight ? "tight" : "streams",
    headline: tight ? "Runs, slowly: the always-needed weights do not fit, so they are re-read from disk for every token." : `Runs by streaming experts from disk (${gib(plan.usedBytes)} in memory, the rest on disk).`,
    speed: range ? `The disk allows ${range}, depending on how often an expert is already in memory (computed upper bounds; real speed is lower).` : "No speed bound could be computed.",
    notes,
  };
}

export async function checkModel(inspect: InspectApi, path: string, o: { hostRamBytes: number; contextTokens: number; disk?: DiskInfo | undefined }): Promise<ModelCheck> {
  const size = await inspect.size(path);
  return checkReader(fileReader(inspect, path, size), o);
}

export async function checkReader(reader: ByteReader, o: { hostRamBytes: number; contextTokens: number; disk?: DiskInfo | undefined }): Promise<ModelCheck> {
  const layout = buildModelLayout(await parseGguf(reader));
  const disk: DiskInfo = o.disk ?? { bytesPerSec: 3 * GiB, measured: false };
  const plan = planMemory(layout, { hostRamBytes: o.hostRamBytes, contextTokens: o.contextTokens, diskBytesPerSec: disk.bytesPerSec, diskMeasured: disk.measured });
  const d = describe(layout, plan, o.hostRamBytes, disk);
  return { name: layout.name, arch: layout.arch, fileBytes: layout.totalBytes, layers: layout.layerCount, experts: layout.expertCount, expertsUsed: layout.expertsUsed, plan, ...d };
}

/** A disk figure from a measurement: the scattered rate when reading experts (that is what streaming does), flagged when it is too fast to be a disk. */
export function diskFromBench(b: { sequentialBytesPerSec: number; scatteredBytesPerSec: number }): { disk: DiskInfo; warning?: string } {
  const bytesPerSec = Math.max(1, b.scatteredBytesPerSec);
  const cached = b.sequentialBytesPerSec > 12e9 || b.scatteredBytesPerSec > 9e9;
  return { disk: { bytesPerSec, measured: true }, ...(cached ? { warning: "That is faster than a disk can be, so the system probably served the file from memory. Run the check on a file that has not been read recently for a real figure." } : {}) };
}
