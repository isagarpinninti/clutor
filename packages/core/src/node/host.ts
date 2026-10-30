import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { arch, cpus, freemem, homedir, platform, release, totalmem } from "node:os";

export interface HostInfo {
  platform: NodeJS.Platform;
  platformName: string;
  arch: string;
  osRelease: string;
  cpuModel: string;
  logicalCores: number;
  totalRamBytes: number;
  /** What the OS reports as completely free. On macOS and Linux this excludes cache that could be reclaimed, so it understates what is usable. */
  freeRamBytes: number;
  /** An estimate of memory a program could get without swapping: free plus reclaimable cache. Equals `freeRamBytes` where the OS gives no better number. */
  availableRamBytes: number;
  homedir: string;
  nodeVersion: string;
}

/** Free memory plus what the system would give back on demand. Falls back to `os.freemem()` if the platform tools are unavailable. */
export function availableMemory(): number {
  try {
    if (platform() === "linux") {
      const m = /MemAvailable:\s+(\d+) kB/.exec(readFileSync("/proc/meminfo", "utf8"));
      if (m) return Number(m[1]) * 1024;
    } else if (platform() === "darwin") {
      const out = execFileSync("vm_stat", { encoding: "utf8", timeout: 2000 });
      const page = Number(/page size of (\d+) bytes/.exec(out)?.[1] ?? 16384);
      const n = (label: string): number => Number(new RegExp(`${label}:\\s+(\\d+)\\.`).exec(out)?.[1] ?? 0);
      const pages = n("Pages free") + n("Pages inactive") + n("Pages speculative") + n("Pages purgeable");
      if (pages > 0) return Math.min(totalmem(), pages * page);
    }
  } catch { /* use the plain figure */ }
  return freemem();
}

export function hostInfo(): HostInfo {
  const c = cpus();
  const p = platform();
  return {
    platform: p, platformName: p === "darwin" ? "macOS" : p === "win32" ? "Windows" : p === "linux" ? "Linux" : p, arch: arch(), osRelease: release(),
    cpuModel: c[0]?.model.trim() ?? "unknown", logicalCores: c.length, totalRamBytes: totalmem(), freeRamBytes: freemem(), availableRamBytes: availableMemory(), homedir: homedir(), nodeVersion: process.versions.node,
  };
}
