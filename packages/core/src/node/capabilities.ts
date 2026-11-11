import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { platform, arch } from "node:os";

/**
 * What this computer can do, found by asking it (sysctl, /proc/cpuinfo, nvidia-smi), never by guessing from a name.
 * "available" means the hardware is there; it does NOT mean Clutor uses it. `usedByClutor` says that, and today the engine computes on the CPU only
 * (a GPU backend is a planned seam, see docs/DIRECTIONS.md), so it is false for every accelerator.
 */
export interface Accelerator { kind: "gpu" | "npu"; name: string; api: string; usedByClutor: boolean }
export interface Capabilities { cpuFeatures: string[]; accelerators: Accelerator[]; notes: string[] }

const sh = (cmd: string, args: string[]): string => {
  try { const r = spawnSync(cmd, args, { encoding: "utf8", timeout: 4000 }); return r.status === 0 ? r.stdout : ""; } catch { return ""; }
};

export function detectCapabilities(): Capabilities {
  const features: string[] = [];
  const accelerators: Accelerator[] = [];
  const notes: string[] = [];
  const p = platform();
  if (p === "darwin") {
    if (arch() === "arm64") {
      features.push("neon");
      for (const [key, name] of [["hw.optional.arm.FEAT_DotProd", "dotprod"], ["hw.optional.arm.FEAT_I8MM", "i8mm"], ["hw.optional.arm.FEAT_BF16", "bf16"]] as const) if (sh("sysctl", ["-n", key]).trim() === "1") features.push(name);
      accelerators.push({ kind: "gpu", name: "Apple GPU", api: "Metal", usedByClutor: false });
      accelerators.push({ kind: "npu", name: "Apple Neural Engine", api: "Core ML", usedByClutor: false });
    } else {
      const f = sh("sysctl", ["-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"]).toLowerCase();
      for (const x of ["avx2", "avx512f", "fma"]) if (f.includes(x)) features.push(x);
    }
  } else if (p === "linux") {
    const info = existsSync("/proc/cpuinfo") ? readFileSync("/proc/cpuinfo", "utf8").toLowerCase() : "";
    for (const x of ["avx2", "avx512f", "fma", "neon", "asimddp", "i8mm"]) if (new RegExp(`\\b${x}\\b`).test(info)) features.push(x === "asimddp" ? "dotprod" : x);
    const smi = sh("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"]).trim();
    if (smi) for (const n of smi.split("\n")) accelerators.push({ kind: "gpu", name: n.trim(), api: "CUDA", usedByClutor: false });
  } else if (p === "win32") {
    const smi = sh("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"]).trim();
    if (smi) for (const n of smi.split("\n")) accelerators.push({ kind: "gpu", name: n.trim(), api: "CUDA", usedByClutor: false });
    else notes.push("GPU detection on Windows only recognises NVIDIA cards (nvidia-smi); other GPUs are not detected.");
  }
  if (accelerators.length) notes.push("The engine computes on the CPU only; the accelerators above are present but unused.");
  return { cpuFeatures: features, accelerators, notes };
}
