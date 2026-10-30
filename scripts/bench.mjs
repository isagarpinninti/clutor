// Times the hot paths. Budgets are generous: they catch a 10x regression, not noise. Set CLUTOR_PERF_SCALE=5 on a slow machine.
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  AdaptiveTinyLfuCache, BASELINE, CATALOG, CLUTOR_DEFAULTS, GiB, LruCache, QUANTS, WTinyLfuCache, approximateLayout, computeUnifiedDiff, generateRoutingTrace, memoryReader, parseGguf, planCommand, planMemory, simulateStreaming,
} from "../packages/core/dist/index.js";

const SCALE = Number(process.env.CLUTOR_PERF_SCALE ?? 1);
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
let failed = false;
const rate = async (name, n, fn, minPerSec) => {
  for (let i = 0; i < Math.min(5, n); i++) await fn(i);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) await fn(i);
  const ms = performance.now() - t0;
  const perSec = n / (ms / 1000);
  const min = minPerSec / SCALE;
  const ok = perSec >= min;
  if (!ok) failed = true;
  console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(44)} ${perSec.toFixed(0).padStart(10)} /s  ${(ms / n).toFixed(3).padStart(9)} ms each  (budget >= ${min.toFixed(0)}/s)`);
};

const bytes = readFileSync(join(root, "packages/core/test/fixtures/stories260K.gguf"));
await rate("GGUF header parse (real 1.2 MB file)", 500, () => parseGguf(memoryReader(bytes)), 300);
const layout = approximateLayout(CATALOG.find((c) => c.id === "deepseek-v3"), QUANTS.Q4_K_M);
await rate("memory plan, DeepSeek-V3", 2000, () => planMemory(layout, { hostRamBytes: 16 * GiB }), 1000);
const a = Array.from({ length: 4000 }, (_, i) => `line ${i}`), b = a.slice(); b[1500] = "changed"; b.splice(3000, 0, "inserted");
const ta = a.join("\n") + "\n", tb = b.join("\n") + "\n";
await rate("unified diff of a 4,000-line file, 2 edits", 100, () => computeUnifiedDiff("f", ta, tb), 50);
await rate("command policy check", 20000, () => planCommand("npm run build && npm test"), 20000);

for (const [name, make] of [["LRU", () => new LruCache(20000)], ["W-TinyLFU", () => new WTinyLfuCache(20000, { expectedKeys: 60000 })], ["adaptive W-TinyLFU", () => new AdaptiveTinyLfuCache(20000, { expectedKeys: 60000 })]]) {
  const c = make();
  let x = 1;
  await rate(`cache ${name}: access + admit`, 200000, () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; const k = x % 60000; if (!c.access(k)) c.admit(k, 1); }, name === "adaptive W-TinyLFU" ? 100000 : 300000);
}

const spec = CATALOG.find((c) => c.id === "qwen3-30b-a3b");
const l2 = approximateLayout(spec, QUANTS.Q4_K_M);
const moe = l2.layers.filter((l) => l.experts.length).map((l) => l.index);
const t0 = performance.now();
const trace = generateRoutingTrace({ moeLayers: moe, experts: spec.experts, used: spec.expertsUsed, tokens: 300, seed: 1 });
console.log(`info routing trace: ${(performance.now() - t0).toFixed(0)} ms for 300 tokens x ${moe.length} layers x ${spec.experts} experts`);
await rate("stream simulation, 300 tokens (clutor policy)", 3, () => simulateStreaming(l2, trace, { ...CLUTOR_DEFAULTS, cacheBytes: 2 * GiB, disk: { bytesPerSec: 3 * GiB, requestLatencySec: 1e-4 }, computePerLayerSec: 5e-4 }), 1);
await rate("stream simulation, 300 tokens (LRU)", 3, () => simulateStreaming(l2, trace, { ...BASELINE, cacheBytes: 2 * GiB, disk: { bytesPerSec: 3 * GiB, requestLatencySec: 1e-4 }, computePerLayerSec: 5e-4 }), 1);
process.exit(failed ? 1 : 0);
