// Writes the app's model catalog from the CLI's download table (packages/cli/src/models.ts), so both use the same URLs, sizes and SHA-256 checksums.
import { readFileSync, writeFileSync } from "node:fs";
const src = readFileSync(new URL("../../packages/cli/src/models.ts", import.meta.url), "utf8");
const names = { "olmoe-1b-7b-q4_k_m": "OLMoE 1B-7B Instruct", "granite-3.0-3b-a800m-q4_k_m": "Granite 3.0 3B-A800M", "qwen3-30b-a3b-q4_k_m": "Qwen3 30B-A3B", "qwen2.5-1.5b-q4_k_m": "Qwen2.5 1.5B Instruct", "llama-3.2-1b-q4_k_m": "Llama 3.2 1B Instruct", "qwen2.5-coder-7b-q4_k_m": "Qwen2.5 Coder 7B Instruct" };
const about = {
  "llama-3.2-1b-q4_k_m": "The smallest and fastest. Good for a phone or an older laptop.",
  "qwen2.5-1.5b-q4_k_m": "Small, quick and multilingual.",
  "granite-3.0-3b-a800m-q4_k_m": "IBM Granite, a mixture of experts: 3B parameters, 0.8B used per token.",
  "olmoe-1b-7b-q4_k_m": "A mixture of experts: 7B parameters, 1B used per token. A good all-rounder.",
  "qwen2.5-coder-7b-q4_k_m": "Tuned for code; the one to use with the coding agent.",
  "qwen3-30b-a3b-q4_k_m": "A 30B mixture of experts, larger than most computers' memory: Clutor streams it from disk.",
};
const items = [];
for (const m of src.matchAll(/\{ id: "([^"]+)", catalogId: "[^"]*", repo: "([^"]+)", file: "([^"]+)", bytes: (\d+), sha256: "([0-9a-f]{64})", note: "([^"]*)"/g)) {
  items.push({ id: m[1], name: names[m[1]] ?? m[1], repo: m[2], file: m[3], bytes: Number(m[4]), sha256: m[5], description: about[m[1]] ?? m[6] });
}
if (items.length < 5) throw new Error(`expected the download table, found ${items.length} entries`);
writeFileSync(new URL("../src-tauri/catalog.json", import.meta.url), `${JSON.stringify(items, null, 2)}\n`);
console.log(`wrote ${items.length} catalog entries`);
