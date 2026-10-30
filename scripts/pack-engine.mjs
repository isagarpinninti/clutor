// Copies the native engine's source into packages/cli/c before `npm pack` / `npm publish`, and removes it afterwards, so the published
// @clutor/cli carries what `clutor engine build` needs. Usage: node scripts/pack-engine.mjs [clean]
import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dest = join(root, "packages", "cli", "c");
rmSync(dest, { recursive: true, force: true });
if (process.argv[2] !== "clean") {
  if (!existsSync(join(root, "c", "Makefile"))) throw new Error("c/Makefile is missing");
  cpSync(join(root, "c", "src"), join(dest, "src"), { recursive: true });
  cpSync(join(root, "c", "Makefile"), join(dest, "Makefile"));
  console.log("engine source copied into packages/cli/c");
}
