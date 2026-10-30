import type { ShadowWorkspace } from "./workspace";

const OTHER_RUNNERS = ["jest", "vitest", "mocha", "jasmine", "ava", "tap", "@jest/globals", "@playwright/test"];

/**
 * A few lines about the project that a small model gets wrong when left to guess: whether `.js` files are ES modules or CommonJS, and which test API
 * exists. Read from package.json only; nothing is run. Empty when the project has no package.json.
 */
export async function workspaceHints(ws: Pick<ShadowWorkspace, "read">): Promise<string> {
  const text = await ws.read("package.json");
  if (text === null) return "";
  let pkg: { type?: unknown; scripts?: Record<string, unknown>; dependencies?: Record<string, unknown>; devDependencies?: Record<string, unknown> };
  try { pkg = JSON.parse(text) as typeof pkg; } catch { return ""; }
  if (!pkg || typeof pkg !== "object") return "";
  const esm = pkg.type === "module";
  const deps = { ...(pkg.dependencies ?? {}), ...(pkg.devDependencies ?? {}) };
  const runner = OTHER_RUNNERS.find((r) => r in deps);
  const testScript = typeof pkg.scripts?.["test"] === "string" ? (pkg.scripts["test"] as string) : "";
  const lines = [
    "Project environment (from package.json):",
    esm
      ? '- "type" is "module": .js files are ES modules. Use import/export. Do not use require, module.exports or __dirname.'
      : '- "type" is not "module": .js files are CommonJS. Use require and module.exports; do not use import/export in .js files (use .mjs for that).',
  ];
  if (runner) lines.push(`- The project depends on ${runner}; write tests for it and follow the style of the existing tests.`);
  else {
    const imp = esm
      ? "import { describe, it, test } from 'node:test'; import assert from 'node:assert/strict';"
      : "const { describe, it, test } = require('node:test'); const assert = require('node:assert/strict');";
    lines.push(`- No test framework is installed. Write tests for Node's built-in runner (\`node --test\`, files named *.test.js): ${imp}`);
    lines.push("- describe, it, test, expect and assert are NOT globals. Import every one you use, as above.");
  }
  if (testScript) lines.push(`- The test script is \`${testScript}\`.`);
  return lines.join("\n");
}
