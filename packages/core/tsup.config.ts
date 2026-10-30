import { defineConfig } from "tsup";

export default defineConfig({
  // `index` is pure (no node:* imports) and runs in browsers; `node` adds files, processes and hardware probing.
  entry: { index: "src/index.ts", node: "src/node/index.ts" },
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  target: "es2022",
  external: [/^node:/],
});
