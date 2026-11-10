import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";

// The conversation logic (chat templates, the cache-aware prompt sync, cancel) is shared with the terminal client and lives in packages/cli/src/tui.
const shared = fileURLToPath(new URL("../packages/cli/src/tui", import.meta.url));

const core = fileURLToPath(new URL("../packages/core/src", import.meta.url));

const pkg = JSON.parse(readFileSync(fileURLToPath(new URL("./package.json", import.meta.url)), "utf8")) as { version: string };

export default defineConfig({
  define: { __APP_VERSION__: JSON.stringify(pkg.version), __APP_BUILT__: JSON.stringify(new Date().toISOString().slice(0, 10)) },
  plugins: [react()],
  resolve: { alias: { "@engine": shared, "@core": core } },
  clearScreen: false,
  server: { port: 1420, strictPort: true, fs: { allow: [".."] } },
  envPrefix: ["VITE_", "TAURI_ENV_*"],
  build: { target: "es2022", sourcemap: true },
  test: { environment: "jsdom", globals: true },
});
