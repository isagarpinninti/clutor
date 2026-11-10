import { mockBridge } from "./mock";
import { tauriBridge } from "./tauri";
import type { Bridge } from "./types";

export * from "./types";

/** The real bridge inside the Tauri shell, the mock in a plain browser. */
export const bridge: Bridge = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window ? tauriBridge : mockBridge;
