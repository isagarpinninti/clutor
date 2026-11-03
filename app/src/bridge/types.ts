import type { NativeReady, Transport } from "@engine/session";

export interface SystemInfo { os: string; arch: string; ramBytes: number; cores: number; modelsDir: string }
export interface ModelInfo { path: string; name: string; bytes: number }
export interface CatalogItem { id: string; name: string; bytes: number; description: string }
export interface DownloadProgress { id: string; done: number; total: number; state: "downloading" | "verifying" | "done" | "error"; error?: string }

/** A running engine: the events and operations of `clutor-engine chat`, and the first (ready) event. */
export interface OpenEngine { transport: Transport; ready: NativeReady }

/** Everything the interface needs from the machine it runs on. The real app implements it with Tauri commands; a browser preview with a mock. */
export interface Bridge {
  readonly native: boolean;
  system(): Promise<SystemInfo>;
  listModels(): Promise<ModelInfo[]>;
  catalog(): Promise<CatalogItem[]>;
  download(id: string, onProgress: (p: DownloadProgress) => void): Promise<void>;
  openEngine(path: string, ctx: number, lock: boolean): Promise<OpenEngine>;
  revealModels(): Promise<void>;
}
