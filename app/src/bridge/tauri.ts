import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { EventQueue, NativeError, type Ev, type NativeReady, type Transport } from "@engine/session";
import type { Bridge, CatalogItem, DownloadProgress, ModelInfo, OpenEngine, SystemInfo } from "./types";

/** The engine runs inside this process (src-tauri links it); its events arrive as Tauri events and its operations go out as commands. */
class TauriTransport implements Transport {
  readonly queue = new EventQueue();
  private unlisten: UnlistenFn | undefined;
  private lastError = "";
  async attach(): Promise<void> {
    this.unlisten = await listen<string>("engine-event", (e) => {
      try { this.queue.push(JSON.parse(e.payload) as Ev); } catch { /* not an event */ }
    });
  }
  send(op: Record<string, unknown>): void {
    invoke("engine_op", { op: JSON.stringify(op) }).catch((err: unknown) => {
      this.lastError = String(err);
      this.queue.push({ ev: "error", msg: this.lastError });
    });
  }
  kill(): void { void invoke("engine_close").catch(() => undefined); this.unlisten?.(); this.queue.push(null); }
  async close(): Promise<void> { this.unlisten?.(); await invoke("engine_close").catch(() => undefined); this.queue.push(null); }
  failure(): string { return this.lastError; }
}

export const tauriBridge: Bridge = {
  native: true,
  system: () => invoke<SystemInfo>("system_info"),
  listModels: () => invoke<ModelInfo[]>("list_models"),
  catalog: () => invoke<CatalogItem[]>("model_catalog"),
  revealModels: () => invoke("reveal_models"),
  async download(id, onProgress) {
    const un = await listen<DownloadProgress>("download-progress", (e) => { if (e.payload.id === id) onProgress(e.payload); });
    try { await invoke("download_model", { id }); } finally { un(); }
  },
  async openEngine(path, ctx, lock): Promise<OpenEngine> {
    const transport = new TauriTransport();
    await transport.attach();
    try {
      const ready = await invoke<string>("engine_open", { path, ctx, lock });
      return { transport, ready: JSON.parse(ready) as NativeReady };
    } catch (e) {
      await transport.close();
      throw new NativeError(String(e));
    }
  },
};
