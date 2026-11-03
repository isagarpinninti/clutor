import { useCallback, useRef, useState } from "react";
import { EngineSession, type NativeReady } from "@engine/session";
import { bridge, type ModelInfo } from "./bridge";
import { shouldPin, type Settings } from "./store";

export interface EngineState {
  status: "idle" | "loading" | "ready" | "error";
  model?: ModelInfo;
  ready?: NativeReady;
  error?: string;
  loadMs?: number;
  /** Pinning was wanted but skipped because the model is large for this machine. */
  unpinned?: boolean;
}

/** Loads a model into the engine and keeps the running session. One model at a time; loading another closes the first. */
export function useEngine() {
  const [state, setState] = useState<EngineState>({ status: "idle" });
  const session = useRef<EngineSession | null>(null);

  const unload = useCallback(async () => {
    const s = session.current;
    session.current = null;
    if (s) await s.close().catch(() => undefined);
    setState({ status: "idle" });
  }, []);

  const load = useCallback(async (model: ModelInfo, settings: Pick<Settings, "ctx" | "lock">, ramBytes?: number) => {
    const old = session.current;
    session.current = null;
    setState({ status: "loading", model });
    if (old) await old.close().catch(() => undefined);
    const t0 = performance.now();
    const pin = shouldPin(settings.lock, model.bytes, ramBytes);
    try {
      const { transport, ready } = await bridge.openEngine(model.path, settings.ctx, pin);
      session.current = new EngineSession(transport, ready);
      setState({ status: "ready", model, ready, loadMs: performance.now() - t0, ...(settings.lock && !pin ? { unpinned: true } : {}) });
    } catch (e) {
      setState({ status: "error", model, error: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  return { state, session, load, unload };
}
