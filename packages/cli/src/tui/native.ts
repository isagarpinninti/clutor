import { spawn, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import type { ChatResult, LlmClient } from "@clutor/core";
import { EngineSession, EventQueue, NativeError, type Ev, type NativeReady, type SampleOptions, type Transport } from "./session";
import type { ChatMessage } from "./stream";

/** The engine as a child process (`clutor-engine chat`) in the terminal; the conversation logic is in session.ts. */

export { NativeError };
export type { NativeReady, SampleOptions };

export interface NativeOptions { bin: string; model: string; ctx: number; extraArgs?: string[]; readyTimeoutMs?: number }

class ProcessTransport implements Transport {
  readonly queue = new EventQueue();
  stderr = "";
  constructor(private readonly child: ChildProcess) {}
  send(op: Record<string, unknown>): void {
    if (this.queue.closed || !this.child.stdin?.writable) return;
    this.child.stdin.write(`${JSON.stringify(op)}\n`);
  }
  kill(): void { try { this.child.kill("SIGKILL"); } catch { /* already gone */ } }
  failure(): string { return this.stderr; }
  async close(): Promise<void> {
    if (this.queue.closed) return;
    try { this.child.stdin?.end(); } catch { /* closed */ }
    await new Promise<void>((resolve) => {
      const t = setTimeout(() => { this.child.kill("SIGKILL"); resolve(); }, 3000);
      this.child.once("exit", () => { clearTimeout(t); resolve(); });
    });
  }
}

export class NativeEngine extends EngineSession {
  static async start(o: NativeOptions): Promise<NativeEngine> {
    const child = spawn(o.bin, ["chat", o.model, "--ctx", String(o.ctx), ...(o.extraArgs ?? [])], { stdio: ["pipe", "pipe", "pipe"] });
    const transport = new ProcessTransport(child);
    child.stderr?.on("data", (b: Buffer) => { transport.stderr = (transport.stderr + b.toString()).slice(-4000); });
    child.on("error", () => undefined);
    const rl = createInterface({ input: child.stdout as NodeJS.ReadableStream });
    rl.on("line", (l) => { let e: Ev; try { e = JSON.parse(l) as Ev; } catch { return; } transport.queue.push(e); });
    rl.on("close", () => transport.queue.push(null));
    child.on("exit", () => transport.queue.push(null));
    const timer = setTimeout(() => { child.kill("SIGKILL"); }, o.readyTimeoutMs ?? 600_000);
    const first = await transport.queue.next();
    clearTimeout(timer);
    if (!first || first.ev !== "ready") {
      child.kill("SIGKILL");
      throw new NativeError(first?.ev === "error" ? String(first["msg"]) : `the engine did not start${transport.stderr.trim() ? `: ${transport.stderr.trim().split("\n").slice(-2).join(" ")}` : ""}`);
    }
    return new NativeEngine(transport, first as unknown as NativeReady);
  }

  /** The agent's view of the engine. Replies are still validated by the agent loop; there is no grammar-constrained decoding here. */
  asLlm(name: string, o: SampleOptions = {}): LlmClient {
    return {
      model: name,
      toolMode: "json",
      chat: async (messages, opts): Promise<ChatResult> => {
        const msgs: ChatMessage[] = messages.map((m) => ({ role: m.role === "system" ? "system" : m.role === "assistant" ? "assistant" : "user", content: m.content ?? "" }));
        const r = await this.complete(msgs, { thinking: false, temp: 0.2, max: Math.max(1024, Math.min(4096, Math.floor(this.ready.nCtx / 4))), ...o, ...(opts?.signal ? { signal: opts.signal } : {}) });   // a whole file written into a tool call is long: a cap of 1,024 tokens cut such replies off
        return { content: r.content, toolCalls: [], finishReason: r.finish, usage: { promptTokens: r.promptTokens, completionTokens: r.completionTokens } };
      },
    };
  }
}
