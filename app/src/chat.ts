import type { EngineSession } from "@engine/session";
import type { ChatMessage } from "@engine/stream";
import type { Msg, MsgStats, Settings } from "./store";

export interface ReplyUpdate { content: string; reasoning: string; stats?: MsgStats; error?: string; interrupted?: boolean }

/** Streams the model's reply to a conversation, reporting the text so far (at most once per frame) and the measured numbers at the end. */
export async function streamReply(session: EngineSession, history: Msg[], settings: Settings, signal: AbortSignal, onUpdate: (u: ReplyUpdate) => void): Promise<void> {
  const messages: ChatMessage[] = [
    { role: "system", content: settings.system },
    ...history.filter((m) => !m.error && (m.content || m.role === "user")).map((m): ChatMessage => ({ role: m.role, content: m.content })),
  ];
  let content = "", reasoning = "", stats: MsgStats | undefined, interrupted = false, error: string | undefined;
  let pending = false;
  const t0 = performance.now();
  const flush = (): void => { pending = false; onUpdate({ content, reasoning, ...(stats ? { stats } : {}), ...(error ? { error } : {}), ...(interrupted ? { interrupted } : {}) }); };
  const schedule = (): void => { if (!pending) { pending = true; requestAnimationFrame(flush); } };
  try {
    for await (const ev of session.stream(messages, { thinking: settings.thinking, temp: settings.temperature, signal })) {
      if (ev.type === "content") { content += ev.text; schedule(); }
      else if (ev.type === "reasoning") { reasoning += ev.text; schedule(); }
      else if (ev.type === "done") {
        interrupted = ev.finish === "cancel";
        stats = {
          tokens: ev.completionTokens ?? 0, ms: performance.now() - t0,
          ...(ev.tokensPerSecond ? { tps: ev.tokensPerSecond } : {}), ...(ev.promptTokens ? { promptTokens: ev.promptTokens } : {}), ...(ev.cache ? { cache: ev.cache } : {}),
        };
      }
    }
  } catch (e) {
    if (signal.aborted) interrupted = true; else error = e instanceof Error ? e.message : String(e);
  }
  flush();
}
