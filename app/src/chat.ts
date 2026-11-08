import type { EngineSession } from "@engine/session";
import type { ChatImage, ChatMessage } from "@engine/stream";
import { imageRgb, withAttachments } from "./attachments";
import type { Msg, MsgStats, Settings } from "./store";

export interface ReplyUpdate { content: string; reasoning: string; stats?: MsgStats; error?: string; interrupted?: boolean }

/** What goes to the model: the instructions, then the conversation (without replies that failed or are still empty). */
/** The pictures of a message that are still in memory, as the engine takes them. */
function imagesOf(m: Msg): { images?: ChatImage[] } {
  const images = (m.files ?? []).flatMap((f) => { const rgb = f.id && f.image ? imageRgb(f.id) : undefined; return rgb && f.image ? [{ w: f.image.w, h: f.image.h, rgb }] : []; });
  return images.length ? { images } : {};
}

export function toChatMessages(history: Msg[], settings: Pick<Settings, "system">): ChatMessage[] {
  return [
    { role: "system", content: settings.system },
    ...history.filter((m) => !m.error && (m.content || m.role === "user")).map((m): ChatMessage => ({ role: m.role, content: m.role === "user" ? withAttachments(m.content, m.files) : m.content, ...(m.role === "user" ? imagesOf(m) : {}) })),
  ];
}

/** Streams the model's reply to a conversation, reporting the text so far (at most once per frame) and the measured numbers at the end. */
export async function streamReply(session: EngineSession, history: Msg[], settings: Settings, signal: AbortSignal, onUpdate: (u: ReplyUpdate) => void, shape: (m: ChatMessage[]) => ChatMessage[] = (m) => m): Promise<void> {
  const messages = shape(toChatMessages(history, settings));
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
