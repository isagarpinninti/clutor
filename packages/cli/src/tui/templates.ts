import type { ChatMessage } from "./stream";

/**
 * Chat templates for the models the native engine runs. A model's GGUF carries its template as Jinja source; rather than run a Jinja
 * interpreter, the four layouts these architectures use are written out, chosen by the markers their template contains. A model whose
 * template matches none of them gets a plain "User: / Assistant:" layout, and the UI says so.
 */

export type TemplateFamily = "chatml" | "llama3" | "granite" | "olmo" | "plain";

export interface TemplateInfo {
  family: TemplateFamily;
  /** The model writes its reasoning between <think> tags (Qwen3, Qwen3.5). */
  thinkTags: boolean;
  /** With thinking on, the template itself opens the block (`<think>\n` ends the prompt), as Qwen3.5's does; Qwen3's model writes the opening tag itself. */
  opensThink?: boolean;
}

export function detectTemplate(jinja: string | undefined): TemplateInfo {
  const t = jinja ?? "";
  const thinkTags = t.includes("<think>");
  if (t.includes("<|im_start|>")) return { family: "chatml", thinkTags, ...(thinkTags && /\{%-?\s*else\s*%\}\s*\{\{-?\s*'<think>\\n'\s*\}\}/.test(t) ? { opensThink: true } : {}) };
  if (t.includes("<|start_header_id|>")) return { family: "llama3", thinkTags: false };
  if (t.includes("<|start_of_role|>")) return { family: "granite", thinkTags: false };
  if (t.includes("<|assistant|>") && t.includes("<|user|>")) return { family: "olmo", thinkTags: false };
  return { family: "plain", thinkTags: false };
}

export interface RenderOptions {
  /** Let a model that supports it think first. Off adds the empty think block Qwen3's own template adds. */
  thinking?: boolean;
  /** Add the opening of the assistant's turn, so the model writes the reply. Default true. */
  generationPrompt?: boolean;
}

/** Everything before the first reply's text, for the messages given. No BOS: the engine adds it. */
/** Where a picture goes in the text: the engine replaces the pad with the picture's vectors. */
export const IMAGE_MARK = "<|vision_start|><|image_pad|><|vision_end|>";

export function renderChat(info: TemplateInfo, messages: ChatMessage[], o: RenderOptions = {}): string {
  const gen = o.generationPrompt ?? true;
  let out = "";
  for (const m of messages) {
    switch (info.family) {
      // A reply generated with thinking off began with an empty think block; rendering it that way keeps the engine's cache valid when the conversation continues.
      case "chatml": out += `<|im_start|>${m.role}\n${(m.images ?? []).map(() => IMAGE_MARK).join("")}${m.role === "assistant" && info.thinkTags && o.thinking !== true ? "<think>\n\n</think>\n\n" : ""}${m.content}<|im_end|>\n`; break;
      case "llama3": out += `<|start_header_id|>${m.role}<|end_header_id|>\n\n${m.content}<|eot_id|>`; break;
      case "granite": out += `<|start_of_role|>${m.role}<|end_of_role|>${m.content}<|end_of_text|>\n`; break;
      case "olmo": out += m.role === "assistant" ? `<|assistant|>\n${m.content}<|endoftext|>\n` : `<|${m.role}|>\n${m.content}\n`; break;
      default: out += m.role === "system" ? `${m.content}\n\n` : m.role === "user" ? `User: ${m.content}\n\n` : `Assistant:${m.content}\n\n`; break;
    }
  }
  if (!gen) return out;
  switch (info.family) {
    case "chatml": return `${out}<|im_start|>assistant\n${info.thinkTags ? (o.thinking === true ? (info.opensThink ? "<think>\n" : "") : "<think>\n\n</think>\n\n") : ""}`;
    case "llama3": return `${out}<|start_header_id|>assistant<|end_header_id|>\n\n`;
    case "granite": return `${out}<|start_of_role|>assistant<|end_of_role|>`;
    case "olmo": return `${out}<|assistant|>\n`;
    default: return `${out}Assistant:`;
  }
}

/**
 * Splits a streamed reply into reasoning (between <think> and </think>) and the answer. Tags can arrive in pieces, so text that might be
 * the start of a tag is held back until it is clear.
 */
export class ThinkSplitter {
  private buf = "";
  private inThink: boolean;
  constructor(private readonly enabled: boolean, startsInThink = false) { this.inThink = enabled && startsInThink; }

  push(text: string): Array<{ kind: "content" | "reasoning"; text: string }> {
    if (!this.enabled) return text ? [{ kind: "content", text }] : [];
    this.buf += text;
    const out: Array<{ kind: "content" | "reasoning"; text: string }> = [];
    for (;;) {
      const tag = this.inThink ? "</think>" : "<think>";
      const i = this.buf.indexOf(tag);
      if (i >= 0) {
        if (i > 0) out.push({ kind: this.inThink ? "reasoning" : "content", text: this.buf.slice(0, i) });
        this.buf = this.buf.slice(i + tag.length);
        this.inThink = !this.inThink;
        continue;
      }
      // Hold back a tail that could still become the tag.
      let keep = 0;
      for (let n = Math.min(tag.length - 1, this.buf.length); n > 0; n--) if (tag.startsWith(this.buf.slice(this.buf.length - n))) { keep = n; break; }
      const emit = this.buf.slice(0, this.buf.length - keep);
      if (emit) out.push({ kind: this.inThink ? "reasoning" : "content", text: emit });
      this.buf = this.buf.slice(this.buf.length - keep);
      return out;
    }
  }

  end(): Array<{ kind: "content" | "reasoning"; text: string }> {
    const rest = this.buf;
    this.buf = "";
    return rest ? [{ kind: this.inThink ? "reasoning" : "content", text: rest }] : [];
  }
}
