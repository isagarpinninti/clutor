import { describe, expect, it } from "vitest";
import { buildRequest, ReplyParser } from "./session";
import { cloudPath, endpoint, explainError, parseCloudPath } from "./providers";
import { SseParser } from "./sse";

describe("server-sent events", () => {
  it("joins pieces that arrive split anywhere", () => {
    const p = new SseParser();
    expect(p.push('event: a\ndata: {"x"')).toEqual([]);
    expect(p.push(':1}\n\ndata: two\n\n')).toEqual([{ event: "a", data: '{"x":1}' }, { event: "message", data: "two" }]);
  });
});

describe("requests", () => {
  const msgs = [{ role: "system" as const, content: "be brief" }, { role: "user" as const, content: "hi" }];
  it("builds an OpenAI-style request with usage asked for", () => {
    const r = buildRequest({ kind: "openai", baseUrl: "https://x.test/v1/" }, "m", msgs, { temp: 0.2, max: 100 });
    expect(r.url).toBe("https://x.test/v1/chat/completions");
    expect(JSON.parse(r.body)).toMatchObject({ model: "m", stream: true, max_tokens: 100, temperature: 0.2, stream_options: { include_usage: true }, messages: msgs });
  });
  it("builds an Anthropic request with the system text apart and a reply continued by its start", () => {
    const r = buildRequest({ kind: "anthropic", baseUrl: "https://a.test/v1" }, "c", [...msgs, { role: "user", content: "more" }], { prefill: '{"tool": "wri ' });
    const b = JSON.parse(r.body);
    expect(r.url).toBe("https://a.test/v1/messages");
    expect(b.system).toBe("be brief");
    expect(b.messages).toEqual([{ role: "user", content: "hi\n\nmore" }, { role: "assistant", content: '{"tool": "wri' }]);
  });
  it("asks an OpenAI-style provider for the rest of a cut-off reply", () => {
    const b = JSON.parse(buildRequest({ kind: "openai", baseUrl: "https://x" }, "m", msgs, { prefill: "abc" }).body);
    expect(b.messages.slice(-2).map((m: { role: string }) => m.role)).toEqual(["assistant", "user"]);
  });
});

describe("replies", () => {
  it("reads an OpenAI-style stream: text, reasoning, usage, end", () => {
    const p = new ReplyParser("openai");
    const ev = p.push('data: {"choices":[{"delta":{"reasoning_content":"hm"}}]}\n\ndata: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\ndata: [DONE]\n\n');
    expect(ev).toEqual([{ type: "reasoning", text: "hm" }, { type: "content", text: "Hi" }]);
    expect([p.promptTokens, p.completionTokens, p.finish, p.done]).toEqual([7, 3, "length", true]);
  });
  it("reads an Anthropic stream", () => {
    const p = new ReplyParser("anthropic");
    const ev = p.push('event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":11}}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Yo"}}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n');
    expect(ev).toEqual([{ type: "content", text: "Yo" }]);
    expect([p.promptTokens, p.completionTokens, p.finish, p.done]).toEqual([11, 4, "end_turn", true]);
  });
  it("raises an error the provider sent inside the stream", () => {
    expect(() => new ReplyParser("openai").push('data: {"error":{"message":"overloaded"}}\n\n')).toThrow(/overloaded/);
  });
});

describe("providers", () => {
  it("round-trips a model path and builds addresses", () => {
    expect(parseCloudPath(cloudPath("openai", "gpt-4o"))).toEqual({ providerId: "openai", model: "gpt-4o" });
    expect(parseCloudPath(cloudPath("openrouter", "openai/gpt:free"))).toEqual({ providerId: "openrouter", model: "openai/gpt:free" });
    expect(parseCloudPath("/models/a.gguf")).toBeUndefined();
    expect(endpoint({ kind: "openai", baseUrl: "http://localhost:11434/v1/" })).toBe("http://localhost:11434/v1/chat/completions");
  });
  it("explains failures in words", () => {
    expect(explainError(0, "401: bad key", "OpenAI")).toMatch(/rejected the API key/);
    expect(explainError(0, "429: slow down", "OpenAI")).toMatch(/too much or are out of credit/);
    expect(explainError(0, "could not reach the provider: dns", "X")).toMatch(/internet connection/);
  });
});
