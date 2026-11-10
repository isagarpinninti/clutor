/** A cloud provider the user set up. The API key is not here: it is kept by the app's native side and never read back. */
export interface CloudProvider {
  id: string;
  name: string;
  /** "openai": the chat-completions API that most providers offer. "anthropic": Anthropic's messages API. */
  kind: "openai" | "anthropic";
  baseUrl: string;
  auth: "bearer" | "anthropic" | "none";
  /** The model names offered in the model list. */
  models: string[];
  /** The context window to plan with, in tokens. */
  ctx: number;
  /** The person confirmed that using this provider sends their messages (and, in coding mode, the files the agent reads) to it. */
  confirmed?: boolean;
}

/** Starting points. The model names are examples that were right when written: providers rename and retire models, so every list can be edited. */
export const PRESETS: Array<Omit<CloudProvider, "confirmed"> & { keyHint: string }> = [
  { id: "anthropic", name: "Anthropic", kind: "anthropic", baseUrl: "https://api.anthropic.com/v1", auth: "anthropic", models: ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"], ctx: 200000, keyHint: "console.anthropic.com" },
  { id: "openai", name: "OpenAI", kind: "openai", baseUrl: "https://api.openai.com/v1", auth: "bearer", models: ["gpt-4o", "gpt-4o-mini"], ctx: 128000, keyHint: "platform.openai.com" },
  { id: "gemini", name: "Google Gemini", kind: "openai", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", auth: "bearer", models: ["gemini-2.0-flash"], ctx: 1000000, keyHint: "aistudio.google.com" },
  { id: "openrouter", name: "OpenRouter", kind: "openai", baseUrl: "https://openrouter.ai/api/v1", auth: "bearer", models: ["openai/gpt-4o-mini"], ctx: 128000, keyHint: "openrouter.ai" },
  { id: "deepseek", name: "DeepSeek", kind: "openai", baseUrl: "https://api.deepseek.com/v1", auth: "bearer", models: ["deepseek-chat", "deepseek-reasoner"], ctx: 64000, keyHint: "platform.deepseek.com" },
  { id: "groq", name: "Groq", kind: "openai", baseUrl: "https://api.groq.com/openai/v1", auth: "bearer", models: ["llama-3.3-70b-versatile"], ctx: 128000, keyHint: "console.groq.com" },
  { id: "mistral", name: "Mistral", kind: "openai", baseUrl: "https://api.mistral.ai/v1", auth: "bearer", models: ["mistral-large-latest"], ctx: 128000, keyHint: "console.mistral.ai" },
  { id: "together", name: "Together AI", kind: "openai", baseUrl: "https://api.together.xyz/v1", auth: "bearer", models: [], ctx: 128000, keyHint: "api.together.ai" },
  { id: "ollama", name: "Ollama (on a computer you run)", kind: "openai", baseUrl: "http://localhost:11434/v1", auth: "none", models: [], ctx: 32000, keyHint: "no key" },
  { id: "lmstudio", name: "LM Studio (on a computer you run)", kind: "openai", baseUrl: "http://localhost:1234/v1", auth: "none", models: [], ctx: 32000, keyHint: "no key" },
  { id: "custom", name: "Other (OpenAI-compatible)", kind: "openai", baseUrl: "https://", auth: "bearer", models: [], ctx: 128000, keyHint: "your provider" },
];

const KEY = "clutor.cloud.v1";
export function loadProviders(): CloudProvider[] { try { const r = JSON.parse(localStorage.getItem(KEY) ?? "[]") as CloudProvider[]; return Array.isArray(r) ? r.filter((p) => p && typeof p.id === "string") : []; } catch { return []; } }
export function saveProviders(list: CloudProvider[]): void { try { localStorage.setItem(KEY, JSON.stringify(list)); } catch { /* not remembered */ } }

export const cloudPath = (providerId: string, model: string): string => `cloud:${providerId}:${model}`;
export function parseCloudPath(path: string): { providerId: string; model: string } | undefined {
  if (!path.startsWith("cloud:")) return undefined;
  const rest = path.slice(6), i = rest.indexOf(":");
  return i > 0 ? { providerId: rest.slice(0, i), model: rest.slice(i + 1) } : undefined;
}

/** The chat address of a provider: its base address with the path of the API that its kind uses. */
export function endpoint(p: Pick<CloudProvider, "kind" | "baseUrl">): string {
  const base = p.baseUrl.trim().replace(/\/+$/, "");
  return `${base}${p.kind === "anthropic" ? "/messages" : "/chat/completions"}`;
}

/** What to tell the person about a failed request, in words rather than a status code. */
export function explainError(status: number, raw: string, provider: string): string {
  const code = status || Number(/^(\d{3}):/.exec(raw)?.[1] ?? 0);
  const detail = raw.replace(/^\d{3}:\s*/, "").slice(0, 300);
  if (code === 401 || code === 403) return `${provider} rejected the API key (${code}). Check the key in Models > Cloud.`;
  if (code === 404) return `${provider} does not know that model or address (404). Check the model name. ${detail}`.trim();
  if (code === 429) return `${provider} says you are sending too much or are out of credit (429). Wait a moment, or check your plan. ${detail}`.trim();
  if (code >= 500) return `${provider} had a problem on its side (${code}). Try again in a moment.`;
  if (code === 400) return `${provider} refused the request (400): ${detail}`;
  return raw.startsWith("could not reach") || raw.startsWith("the connection") || raw.startsWith("the provider stopped") ? `${raw} Check your internet connection.` : raw || "The request failed.";
}
