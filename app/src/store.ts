import type { ModelInfo } from "./bridge";

export interface MsgStats { tokens: number; tps?: number; promptTokens?: number; ms: number; cache?: { hitRate: number; gbRead: number } }
export interface Source { title: string; url: string }
export interface Msg { id: string; role: "user" | "assistant"; content: string; sources?: Source[]; files?: Array<{ name: string; bytes: number; text: string; truncated: boolean }>; web?: { query: string; state: "searching" | "done" | "failed"; error?: string }; jobId?: string; reasoning?: string; stats?: MsgStats; error?: string; interrupted?: boolean }
export interface Conversation { id: string; title: string; updated: number; messages: Msg[] }

export interface Settings {
  temperature: number;
  thinking: boolean;
  showReasoning: boolean;
  system: string;
  ctx: number;
  /** The longest answer a background job may write, in tokens. */
  jobTokens: number;
  lock: boolean;
  /** Chat looks things up on the web before answering. */
  browse: boolean;
  /** The model used last time: it is selected again when the app opens (and loaded when you first need it). */
  model?: ModelInfo;
  /** True (the default) when Chat and Code use the same model; false lets each have its own. */
  sameModel?: boolean;
  /** The model for each mode, used only when `sameModel` is false. */
  models?: { chat?: ModelInfo; code?: ModelInfo };
  /** A folder you chose to look for models in. */
  modelsDir?: string;
  /** Remote control from a paired phone. Off until you turn it on in Settings > Remote. */
  remote?: { projectsRoot?: string; allowedRoots?: string[]; keepAwake?: boolean };
  /** What to call you in greetings, asked on first launch. */
  userName?: string;
  /** The welcome screen has been completed or skipped. */
  onboarded?: boolean;
  /** Default permission mode for the agent. */
  permission: "ask" | "auto" | "plan";
  /** Most steps one agent task may take. */
  maxSteps: number;
  textSize: "small" | "medium" | "large";
  notifyJobs: boolean;
  notifyAgent: boolean;
  /** Normal keeps the agent's steps closed; Verbose opens each to show its output. */
  transcript: "normal" | "verbose";
  /** Suggestion chips on the chat home screen. */
  suggestions: boolean;
}

export const DEFAULT_SYSTEM = "You are a helpful assistant running on the user's own device through Clutor. Answer directly and concisely. Use Markdown only where it helps.";
export const DEFAULT_SETTINGS: Settings = { temperature: 0.7, thinking: false, showReasoning: false, system: DEFAULT_SYSTEM, ctx: 4096, jobTokens: 1024, lock: true, browse: true, permission: "ask", maxSteps: 20, textSize: "medium", notifyJobs: true, notifyAgent: true, transcript: "normal", suggestions: true };

const CONV = "clutor.conversations.v1";
const SET = "clutor.settings.v1";

function read<T>(key: string, fallback: T): T {
  try { const raw = localStorage.getItem(key); return raw ? (JSON.parse(raw) as T) : fallback; } catch { return fallback; }
}
function write(key: string, value: unknown): void {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage full or blocked: the app still works, it just forgets */ }
}

export const loadConversations = (): Conversation[] => read<Conversation[]>(CONV, []);
export const saveConversations = (c: Conversation[]): void => write(CONV, c.slice(0, 200));
export const loadSettings = (): Settings => ({ ...DEFAULT_SETTINGS, ...read<Partial<Settings>>(SET, {}) });
export const saveSettings = (s: Settings): void => write(SET, s);

export const uid = (): string => (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`);

export function titleFrom(text: string): string {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 48 ? `${t.slice(0, 47)}…` : t || "New chat";
}

export function fmtBytes(n: number): string {
  if (n >= 2 ** 30) return `${(n / 2 ** 30).toFixed(1)} GB`;
  if (n >= 2 ** 20) return `${(n / 2 ** 20).toFixed(0)} MB`;
  return `${Math.round(n / 1024)} KB`;
}

/** Whether to pin the model in memory. Pinning makes a model that fits much faster, but reading and pinning several GB at once stalls the whole computer, so a model over 40% of RAM is not pinned. */
export function shouldPin(wanted: boolean, modelBytes: number, ramBytes: number | undefined): boolean {
  return wanted && (ramBytes === undefined || ramBytes <= 0 || modelBytes <= ramBytes * 0.4);
}

/** The model a mode should run: the shared one, or, when the modes were given their own, the one chosen for that mode (falling back to the shared one). */
export function modelForMode(settings: Pick<Settings, "sameModel" | "models">, shared: ModelInfo | undefined, mode: "chat" | "code"): ModelInfo | undefined {
  return (settings.sameModel === false ? settings.models?.[mode] : undefined) ?? shared;
}
