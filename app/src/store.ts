export interface MsgStats { tokens: number; tps?: number; promptTokens?: number; ms: number; cache?: { hitRate: number; gbRead: number } }
export interface Source { title: string; url: string }
export interface Msg { id: string; role: "user" | "assistant"; content: string; sources?: Source[]; jobId?: string; reasoning?: string; stats?: MsgStats; error?: string; interrupted?: boolean }
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
  lastModel?: string;
}

export const DEFAULT_SYSTEM = "You are a helpful assistant running on the user's own device through Clutor. Answer directly and concisely. Use Markdown only where it helps.";
export const DEFAULT_SETTINGS: Settings = { temperature: 0.7, thinking: false, showReasoning: false, system: DEFAULT_SYSTEM, ctx: 4096, jobTokens: 1024, lock: true, browse: false };

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
