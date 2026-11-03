import type { NativeReady, Transport } from "@engine/session";

export interface SystemInfo { os: string; arch: string; ramBytes: number; cores: number; modelsDir: string }
export interface ModelInfo { path: string; name: string; bytes: number }
export interface CatalogItem { id: string; name: string; bytes: number; description: string }
export interface DownloadProgress { id: string; done: number; total: number; state: "downloading" | "verifying" | "done" | "error"; error?: string }

/** A running engine: the events and operations of `clutor-engine chat`, and the first (ready) event. */
export interface OpenEngine { transport: Transport; ready: NativeReady }

export interface RunOutcome { exit_code: number | null; output: string; timed_out: boolean; duration_ms: number }
export interface McpLaunch { command: string; args: string[]; env: Record<string, string>; cwd?: string }

/** The agent's view of the machine: one project folder, and MCP servers over standard input and output. */
export interface AgentHost {
  /** Asks the user for a folder (null if they cancel). */
  pickFolder(): Promise<string | null>;
  list(root: string): Promise<string[]>;
  read(root: string, path: string): Promise<string | null>;
  /** Writes the file, or deletes it when content is null. */
  write(root: string, path: string, content: string | null): Promise<void>;
  /** Runs the steps in a scratch copy of the project that includes `overlay`; the real folder is not touched. */
  run(root: string, overlay: Array<{ path: string; content: string | null }>, steps: string[][], timeoutMs: number): Promise<RunOutcome>;
  mcpStart(id: string, l: McpLaunch, onLine: (line: string) => void, onExit: () => void, onLog: (line: string) => void): Promise<void>;
  mcpSend(id: string, line: string): Promise<void>;
  mcpStop(id: string): Promise<void>;
}

/** Everything the interface needs from the machine it runs on. The real app implements it with Tauri commands; a browser preview with a mock. */
export interface Bridge {
  readonly native: boolean;
  readonly host: AgentHost;
  system(): Promise<SystemInfo>;
  listModels(): Promise<ModelInfo[]>;
  catalog(): Promise<CatalogItem[]>;
  download(id: string, onProgress: (p: DownloadProgress) => void): Promise<void>;
  openEngine(path: string, ctx: number, lock: boolean): Promise<OpenEngine>;
  revealModels(): Promise<void>;
  /** A desktop notification (asks permission the first time). Silent if the system refuses. */
  notify(title: string, body: string): Promise<void>;
}
