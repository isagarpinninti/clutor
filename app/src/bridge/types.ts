import type { NativeReady, Transport } from "@engine/session";

export interface SystemInfo { os: string; arch: string; ramBytes: number; cores: number; modelsDir: string }
export interface FileInfo { path: string; name: string; bytes: number }
export interface ModelInfo { path: string; name: string; bytes: number; /** Set for a cloud model: the provider's name. The path is then `cloud:<provider id>:<model>`. */ cloud?: string }
export interface CatalogItem { id: string; name: string; bytes: number; description: string }
export interface DownloadProgress { id: string; done: number; total: number; state: "downloading" | "verifying" | "done" | "error"; error?: string }

/** A running engine: the events and operations of `clutor-engine chat`, and the first (ready) event. */
export interface OpenEngine { transport: Transport; ready: NativeReady }

export interface SearchHit { title: string; url: string; snippet: string }
export interface WebPage { url: string; title: string; text: string; truncated: boolean; status: number }
/** Web access, done by the shell: addresses on this computer or its network are refused there. */
export interface WebApi { search(query: string): Promise<SearchHit[]>; fetch(url: string): Promise<WebPage>; openExternal(url: string): Promise<void> }

export interface RemoteDevice { id: string; name: string; public: string; created: number; lastSeen: number }
export interface RemoteStatus { on: boolean; port?: number; addresses: string[]; hostId: string; devices: RemoteDevice[]; clients: number; offerOpen: boolean; awake: boolean }
export interface PairingOffer { qr: string; expiresIn: number; addresses: string[] }
/** Remote control of this computer from a paired phone (see docs/REMOTE.md). */
export interface RemoteApi {
  status(): Promise<RemoteStatus>;
  start(keepAwake: boolean): Promise<RemoteStatus>;
  stop(): Promise<RemoteStatus>;
  setAwake(on: boolean): Promise<RemoteStatus>;
  pairStart(): Promise<PairingOffer>;
  pairCancel(): Promise<void>;
  revoke(id: string): Promise<RemoteStatus>;
  send(client: number, text: string): Promise<boolean>;
  broadcast(text: string): Promise<void>;
  kick(client: number): Promise<void>;
  log(line: string): Promise<void>;
  readLog(max: number): Promise<string[]>;
  /** Phones connecting, sending and leaving. Each returns a function that stops listening. */
  onOpen(cb: (client: number, device: RemoteDevice) => void): Promise<() => void>;
  onMessage(cb: (client: number, text: string) => void): Promise<() => void>;
  onClose(cb: (client: number) => void): Promise<() => void>;
}

export interface DiskSpeed { sequentialBytesPerSec: number; scatteredBytesPerSec: number; seconds: number }
/** Looking at a model file without loading it. */
export interface InspectApi {
  size(path: string): Promise<number>;
  read(path: string, offset: number, length: number): Promise<Uint8Array>;
  /** Times reading the file (sequential and scattered), about `seconds` in all. */
  diskBench(path: string, seconds: number): Promise<DiskSpeed>;
}

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
  /** Runs a command typed by the user in the project folder (their own shell, no sandbox); output arrives line by line. */
  termRun(id: string, root: string, command: string, onOut: (text: string) => void, onExit: (code: number | null) => void): Promise<void>;
  termKill(id: string): Promise<void>;
  /** The folder's real path if it lies inside one of the allowed folders (symbolic links followed); an error otherwise. */
  checkRoot(root: string, allowed: string[]): Promise<string>;
  /** Creates a new empty project folder inside `parent` and returns its path. */
  newProject(parent: string, name: string): Promise<string>;
  /** The folders directly inside `root`. */
  dirs(root: string): Promise<string[]>;
}

/** Cloud models: requests go through the app's native side, which holds the API keys (the page never sees a key). */
export interface CloudApi {
  keySet(provider: string, key: string): Promise<void>;
  keyHas(provider: string): Promise<boolean>;
  keyClear(provider: string): Promise<void>;
  /** POSTs `body` and calls `onChunk` with the response text as it arrives; resolves when the response ends (or fails: `error` is set). */
  request(o: { id: string; provider: string; url: string; auth: "bearer" | "anthropic" | "none"; body: string }, onChunk: (text: string) => void): Promise<{ status: number; error?: string }>;
  cancel(id: string): Promise<void>;
}

/** Image generation inside the app (see src-tauri/src/image.rs). */
export interface ImageRequest { diffusion: string; text_encoder: string; vae: string; prompt: string; negative?: string; width: number; height: number; steps: number; cfg: number; seed: number; turbo: boolean; low_memory: boolean }
export interface ImageProgress { step: number; steps: number; seconds: number }
export interface ImageApi {
  /** True when this build includes the image engine. */
  available(): Promise<boolean>;
  /** Makes one picture and saves it; resolves with where. Rejects with the reason (or "stopped"). */
  generate(req: ImageRequest, onProgress: (p: ImageProgress) => void): Promise<{ path: string; seconds: number }>;
  cancel(): Promise<void>;
  /** The picture as base64 PNG. */
  read(path: string): Promise<string>;
  /** Asks where to save a copy and saves it; false if the person cancelled. */
  saveAs(path: string, suggestedName: string): Promise<boolean>;
}

/** Everything the interface needs from the machine it runs on. The real app implements it with Tauri commands; a browser preview with a mock. */
export interface Bridge {
  readonly native: boolean;
  readonly host: AgentHost;
  readonly web: WebApi;
  readonly remote: RemoteApi;
  readonly inspect: InspectApi;
  readonly cloud: CloudApi;
  readonly image: ImageApi;
  system(): Promise<SystemInfo>;
  /** The models in the app's folder, the command line's, and any folders you chose. */
  listModels(folders?: string[]): Promise<ModelInfo[]>;
  /** Every model-like file (GGUF and safetensors) in the folders, with its size: what the library reads headers from. */
  scanFiles(folders?: string[]): Promise<FileInfo[]>;
  catalog(): Promise<CatalogItem[]>;
  download(id: string, onProgress: (p: DownloadProgress) => void): Promise<void>;
  openEngine(path: string, ctx: number, lock: boolean, kv8?: boolean): Promise<OpenEngine>;
  revealModels(): Promise<void>;
  /** A desktop notification (asks permission the first time). Silent if the system refuses. */
  notify(title: string, body: string): Promise<void>;
}
