import { parsePlan, type PlanStep } from "./sessions";

/**
 * What the model is writing right now, read from its reply while it is still being written. In the agent's JSON protocol a reply is one JSON object, so a half-written one is
 * still readable: the tool being called, the file it is for, and the text so far (the file's content, the plan, the summary).
 */
export interface Live {
  /** Words before the JSON, if any. */
  prose: string;
  tool?: string;
  path?: string;
  /** The content of a file being written, the steps of a plan, or a summary, as far as it has come. */
  text?: string;
  plan?: PlanStep[];
  /** Characters of the reply so far. */
  chars: number;
  /** One line saying what is happening. */
  label: string;
}

function unescapeJson(s: string): string {
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i] as string;
    if (c !== "\\") { out += c; continue; }
    const n = s[++i];
    if (n === undefined) break;   // a backslash at the very end: the rest has not come yet
    out += n === "n" ? "\n" : n === "t" ? "\t" : n === "r" ? "\r" : n === "u" ? ((): string => { const h = s.slice(i + 1, i + 5); if (/^[0-9a-fA-F]{4}$/.test(h)) { i += 4; return String.fromCharCode(parseInt(h, 16)); } return ""; })() : n;
  }
  return out;
}

/** The string value of `key`, as far as it has been written (it need not be closed yet). */
function partialString(json: string, key: string): string | undefined {
  const m = new RegExp(`"${key}"\\s*:\\s*"`).exec(json);
  if (!m) return undefined;
  let i = m.index + m[0].length, end = json.length;
  for (let j = i; j < json.length; j++) { if (json[j] === "\\") { j++; continue; } if (json[j] === '"') { end = j; break; } }
  return unescapeJson(json.slice(i, end));
}

export function parseLive(reply: string): Live {
  const body = reply.replace(/<think>[\s\S]*?(<\/think>|$)/g, "");
  const at = body.indexOf("{");
  const prose = (at < 0 ? body : body.slice(0, at)).replace(/```(?:json)?/g, "").trim();
  const json = at < 0 ? "" : body.slice(at);
  const tool = /"(?:tool|name)"\s*:\s*"([a-z_]+)"/.exec(json)?.[1];
  const path = partialString(json, "path");
  const base: Live = { prose, chars: reply.length, label: "Thinking" };
  if (!tool) return { ...base, label: prose ? "Thinking it through" : json ? "Starting a step" : "Thinking" };
  const key = tool === "write_file" ? "content" : tool === "update_plan" ? "steps" : tool === "finish" ? "summary" : tool === "edit_file" ? "new_text" : undefined;
  const text = key ? partialString(json, key) : undefined;
  const command = tool === "run_command" ? partialString(json, "command") : undefined;
  const label = ({
    write_file: `Writing ${path ?? "a file"}`, edit_file: `Editing ${path ?? "a file"}`, read_file: `Reading ${path ?? "a file"}`, delete_file: `Deleting ${path ?? "a file"}`,
    list_files: "Listing files", search: "Searching the project", repo_map: "Mapping the project", run_command: command ? `Running ${command.slice(0, 60)}` : "Running a command",
    update_plan: "Writing the plan", finish: "Writing the summary", web_search: "Searching the web", fetch_url: "Opening a page", use_tool: "Using a tool",
  } as Record<string, string>)[tool] ?? tool;
  return { ...base, tool, ...(path ? { path } : {}), ...(text !== undefined ? { text } : {}), ...(tool === "update_plan" && text ? { plan: parsePlan(text) } : {}), label };
}

/** The reply being written, per task, outside React state: it changes several times a second and only the progress panel needs it. */
type Listener = () => void;
const texts = new Map<string, string>();
const listeners = new Set<Listener>();
export const liveStore = {
  set(taskId: string, text: string): void { texts.set(taskId, text.length > 60_000 ? text.slice(-60_000) : text); for (const l of [...listeners]) l(); },
  clear(taskId: string): void { if (texts.delete(taskId)) for (const l of [...listeners]) l(); },
  get(taskId: string): string | undefined { return texts.get(taskId); },
  subscribe(l: Listener): () => void { listeners.add(l); return () => { listeners.delete(l); }; },
};
