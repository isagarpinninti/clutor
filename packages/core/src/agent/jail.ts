/**
 * The path jail: every path the agent names must stay inside the project folder, and a few places are off limits even inside it.
 * This is pure string logic, so it can be tested exhaustively; the Node layer adds the check that a path is not a symlink out.
 */

export class JailError extends Error {
  constructor(message: string, readonly code: "absolute" | "escape" | "protected" | "invalid") {
    super(message);
    this.name = "JailError";
  }
}

/** Places the agent may read but never write: version control, secrets, its own state. */
const PROTECTED_WRITE = [/^\.git(\/|$)/, /^\.clutor(\/|$)/, /(^|\/)\.env($|\.)/, /(^|\/)id_(rsa|ed25519|ecdsa)(\.pub)?$/, /(^|\/)\.ssh(\/|$)/, /(^|\/)\.aws(\/|$)/, /\.(pem|key|p12|pfx)$/i, /(^|\/)\.npmrc$/, /(^|\/)\.netrc$/];
/** Places the agent may not even read. */
const PROTECTED_READ = [/(^|\/)\.env($|\.)/, /(^|\/)id_(rsa|ed25519|ecdsa)$/, /(^|\/)\.ssh(\/|$)/, /(^|\/)\.aws(\/|$)/, /\.(pem|key|p12|pfx)$/i, /(^|\/)\.netrc$/, /^\.clutor\/keys\//];

/** Returns the cleaned project-relative path (forward slashes, no `.` or `..` segments), or throws a JailError. */
export function normalizeRelPath(input: string): string {
  if (typeof input !== "string" || input.length === 0 || input.length > 1024) throw new JailError("a path is required (at most 1024 characters)", "invalid");
  if (input.includes("\0")) throw new JailError("a path may not contain a NUL byte", "invalid");
  // Treat both separators the same, so a Windows-style path cannot sneak through on a POSIX host or the other way round.
  const p = input.replace(/\\/g, "/");
  if (p.startsWith("/") || /^[A-Za-z]:/.test(p) || p.startsWith("~") || p.startsWith("//")) throw new JailError(`"${input}" is an absolute path; use a path relative to the project`, "absolute");
  const out: string[] = [];
  for (const seg of p.split("/")) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      if (out.length === 0) throw new JailError(`"${input}" leaves the project folder`, "escape");
      out.pop();
      continue;
    }
    if (/^\s|\s$/.test(seg) && seg.trim() === "") throw new JailError("a path segment may not be only spaces", "invalid");
    out.push(seg);
  }
  if (out.length === 0) throw new JailError(`"${input}" is the project folder itself`, "invalid");
  return out.join("/");
}

export function checkRead(rel: string): string {
  const p = normalizeRelPath(rel);
  if (PROTECTED_READ.some((r) => r.test(p))) throw new JailError(`"${p}" may hold secrets and is not readable by the agent`, "protected");
  return p;
}

export function checkWrite(rel: string): string {
  const p = normalizeRelPath(rel);
  if (PROTECTED_WRITE.some((r) => r.test(p))) throw new JailError(`"${p}" is protected and cannot be changed by the agent`, "protected");
  if (/(^|\/)node_modules(\/|$)/.test(p)) throw new JailError("node_modules is managed by the package manager, not the agent", "protected");
  return p;
}
