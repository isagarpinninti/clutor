import { spawnSync } from "node:child_process";
import { existsSync, realpathSync } from "node:fs";
import { homedir } from "node:os";

/**
 * An operating-system sandbox around the commands the agent runs, on top of the command policy and the scratch copy.
 *
 * What it does, for the command and everything it starts:
 *   - writes are allowed only inside the scratch folder (and the system's temporary folders, which tools need);
 *   - the places where secrets live are unreadable: ~/.ssh, ~/.aws, ~/.gnupg, ~/.kube, ~/.config, ~/.npmrc, the keychain folders, browser profiles;
 *   - the network is off, except for a step the policy already marks as downloading (an install).
 *
 * macOS uses `sandbox-exec` (built in); Linux uses `bwrap` (bubblewrap) when it is installed. Windows has no equivalent here.
 * When no sandbox is available the command still runs under the other protections and `sandboxMode()` says "none", so the app can show that.
 * `CLUTOR_SANDBOX=off` turns it off (for a tool that does not work inside it). It is not a guarantee: a kernel bug or a sandbox-exec
 * profile mistake would break it, and it does not limit CPU, memory or disk use. Check `sandboxMode()` rather than assuming.
 */

export type SandboxMode = "seatbelt" | "bubblewrap" | "none";

/** Folders under the home directory that hold credentials, tokens or private browsing data. */
export const SECRET_DIRS = [".ssh", ".aws", ".gnupg", ".kube", ".config", ".docker", ".azure", ".npmrc", ".netrc", ".pypirc", ".git-credentials", ".cargo/credentials.toml", "Library/Keychains", "Library/Application Support/Google/Chrome", "Library/Application Support/Firefox", "Library/Safari", "Library/Cookies"];

let cached: SandboxMode | undefined;
export function sandboxMode(): SandboxMode {
  if (process.env["CLUTOR_SANDBOX"] === "off") return "none";
  if (cached) return cached;
  cached = "none";
  if (process.platform === "darwin" && existsSync("/usr/bin/sandbox-exec")) {
    // Prove it works on this machine (a restricted environment can forbid sandbox-exec itself) by running a harmless command in a profile.
    const r = spawnSync("/usr/bin/sandbox-exec", ["-p", "(version 1)(allow default)", "/usr/bin/true"], { timeout: 5000 });
    if (r.status === 0) cached = "seatbelt";
  } else if (process.platform === "linux") {
    const r = spawnSync("bwrap", ["--ro-bind", "/", "/", "--dev", "/dev", "true"], { timeout: 5000 });
    if (r.status === 0) cached = "bubblewrap";
  }
  return cached;
}

const q = (p: string): string => `"${p.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;

/** The seatbelt profile text for a scratch folder. Exported so it can be tested and read. */
export function seatbeltProfile(scratch: string, network: boolean, home = homedir()): string {
  const real = (p: string): string => { try { return realpathSync(p); } catch { return p; } };
  const lines = [
    "(version 1)",
    "(allow default)",
    // later rules win in SBPL: deny everything that writes, then allow the few places a build needs
    "(deny file-write*)",
    `(allow file-write* (subpath ${q(real(scratch))}) (subpath "/private/tmp") (subpath "/private/var/folders") (subpath "/dev"))`,
    ...SECRET_DIRS.map((d) => `(deny file-read* (subpath ${q(`${home}/${d}`)}))`),
  ];
  if (!network) lines.push("(deny network*)", '(allow network* (local ip "localhost:*"))');
  return lines.join("\n");
}

/** The command wrapped for the sandbox, or unchanged when none is available. */
export function sandboxed(argv: string[], scratch: string, network: boolean): string[] {
  const mode = sandboxMode();
  if (mode === "seatbelt") return ["/usr/bin/sandbox-exec", "-p", seatbeltProfile(scratch, network), ...argv];
  if (mode === "bubblewrap") {
    const home = homedir();
    const args = ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--bind", scratch, scratch]   // the rest of the file system stays read-only; TMPDIR is the scratch folder, so tools need no other writable place;
    for (const d of SECRET_DIRS) if (existsSync(`${home}/${d}`)) args.push("--tmpfs", `${home}/${d}`);
    if (!network) args.push("--unshare-net");
    return ["bwrap", ...args, "--chdir", scratch, ...argv];
  }
  return argv;
}
