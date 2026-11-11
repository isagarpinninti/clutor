import { mkdtempSync, existsSync, rmSync, readFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { sandboxMode, sandboxed, seatbeltProfile, SECRET_DIRS } from "../src/node/osandbox";

const run = (argv: string[]) => spawnSync(argv[0] as string, argv.slice(1), { encoding: "utf8", timeout: 20_000 });
const on = sandboxMode() === "seatbelt";

describe("sandbox profile", () => {
  it("denies network unless asked, and names every secret folder", () => {
    const p = seatbeltProfile("/tmp/x", false, "/home/u");
    expect(p).toContain("(deny network*)");
    for (const d of SECRET_DIRS) expect(p).toContain(`/home/u/${d}`);
    expect(seatbeltProfile("/tmp/x", true, "/home/u")).not.toContain("(deny network*)");
  });
  it("leaves the command alone when no sandbox is available", () => {
    if (sandboxMode() === "none") expect(sandboxed(["ls"], "/tmp", false)).toEqual(["ls"]);
  });
});

describe.skipIf(!on)("sandbox on macOS (runs the real thing)", () => {
  const scratch = mkdtempSync(join(tmpdir(), "clutor-sbx-"));
  it("lets a command write in the scratch folder", () => {
    const r = run(sandboxed(["/bin/sh", "-c", `echo ok > ${scratch}/a.txt`], scratch, false));
    expect(r.status).toBe(0);
    expect(readFileSync(join(scratch, "a.txt"), "utf8").trim()).toBe("ok");
  });
  it("blocks a write outside it", () => {
    const target = join(homedir(), `.clutor-sandbox-test-${process.pid}`);
    const r = run(sandboxed(["/bin/sh", "-c", `echo no > ${target}`], scratch, false));
    expect(r.status).not.toBe(0);
    expect(existsSync(target)).toBe(false);
    rmSync(target, { force: true });
  });
  it("blocks reading a secret folder", () => {
    const ssh = join(homedir(), ".ssh");
    if (!existsSync(ssh)) return;
    const r = run(sandboxed(["/bin/ls", ssh], scratch, false));
    expect(r.status).not.toBe(0);
  });
  it("blocks the network, and allows it for an install step", () => {
    const probe = ["/usr/bin/curl", "-s", "-m", "5", "-o", "/dev/null", "https://example.com"];
    const blocked = run(sandboxed(probe, scratch, false));
    expect(blocked.status).not.toBe(0);
    // the open case depends on this machine being online; only check that the profile does not itself forbid it
    expect(seatbeltProfile(scratch, true)).not.toContain("(deny network*)");
  });
});
