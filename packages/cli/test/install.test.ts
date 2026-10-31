import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { buildEngine, enginePlatform, findEngine, installPrebuilt } from "../src/engine";

// A release served from a local HTTP server: the real engine binary packed the way .github/workflows/release-binaries.yml packs it.
let dir: string;
let server: Server;
let base: string;
let tarball: Buffer;
const platform = enginePlatform();
const asset = `clutor-engine-${platform}.tar.gz`;
const files = new Map<string, Buffer | string>();

beforeAll(async () => {
  dir = mkdtempSync(join(tmpdir(), "clutor-install-"));
  const bin = findEngine() ?? (await buildEngine());
  const stage = join(dir, "stage", `clutor-engine-${platform}`);
  mkdirSync(stage, { recursive: true });
  copyFileSync(bin, join(stage, "clutor-engine"));
  execFileSync("tar", ["-C", join(dir, "stage"), "-czf", join(dir, asset), `clutor-engine-${platform}`]);
  tarball = readFileSync(join(dir, asset));
  files.set(`/${asset}`, tarball);
  files.set("/SHA256SUMS", `${createHash("sha256").update(tarball).digest("hex")}  ${asset}\n`);
  server = createServer((req, res) => {
    const body = files.get(req.url ?? "");
    if (body === undefined) { res.statusCode = 404; res.end(); return; }
    res.end(body);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  process.env["CLUTOR_RELEASE_BASE"] = base;
}, 180_000);
afterAll(async () => { delete process.env["CLUTOR_RELEASE_BASE"]; server.close(); rmSync(dir, { recursive: true, force: true }); });

describe("clutor engine install", () => {
  it("has a published platform name for the machine running the tests", () => {
    expect(platform).toMatch(/^(darwin|linux)-(arm64|x64)$/);
  });

  it("downloads, verifies and installs a runnable engine", async () => {
    const dest = join(dir, "dest");
    const out = await installPrebuilt("9.9.9", dest);
    expect(out).toBe(join(dest, "clutor-engine"));
    expect(execFileSync(out, ["--version"]).toString()).toMatch(/^\d+\.\d+\.\d+/);
  }, 60_000);

  it("refuses a tarball whose checksum does not match, and installs nothing", async () => {
    const good = files.get(`/${asset}`);
    files.set(`/${asset}`, Buffer.concat([tarball, Buffer.from("tampered")]));
    const dest = join(dir, "dest-bad");
    await expect(installPrebuilt("9.9.9", dest)).rejects.toThrow(/checksum mismatch/);
    expect(() => readFileSync(join(dest, "clutor-engine"))).toThrow();
    files.set(`/${asset}`, good as Buffer);
  }, 60_000);

  it("says so when the checksum file does not list this platform, or the release is missing", async () => {
    const sums = files.get("/SHA256SUMS");
    files.set("/SHA256SUMS", "0000  some-other-file.tar.gz\n");
    await expect(installPrebuilt("9.9.9", join(dir, "d2"))).rejects.toThrow(/lists no/);
    files.delete("/SHA256SUMS");
    await expect(installPrebuilt("9.9.9", join(dir, "d3"))).rejects.toThrow(/404|no prebuilt engine is published/);
    files.set("/SHA256SUMS", sums as string);
  }, 60_000);

  it("will not download over plain http from anywhere but this machine", async () => {
    process.env["CLUTOR_RELEASE_BASE"] = "http://example.com/release";
    await expect(installPrebuilt("9.9.9", join(dir, "d4"))).rejects.toThrow(/insecure/);
    process.env["CLUTOR_RELEASE_BASE"] = base;
  });
});
