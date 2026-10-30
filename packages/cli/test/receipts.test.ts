import { createPublicKey, verify } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { generateReceiptSigner, sealReceipt, type Receipt } from "@clutor/core";
import { ensureSlotDir, slotDir } from "../src";
import { cleanup, tmp } from "./helpers";

afterEach(cleanup);

/** The same canonical JSON the signer uses, written again here so the check does not share code with the thing it checks. */
const canonical = (v: unknown): string => {
  if (Array.isArray(v)) return `[${v.map(canonical).join(",")}]`;
  if (v && typeof v === "object") return `{${Object.keys(v as object).sort().map((k) => `${JSON.stringify(k)}:${canonical((v as Record<string, unknown>)[k])}`).join(",")}}`;
  return JSON.stringify(v);
};

describe("ensureSlotDir", () => {
  it("creates ~/.clutor/slots, owner-only, when it is missing (llama-server refuses --slot-save-path for a folder that does not exist)", () => {
    const home = tmp();
    const saved = { HOME: process.env["HOME"], USERPROFILE: process.env["USERPROFILE"] };
    process.env["HOME"] = home;
    process.env["USERPROFILE"] = home;
    try {
      expect(slotDir()).toBe(join(home, ".clutor", "slots"));
      expect(existsSync(slotDir())).toBe(false);
      expect(ensureSlotDir()).toBe(slotDir());
      expect(statSync(slotDir()).isDirectory()).toBe(true);
      if (process.platform !== "win32") expect(statSync(slotDir()).mode & 0o777).toBe(0o700);
      expect(ensureSlotDir()).toBe(slotDir()); // a second call is harmless
    } finally {
      for (const [k, v] of Object.entries(saved)) { if (v === undefined) delete process.env[k]; else process.env[k] = v; }
    }
  });
});

describe("receipts are ECDSA P-256 with SHA-256, checkable without Clutor's code", () => {
  const draft = { v: 2 as const, id: "rcpt_x", task: "t", model: "m", startedAt: "2026-01-01T00:00:00.000Z", finishedAt: "2026-01-01T00:00:01.000Z", status: "finished", files: [], commands: [], steps: 1, previous: "clt_genesis" };

  async function sealed(): Promise<Receipt> { return sealReceipt(await generateReceiptSigner(), draft); }

  it("verifies with crypto.verify('sha256', payload, publicKey, signature) from the embedded public key", async () => {
    const r = await sealed();
    expect(r.publicKey).toMatchObject({ kty: "EC", crv: "P-256" });
    const { tag, ...body } = r;
    const sig = Buffer.from(tag.slice(4), "base64url");
    const key = createPublicKey({ key: r.publicKey, format: "jwk" });
    expect(verify("sha256", Buffer.from(canonical(body)), { key, dsaEncoding: "ieee-p1363" }, sig)).toBe(true);
    expect(sig.length).toBe(64); // r || s, 32 bytes each
  });

  it("fails when one byte of the payload or of the signature changes", async () => {
    const r = await sealed();
    const { tag, ...body } = r;
    const key = createPublicKey({ key: r.publicKey, format: "jwk" });
    const sig = Buffer.from(tag.slice(4), "base64url");
    const payload = Buffer.from(canonical(body));
    const flippedPayload = Buffer.from(payload); flippedPayload[10] = (flippedPayload[10] as number) ^ 1;
    const flippedSig = Buffer.from(sig); flippedSig[5] = (flippedSig[5] as number) ^ 1;
    expect(verify("sha256", flippedPayload, { key, dsaEncoding: "ieee-p1363" }, sig)).toBe(false);
    expect(verify("sha256", payload, { key, dsaEncoding: "ieee-p1363" }, flippedSig)).toBe(false);
    expect(verify("sha256", payload, { key: createPublicKey({ key: (await generateReceiptSigner()).publicJwk, format: "jwk" }), dsaEncoding: "ieee-p1363" }, sig)).toBe(false);
  });
});

describe("the two command names", () => {
  it("package.json links both clutor and clt to the same file", () => {
    const pkg = JSON.parse(readFileSync(join(__dirname, "..", "package.json"), "utf8")) as { bin: Record<string, string> };
    expect(pkg.bin["clutor"]).toBe("dist/bin.js");
    expect(pkg.bin["clt"]).toBe("dist/bin.js");
  });
});
