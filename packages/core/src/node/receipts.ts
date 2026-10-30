import { appendFile, chmod, mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { GENESIS_TAG, generateReceiptSigner, keyThumbprint, verifyReceiptChain, type PublicJwk, type Receipt, type ReceiptSigner } from "../agent/receipt";

const DIR = ".clutor";
const KEYS = join(DIR, "keys");
const PRIVATE = "receipt-p256.jwk.json";
const PUBLIC = "receipt-p256.pub.jwk.json";

/** The receipt signing key pair, created on first use. The private key is readable only by the user; the public key is meant to be shared. */
export async function loadOrCreateReceiptSigner(root: string): Promise<ReceiptSigner> {
  try {
    const privateJwk = JSON.parse(await readFile(join(root, KEYS, PRIVATE), "utf8")) as JsonWebKey;
    const publicJwk = JSON.parse(await readFile(join(root, KEYS, PUBLIC), "utf8")) as PublicJwk;
    return { privateJwk, publicJwk };
  } catch { /* create it */ }
  await mkdir(join(root, KEYS), { recursive: true });
  const signer = await generateReceiptSigner();
  await writeFile(join(root, KEYS, PRIVATE), JSON.stringify(signer.privateJwk), { mode: 0o600 });
  await chmod(join(root, KEYS, PRIVATE), 0o600);
  await writeFile(join(root, KEYS, PUBLIC), `${JSON.stringify(signer.publicJwk, null, 2)}\n`);
  return signer;
}

/** The thumbprint of this project's receipt key: record it somewhere you trust, and others can check receipts against it. */
export async function receiptKeyId(root: string): Promise<string> {
  return keyThumbprint((await loadOrCreateReceiptSigner(root)).publicJwk);
}

export async function readReceipts(root: string): Promise<Receipt[]> {
  try {
    return (await readFile(join(root, DIR, "receipts.jsonl"), "utf8")).split("\n").filter(Boolean).map((l) => JSON.parse(l) as Receipt);
  } catch { return []; }
}

export async function appendReceipt(root: string, r: Receipt): Promise<void> {
  await mkdir(join(root, DIR), { recursive: true });
  await appendFile(join(root, DIR, "receipts.jsonl"), `${JSON.stringify(r)}\n`);
}

export async function lastReceiptTag(root: string): Promise<string> {
  const all = await readReceipts(root);
  return all.at(-1)?.tag ?? GENESIS_TAG;
}

/** Verifies the log's hash chain and every ECDSA signature. Needs no secret, so it works on a copy of the log. */
export async function verifyReceiptLog(root: string, opts: { trustedKeyIds?: string[] } = {}): Promise<{ ok: true; count: number; keyIds: string[] } | { ok: false; index: number; reason: string }> {
  const all = await readReceipts(root);
  const r = await verifyReceiptChain(all, opts);
  return r.ok ? { ok: true, count: all.length, keyIds: r.keyIds } : r;
}
