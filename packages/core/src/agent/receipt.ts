/**
 * Execution receipts: a tamper-evident record of what the agent did. Each receipt is signed with ECDSA P-256 (SHA-256) and carries
 * the signer's public key and its thumbprint (RFC 7638) inside the signed body, plus the previous receipt's tag, so deleting, editing
 * or reordering one breaks the chain. Anyone can verify a log with no secret. A receipt proves the log was not altered by someone
 * WITHOUT the private key; to know the key is YOURS, compare its key id with one you recorded (`trustedKeyIds`). It does not prove which
 * model ran or that the output is correct.
 */

export interface PublicJwk { kty: "EC"; crv: "P-256"; x: string; y: string }
export interface ReceiptSigner { privateJwk: JsonWebKey; publicJwk: PublicJwk }

export interface ReceiptBody {
  /** 2 is the original; 3 adds the optional fields below. Both verify. */
  v: 2 | 3;
  id: string;
  task: string;
  model: string;
  startedAt: string;
  finishedAt: string;
  status: string;
  files: Array<{ path: string; status: "added" | "modified" | "deleted"; beforeSha256: string | null; afterSha256: string | null }>;
  /** `network` (v3): the command was allowed to use the network (a package install the person approved); false means it ran with the network off where an OS sandbox exists. */
  commands: Array<{ command: string; exitCode: number | null; approved: boolean; network?: boolean }>;
  steps: number;
  /** v3: files the agent read with read_file and the read succeeded. Paths only, never contents. */
  filesRead?: string[];
  /** v3: tokens the model reported using for this run (0 when the server does not report usage). */
  tokens?: { prompt: number; completion: number };
  /** v3: wall-clock milliseconds from start to finish. */
  durationMs?: number;
  previous: string;
  /** Thumbprint of `publicKey`. */
  keyId: string;
  publicKey: PublicJwk;
}
export interface Receipt extends ReceiptBody {
  /** `clt_` + base64url of the 64-byte ECDSA signature (r || s) over the canonical JSON of the body. */
  tag: string;
}

/** What the caller supplies; `sealReceipt` adds the key fields. */
export type ReceiptDraft = Omit<ReceiptBody, "keyId" | "publicKey">;

export const GENESIS_TAG = "clt_genesis";
const ALG = { name: "ECDSA", namedCurve: "P-256" } as const;
const SIGN = { name: "ECDSA", hash: "SHA-256" } as const;

function canonical(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(canonical).join(",")}]`;
  if (v && typeof v === "object") return `{${Object.keys(v as object).sort().map((k) => `${JSON.stringify(k)}:${canonical((v as Record<string, unknown>)[k])}`).join(",")}}`;
  return JSON.stringify(v);
}
const b64url = (b: ArrayBuffer | Uint8Array): string => btoa(String.fromCharCode(...new Uint8Array(b instanceof Uint8Array ? b : b))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const unb64url = (s: string): Uint8Array<ArrayBuffer> => {
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
};

/** RFC 7638 JWK thumbprint (SHA-256 of the required members in lexicographic order), base64url. */
export async function keyThumbprint(k: PublicJwk): Promise<string> {
  const json = `{"crv":${JSON.stringify(k.crv)},"kty":${JSON.stringify(k.kty)},"x":${JSON.stringify(k.x)},"y":${JSON.stringify(k.y)}}`;
  return b64url(await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(json)));
}

export async function generateReceiptSigner(): Promise<ReceiptSigner> {
  const pair = await globalThis.crypto.subtle.generateKey(ALG, true, ["sign", "verify"]);
  const privateJwk = await globalThis.crypto.subtle.exportKey("jwk", pair.privateKey);
  const pub = await globalThis.crypto.subtle.exportKey("jwk", pair.publicKey);
  return { privateJwk, publicJwk: { kty: "EC", crv: "P-256", x: pub.x as string, y: pub.y as string } };
}

export async function sealReceipt(signer: ReceiptSigner, draft: ReceiptDraft): Promise<Receipt> {
  const body: ReceiptBody = { ...draft, keyId: await keyThumbprint(signer.publicJwk), publicKey: signer.publicJwk };
  const key = await globalThis.crypto.subtle.importKey("jwk", signer.privateJwk, ALG, false, ["sign"]);
  const sig = await globalThis.crypto.subtle.sign(SIGN, key, new TextEncoder().encode(canonical(body)));
  return { ...body, tag: `clt_${b64url(sig)}` };
}

export type ChainResult = { ok: true; keyIds: string[] } | { ok: false; index: number; reason: string };

/** Verifies one receipt's signature and key id (not its place in a chain). */
export async function verifyReceiptSignature(r: Receipt): Promise<{ ok: true } | { ok: false; reason: string }> {
  try {
    if (r.v !== 2 && r.v !== 3) return { ok: false, reason: "not a version 2 or 3 (ECDSA) receipt" };
    if (!r.tag.startsWith("clt_")) return { ok: false, reason: "the tag does not start with clt_" };
    if ((await keyThumbprint(r.publicKey)) !== r.keyId) return { ok: false, reason: "the key id does not match the embedded public key" };
    const { tag, ...body } = r;
    const key = await globalThis.crypto.subtle.importKey("jwk", { ...r.publicKey, ext: true }, ALG, false, ["verify"]);
    const good = await globalThis.crypto.subtle.verify(SIGN, key, unb64url(tag.slice(4)), new TextEncoder().encode(canonical(body)));
    return good ? { ok: true } : { ok: false, reason: "its contents do not match its signature" };
  } catch {
    return { ok: false, reason: "it is malformed" };
  }
}

/**
 * Checks a list of receipts, oldest first: every signature is valid and every `previous` is the tag before it.
 * With `trustedKeyIds`, receipts signed by any other key are refused.
 */
export async function verifyReceiptChain(receipts: Receipt[], opts: { trustedKeyIds?: string[] } = {}): Promise<ChainResult> {
  let prev = GENESIS_TAG;
  const keyIds = new Set<string>();
  for (let i = 0; i < receipts.length; i++) {
    const r = receipts[i] as Receipt;
    if (r.previous !== prev) return { ok: false, index: i, reason: "does not follow the receipt before it (one was removed, reordered or edited)" };
    const s = await verifyReceiptSignature(r);
    if (!s.ok) return { ok: false, index: i, reason: s.reason };
    if (opts.trustedKeyIds && !opts.trustedKeyIds.includes(r.keyId)) return { ok: false, index: i, reason: `signed by key ${r.keyId}, which is not one you trust` };
    keyIds.add(r.keyId);
    prev = r.tag;
  }
  return { ok: true, keyIds: [...keyIds] };
}
