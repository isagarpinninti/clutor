import { createHash } from "node:crypto";
import { sealReceipt, type Receipt } from "@clutor/core";
import { appendReceipt, lastReceiptTag, loadOrCreateReceiptSigner } from "@clutor/core/node";

export interface DecideEndpoint { baseUrl: string; model: string; apiKey?: string }

export interface DecideResult {
  choice: string;
  probabilities: Record<string, number>;
  probabilityMassOnChoices: number;
  latencyMs: number;
}

/** The request was wrong (400) or the model server could not answer it (502). */
export class DecideError extends Error {
  constructor(readonly status: 400 | 502, message: string) {
    super(message);
  }
}

/**
 * One forward pass, one token: the model's own probability for each choice's letter, read from the first-token log-probabilities.
 * Not calibrated. Shared by `clutor decide` and `POST /v1/decide`.
 */
export async function runDecide(e: DecideEndpoint, prompt: string, choices: string[], f: typeof fetch = globalThis.fetch): Promise<DecideResult> {
  if (!prompt || choices.length < 2 || choices.length > 26) throw new DecideError(400, "a prompt and 2 to 26 choices are required");
  if (new Set(choices).size !== choices.length) throw new DecideError(400, "choices must be different from each other");
  const letters = choices.map((_, i) => String.fromCharCode(65 + i));
  const content = `${prompt}\n\nChoose exactly one option and answer with its letter only.\n${choices.map((c, i) => `${letters[i]}) ${c}`).join("\n")}\nAnswer:`;
  const t0 = Date.now();
  const res = await f(`${e.baseUrl.replace(/\/$/, "")}/chat/completions`, {
    method: "POST", headers: { "content-type": "application/json", ...(e.apiKey ? { authorization: `Bearer ${e.apiKey}` } : {}) },
    body: JSON.stringify({ model: e.model, messages: [{ role: "user", content }], max_tokens: 1, temperature: 0, logprobs: true, top_logprobs: 20 }),
  });
  if (!res.ok) throw new DecideError(502, `the model server answered ${res.status}: ${(await res.text().catch(() => "")).slice(0, 300)}`);
  const body = (await res.json()) as { choices?: Array<{ logprobs?: { content?: Array<{ top_logprobs?: Array<{ token: string; logprob: number }> }> } }> };
  const top = body.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs;
  if (!top) throw new DecideError(502, "the model server returned no log-probabilities; decide needs a server that supports logprobs (llama-server does)");
  const mass = letters.map((l) => top.filter((t) => t.token.trim() === l).reduce((a, t) => a + Math.exp(t.logprob), 0));
  const total = mass.reduce((a, b) => a + b, 0);
  if (total === 0) throw new DecideError(502, "the model did not put any probability on the choice letters; try rephrasing the prompt");
  const probs = mass.map((m) => m / total);
  const best = probs.indexOf(Math.max(...probs));
  return { choice: choices[best] as string, probabilities: Object.fromEntries(choices.map((c, i) => [c, probs[i] as number])), probabilityMassOnChoices: total, latencyMs: Date.now() - t0 };
}

/**
 * Signs a decision into the project's receipt log, so it chains with the agent's receipts. The receipt records a SHA-256 of the prompt, not the
 * prompt itself, because the log is meant to be shared for verification.
 */
export async function sealDecision(root: string, model: string, prompt: string, choices: string[], r: DecideResult, startedAt: string): Promise<Receipt> {
  const signer = await loadOrCreateReceiptSigner(root);
  const promptSha = createHash("sha256").update(prompt).digest("hex");
  const receipt = await sealReceipt(signer, {
    v: 2, id: `dec_${Date.now().toString(36)}`, task: `decide sha256:${promptSha} among [${choices.join("|")}] -> ${r.choice}`, model, startedAt,
    finishedAt: new Date().toISOString(), status: "finished", files: [], commands: [], steps: 1, previous: await lastReceiptTag(root),
  });
  await appendReceipt(root, receipt);
  return receipt;
}
