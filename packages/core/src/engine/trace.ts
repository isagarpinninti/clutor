/**
 * Routing traces: for each generated token, which experts each mixture-of-experts layer chose, and the router's probabilities.
 *
 * A trace from a real model is the right input. Without one, `generateRoutingTrace` makes a SYNTHETIC trace with the properties
 * that matter to a cache and are reported for real routers: a skewed expert popularity, a token representation that changes slowly
 * (so consecutive tokens pick similar experts), and layers that see correlated representations (so layer L predicts layer L+1).
 * Results on a synthetic trace show how a policy behaves on that kind of workload; they are not a claim about any particular model.
 */

export interface LayerRouting {
  /** Index into the model's layers. */
  layer: number;
  /** The best `candidates` experts by router probability, best first. The first `used` are the ones the model actually computes. */
  experts: number[];
  /** Softmax probabilities over ALL experts of the layer for those experts (so they do not sum to 1). */
  probs: number[];
}
export type TokenRouting = LayerRouting[];

export interface TraceSpec {
  /** Layers that have experts. */
  moeLayers: number[];
  experts: number;
  used: number;
  /** How many ranked candidates to record per layer (>= used). */
  candidates?: number;
  tokens: number;
  seed?: number;
  /** Zipf exponent of expert popularity: 0 is uniform; around 1 is strongly skewed. */
  skew?: number;
  /** Autocorrelation of the token representation from one token to the next (0 to 1). */
  locality?: number;
  /** How alike different layers' views of a token are (0 to 1). */
  layerCorrelation?: number;
  /** How strongly the token representation (against popularity) decides routing. */
  contentWeight?: number;
  /** Every this many tokens, replace the representation (a new topic). 0 = never. */
  topicEvery?: number;
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(r: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
}

export function generateRoutingTrace(spec: TraceSpec): TokenRouting[] {
  const r = mulberry32(spec.seed ?? 1);
  const D = 8;
  const E = spec.experts;
  const R = Math.min(E, Math.max(spec.used, spec.candidates ?? spec.used + 8));
  const skew = spec.skew ?? 1;
  const rho = spec.locality ?? 0.9;
  const lc = spec.layerCorrelation ?? 0.8;
  const beta = spec.contentWeight ?? 3;
  const layers = spec.moeLayers;

  const w = layers.map(() => Array.from({ length: E }, () => Array.from({ length: D }, () => gaussian(r) / Math.sqrt(D))));
  const logPop = layers.map(() => {
    const perm = Array.from({ length: E }, (_, i) => i);
    for (let i = E - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j] as number, perm[i] as number]; }
    const lp = new Array<number>(E).fill(0);
    perm.forEach((e, rank) => { lp[e] = -skew * Math.log(rank + 1); });
    return lp;
  });

  let z = Array.from({ length: D }, () => gaussian(r));
  const trace: TokenRouting[] = [];
  const logits = new Float64Array(E);
  for (let t = 0; t < spec.tokens; t++) {
    if (spec.topicEvery && t > 0 && t % spec.topicEvery === 0) z = Array.from({ length: D }, () => gaussian(r));
    else z = z.map((v) => rho * v + Math.sqrt(1 - rho * rho) * gaussian(r));
    const token: TokenRouting = [];
    layers.forEach((layer, li) => {
      const zl = z.map((v) => lc * v + Math.sqrt(1 - lc * lc) * gaussian(r));
      let max = -Infinity;
      for (let e = 0; e < E; e++) {
        const we = (w[li] as number[][])[e] as number[];
        let dot = 0;
        for (let d = 0; d < D; d++) dot += (we[d] as number) * (zl[d] as number);
        const v = beta * dot + (logPop[li] as number[])[e]!;
        logits[e] = v;
        if (v > max) max = v;
      }
      let sum = 0;
      for (let e = 0; e < E; e++) { logits[e] = Math.exp((logits[e] as number) - max); sum += logits[e] as number; }
      const order = Array.from({ length: E }, (_, i) => i).sort((a, b) => (logits[b] as number) - (logits[a] as number)).slice(0, R);
      token.push({ layer, experts: order, probs: order.map((e) => (logits[e] as number) / sum) });
    });
    trace.push(token);
  }
  return trace;
}

/** One JSON object per line: `{"layer":3,"experts":[...],"probs":[...]}` per layer, tokens separated by a blank line. */
export function parseRoutingTrace(text: string): TokenRouting[] {
  const out: TokenRouting[] = [];
  let cur: TokenRouting = [];
  for (const line of text.split("\n")) {
    if (line.trim() === "") { if (cur.length) { out.push(cur); cur = []; } continue; }
    const o = JSON.parse(line) as LayerRouting;
    if (!Number.isInteger(o.layer) || !Array.isArray(o.experts) || !Array.isArray(o.probs) || o.experts.length !== o.probs.length || o.experts.length === 0) throw new Error("a trace line needs layer, experts and probs of equal length");
    cur.push(o);
  }
  if (cur.length) out.push(cur);
  return out;
}
