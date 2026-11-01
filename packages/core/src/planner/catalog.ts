import { type LayerLayout, type ModelLayout } from "../gguf/layout";

/**
 * Architecture descriptions of well-known models, so the planner can answer "would this fit?" before you download 400 GB. Every
 * number is a published hyperparameter; sizes are computed from them (not remembered), and the result is labelled approximate.
 * A real GGUF file always overrides this: load it and the layout comes from its tensor table.
 */

export type Attention =
  | { kind: "gqa"; heads: number; kvHeads: number; headDim: number }
  | { kind: "mla"; heads: number; qLoraRank: number; kvLoraRank: number; qkNopeDim: number; qkRopeDim: number; vHeadDim: number };

export interface ArchitectureSpec {
  id: string;
  name: string;
  family: string;
  layers: number;
  /** Leading layers that use a dense feed-forward instead of experts. */
  denseLayers: number;
  hidden: number;
  attention: Attention;
  vocab: number;
  denseFfn: number;
  experts: number;
  expertsUsed: number;
  sharedExperts: number;
  expertFfn: number;
  maxContext: number;
  /** The output projection is the embedding matrix itself (small Qwen2.5 models), so the vocabulary matrix is counted once. */
  tiedEmbeddings?: boolean;
  /** The headline parameter count the vendor publishes, to check the computation against. */
  publishedParamsB: number;
  source: string;
}

export const CATALOG: ArchitectureSpec[] = [
  {
    id: "deepseek-v3", name: "DeepSeek-V3 / R1", family: "deepseek", layers: 61, denseLayers: 3, hidden: 7168, vocab: 129280, denseFfn: 18432,
    attention: { kind: "mla", heads: 128, qLoraRank: 1536, kvLoraRank: 512, qkNopeDim: 128, qkRopeDim: 64, vHeadDim: 128 },
    experts: 256, expertsUsed: 8, sharedExperts: 1, expertFfn: 2048, maxContext: 131072, publishedParamsB: 671, source: "DeepSeek-V3 technical report",
  },
  {
    // Not a published configuration: the layer and expert counts and the two sizes (about 9.9 GiB dense, 419 GiB routed experts) come from the
    // stress-test brief, and the remaining hyperparameters are chosen so the computed sizes land on them. Treat it as a 744B-class stand-in.
    id: "glm-5-744b", name: "GLM-5 744B (stress-test profile)", family: "glm-moe", layers: 78, denseLayers: 0, hidden: 6144, vocab: 154880, denseFfn: 0,
    attention: { kind: "gqa", heads: 64, kvHeads: 8, headDim: 128 }, experts: 256, expertsUsed: 8, sharedExperts: 2, expertFfn: 2016, maxContext: 131072,
    publishedParamsB: 744, source: "stress-test profile built from the sizes in the request; not the vendor's published configuration",
  },
  {
    id: "qwen3-235b-a22b", name: "Qwen3-235B-A22B", family: "qwen3-moe", layers: 94, denseLayers: 0, hidden: 4096, vocab: 151936, denseFfn: 0,
    attention: { kind: "gqa", heads: 64, kvHeads: 4, headDim: 128 }, experts: 128, expertsUsed: 8, sharedExperts: 0, expertFfn: 1536, maxContext: 131072,
    publishedParamsB: 235, source: "Qwen3 model card",
  },
  {
    id: "qwen3-30b-a3b", name: "Qwen3-30B-A3B", family: "qwen3-moe", layers: 48, denseLayers: 0, hidden: 2048, vocab: 151936, denseFfn: 0,
    attention: { kind: "gqa", heads: 32, kvHeads: 4, headDim: 128 }, experts: 128, expertsUsed: 8, sharedExperts: 0, expertFfn: 768, maxContext: 131072,
    publishedParamsB: 30.5, source: "Qwen3 model card",
  },
  {
    id: "mixtral-8x22b", name: "Mixtral 8x22B", family: "mixtral", layers: 56, denseLayers: 0, hidden: 6144, vocab: 32768, denseFfn: 0,
    attention: { kind: "gqa", heads: 48, kvHeads: 8, headDim: 128 }, experts: 8, expertsUsed: 2, sharedExperts: 0, expertFfn: 16384, maxContext: 65536,
    publishedParamsB: 141, source: "Mistral AI announcement",
  },
  {
    id: "llama-3.3-70b", name: "Llama 3.3 70B (dense)", family: "llama", layers: 80, denseLayers: 80, hidden: 8192, vocab: 128256, denseFfn: 28672,
    attention: { kind: "gqa", heads: 64, kvHeads: 8, headDim: 128 }, experts: 0, expertsUsed: 0, sharedExperts: 0, expertFfn: 0, maxContext: 131072,
    publishedParamsB: 70.6, source: "Meta model card",
  },
  {
    id: "olmoe-1b-7b", name: "OLMoE-1B-7B", family: "olmoe", layers: 16, denseLayers: 0, hidden: 2048, vocab: 50304, denseFfn: 0,
    attention: { kind: "gqa", heads: 16, kvHeads: 16, headDim: 128 }, experts: 64, expertsUsed: 8, sharedExperts: 0, expertFfn: 1024, maxContext: 4096,
    publishedParamsB: 6.9, source: "OLMoE paper (Muennighoff et al., 2024)",
  },
  {
    id: "granite-3.0-3b-a800m", name: "Granite 3.0 3B-A800M (IBM)", family: "granite-moe", layers: 32, denseLayers: 0, hidden: 1536, vocab: 49155, denseFfn: 0,
    attention: { kind: "gqa", heads: 24, kvHeads: 8, headDim: 64 }, experts: 40, expertsUsed: 8, sharedExperts: 0, expertFfn: 512, maxContext: 4096,
    publishedParamsB: 3.4, source: "IBM Granite 3.0 model card; shapes read from the GGUF header of granite-3.0-3b-a800m-instruct",
  },
  {
    id: "qwen2.5-coder-7b", name: "Qwen2.5-Coder-7B (dense)", family: "qwen2", layers: 28, denseLayers: 28, hidden: 3584, vocab: 152064, denseFfn: 18944,
    attention: { kind: "gqa", heads: 28, kvHeads: 4, headDim: 128 }, experts: 0, expertsUsed: 0, sharedExperts: 0, expertFfn: 0, maxContext: 131072,
    publishedParamsB: 7.6, source: "Qwen2.5-Coder model card",
  },
  {
    id: "qwen2.5-1.5b", name: "Qwen2.5-1.5B-Instruct (dense)", family: "qwen2", layers: 28, denseLayers: 28, hidden: 1536, vocab: 151936, denseFfn: 8960,
    attention: { kind: "gqa", heads: 12, kvHeads: 2, headDim: 128 }, experts: 0, expertsUsed: 0, sharedExperts: 0, expertFfn: 0, maxContext: 32768, tiedEmbeddings: true,
    publishedParamsB: 1.54, source: "Qwen2.5 model card; shapes read from the GGUF header of qwen2.5-1.5b-instruct",
  },
];

/** Parameters in one layer: [attention, dense feed-forward, routed experts (all), each routed expert, shared experts, router]. */
export function layerParams(s: ArchitectureSpec, moe: boolean): { attention: number; denseFfn: number; routed: number; perExpert: number; shared: number; router: number } {
  const a = s.attention;
  const attention =
    a.kind === "gqa"
      ? s.hidden * a.heads * a.headDim + 2 * s.hidden * a.kvHeads * a.headDim + a.heads * a.headDim * s.hidden
      : s.hidden * a.qLoraRank + a.qLoraRank * a.heads * (a.qkNopeDim + a.qkRopeDim) + s.hidden * (a.kvLoraRank + a.qkRopeDim) + a.kvLoraRank * a.heads * (a.qkNopeDim + a.vHeadDim) + a.heads * a.vHeadDim * s.hidden;
  const perExpert = 3 * s.hidden * s.expertFfn;
  return {
    attention,
    denseFfn: moe ? 0 : 3 * s.hidden * s.denseFfn,
    routed: moe ? perExpert * s.experts : 0,
    perExpert: moe ? perExpert : 0,
    shared: moe ? perExpert * s.sharedExperts : 0,
    router: moe ? s.hidden * s.experts : 0,
  };
}

export function totalParams(s: ArchitectureSpec): number {
  let n = (s.tiedEmbeddings ? 1 : 2) * s.vocab * s.hidden;
  for (let l = 0; l < s.layers; l++) {
    const p = layerParams(s, s.experts > 0 && l >= s.denseLayers);
    n += p.attention + p.denseFfn + p.routed + p.shared + p.router;
  }
  return n;
}

/** Bits per weight of common quantisations, for what-if sizing. */
export const QUANTS: Record<string, number> = { F16: 16, Q8_0: 8.5, Q6_K: 6.5625, Q5_K_M: 5.5, Q4_K_M: 4.85, Q4_0: 4.5, IQ4_XS: 4.25, Q3_K_M: 3.9, Q2_K: 2.625, IQ2_XXS: 2.06 };

/** A layout from architecture numbers at a given quantisation. Experts are placed at made-up offsets (the file does not exist). */
export function approximateLayout(s: ArchitectureSpec, bitsPerWeight: number): ModelLayout {
  const bytes = (params: number): number => Math.ceil((params * bitsPerWeight) / 8);
  let offset = 0;
  const layers: LayerLayout[] = [];
  let total = 0;
  const resident = bytes((s.tiedEmbeddings ? 1 : 2) * s.vocab * s.hidden);
  for (let l = 0; l < s.layers; l++) {
    const moe = s.experts > 0 && l >= s.denseLayers;
    const p = layerParams(s, moe);
    const dense = bytes(p.attention + p.denseFfn + p.shared);
    const router = Math.ceil((p.router * 32) / 8); // routers stay in full precision
    const eBytes = moe ? bytes(p.perExpert) : 0;
    const experts = moe ? Array.from({ length: s.experts }, () => [{ offset: (offset += eBytes) - eBytes, length: eBytes }]) : [];
    layers.push({ index: l, denseBytes: dense, denseRanges: [{ offset: 0, length: dense }], routerBytes: router, experts, expertBytes: eBytes });
    total += dense + router + eBytes * (moe ? s.experts : 0);
  }
  const a = s.attention;
  return {
    arch: s.family, name: s.name, layerCount: s.layers, expertCount: s.experts, expertsUsed: s.expertsUsed, sharedExperts: s.sharedExperts,
    residentBytes: resident + layers.reduce((n, x) => n + x.routerBytes, 0), layers, totalBytes: total + resident,
    // MLA caches one compressed latent per token (kv_lora_rank + rope dims) shared by all heads; modelled as one "head".
    kv: a.kind === "gqa" ? { kvHeads: a.kvHeads, headDim: a.headDim, maxContext: s.maxContext, layers: s.layers } : { kvHeads: 1, headDim: (a.kvLoraRank + a.qkRopeDim) / 2, maxContext: s.maxContext, layers: s.layers },
    quantisations: { [`~${bitsPerWeight}bpw`]: total + resident }, approximate: true, unsizedTensors: 0,
  };
}
