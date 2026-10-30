import { metaNumber, metaString, type GgufFile } from "./parse";

/**
 * What the planner and the streamer need to know about a model, derived from a GGUF file's tensor table: which bytes must always be
 * in memory, which are read layer by layer, and where each routed expert lives.
 */

export interface ByteRange {
  offset: number;
  length: number;
}

export interface LayerLayout {
  index: number;
  /** Attention, norms, shared experts and (for dense layers) the feed-forward: read for every token. */
  denseBytes: number;
  denseRanges: ByteRange[];
  /** The router (gating) weights: small, read for every token, kept resident. */
  routerBytes: number;
  /** Routed experts; empty for a dense layer. Each is a few byte ranges (gate, up, down). */
  experts: ByteRange[][];
  expertBytes: number;
}

export interface ModelLayout {
  arch: string;
  name: string;
  layerCount: number;
  expertCount: number;
  expertsUsed: number;
  sharedExperts: number;
  /** Token embeddings, output head, final norm: needed for every token. */
  residentBytes: number;
  layers: LayerLayout[];
  totalBytes: number;
  /** KV-cache geometry. */
  kv: { kvHeads: number; headDim: number; maxContext: number; layers: number };
  quantisations: Record<string, number>;
  /** True when this layout was computed from architecture numbers, not read from a file. */
  approximate: boolean;
  /** Tensors whose size this reader could not work out; their bytes are missing from the totals. */
  unsizedTensors: number;
}

const BLK = /^blk\.(\d+)\.(.+)$/;
const FUSED = /^ffn_(gate|up|down)_exps\.weight$/;
const PER_EXPERT = /^ffn_(gate|up|down)\.(\d+)\.weight$/;
const ROUTER = /^ffn_gate_inp\.weight$/;

export function buildModelLayout(file: GgufFile): ModelLayout {
  const m = file.metadata;
  const arch = metaString(m, "general.architecture") ?? "unknown";
  const layerCount = metaNumber(m, `${arch}.block_count`) ?? 0;
  const declaredExperts = metaNumber(m, `${arch}.expert_count`) ?? 0;
  const expertsUsed = metaNumber(m, `${arch}.expert_used_count`) ?? 0;
  const sharedExperts = metaNumber(m, `${arch}.expert_shared_count`) ?? 0;
  const embd = metaNumber(m, `${arch}.embedding_length`) ?? 0;
  const heads = metaNumber(m, `${arch}.attention.head_count`) ?? 0;
  const kvHeads = metaNumber(m, `${arch}.attention.head_count_kv`) ?? heads;
  const keyLen = metaNumber(m, `${arch}.attention.key_length`);
  const headDim = keyLen ?? (heads > 0 ? Math.floor(embd / heads) : 0);

  const layers: LayerLayout[] = Array.from({ length: layerCount }, (_, index) => ({ index, denseBytes: 0, denseRanges: [], routerBytes: 0, experts: [], expertBytes: 0 }));
  const quantisations: Record<string, number> = {};
  let residentBytes = 0;
  let totalBytes = 0;
  let unsized = 0;
  let expertCount = declaredExperts;

  for (const t of file.tensors) {
    if (t.nbytes === undefined) { unsized++; continue; }
    totalBytes += t.nbytes;
    quantisations[t.typeName] = (quantisations[t.typeName] ?? 0) + t.nbytes;
    const b = BLK.exec(t.name);
    const layer = b ? layers[Number(b[1])] : undefined;
    if (!b || !layer) { residentBytes += t.nbytes; continue; }
    const rest = b[2] as string;
    const range: ByteRange = { offset: t.fileOffset, length: t.nbytes };

    if (ROUTER.test(rest)) { layer.routerBytes += t.nbytes; residentBytes += t.nbytes; continue; }

    if (FUSED.test(rest)) {
      // One 3-D tensor holds every expert of this layer: experts are the slowest-varying dimension, so each is one contiguous slab.
      const n = t.dims[t.dims.length - 1] ?? 0;
      if (n > 0 && t.nbytes % n === 0) {
        expertCount = Math.max(expertCount, n);
        const slab = t.nbytes / n;
        for (let e = 0; e < n; e++) {
          (layer.experts[e] ??= []).push({ offset: t.fileOffset + e * slab, length: slab });
        }
        layer.expertBytes += slab;
        continue;
      }
    }

    const pe = PER_EXPERT.exec(rest);
    if (pe && declaredExperts > 0) {
      const e = Number(pe[2]);
      (layer.experts[e] ??= []).push(range);
      layer.expertBytes = Math.max(layer.expertBytes, (layer.experts[e] ?? []).reduce((s, r) => s + r.length, 0));
      expertCount = Math.max(expertCount, e + 1);
      continue;
    }

    layer.denseBytes += t.nbytes;
    layer.denseRanges.push(range);
  }

  const name = metaString(m, "general.name") ?? metaString(m, "general.basename") ?? arch;
  return {
    arch, name, layerCount, expertCount, expertsUsed, sharedExperts, residentBytes, layers, totalBytes,
    kv: { kvHeads, headDim, maxContext: metaNumber(m, `${arch}.context_length`) ?? 0, layers: layerCount },
    quantisations, approximate: false, unsizedTensors: unsized,
  };
}

/** Bytes of one routed expert in the first layer that has experts (all experts of a model have the same size per layer). */
export function expertSizeBytes(layout: ModelLayout): number {
  return layout.layers.find((l) => l.experts.length > 0)?.expertBytes ?? 0;
}
