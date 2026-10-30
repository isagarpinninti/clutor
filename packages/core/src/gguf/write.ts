import { GGML_TYPES } from "./parse";

/**
 * Builds a valid GGUF v3 header for a described model. Used to make test fixtures and benchmark shards: the tensors are real
 * (names, shapes, types, alignment, offsets) and a loader reads them like any other file. The data section is left to the caller,
 * because a benchmark shard can be many gigabytes.
 */

export interface SyntheticTensor {
  name: string;
  dims: number[];
  /** ggml type id (default 8 = Q8_0). */
  type?: number;
}

export interface SyntheticModelSpec {
  arch: string;
  name: string;
  layers: number;
  embedding: number;
  heads: number;
  kvHeads: number;
  contextLength: number;
  expertCount: number;
  expertsUsed: number;
  expertFfn: number;
  denseFfn: number;
  vocab: number;
  sharedExperts?: number;
  quantType?: number;
}

const enc = new TextEncoder();

class Out {
  parts: Uint8Array[] = [];
  length = 0;
  push(b: Uint8Array): void { this.parts.push(b); this.length += b.length; }
  u32(v: number): void { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); this.push(b); }
  u64(v: number): void { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); this.push(b); }
  str(s: string): void { const e = enc.encode(s); this.u64(e.length); this.push(e); }
  bytes(): Uint8Array { const o = new Uint8Array(this.length); let p = 0; for (const x of this.parts) { o.set(x, p); p += x.length; } return o; }
}

export function tensorByteSize(t: SyntheticTensor): number {
  const [block, size] = GGML_TYPES[t.type ?? 8] as readonly [number, number, string];
  const first = t.dims[0] as number;
  if (first % block !== 0) throw new Error(`${t.name}: ${first} is not a multiple of the block size ${block}`);
  return t.dims.slice(1).reduce((a, d) => a * d, (first / block) * size);
}

export function syntheticTensors(spec: SyntheticModelSpec): SyntheticTensor[] {
  const q = spec.quantType ?? 8;
  const kvDim = (spec.embedding / spec.heads) * spec.kvHeads;
  const out: SyntheticTensor[] = [{ name: "token_embd.weight", dims: [spec.embedding, spec.vocab], type: q }];
  for (let l = 0; l < spec.layers; l++) {
    const p = `blk.${l}.`;
    out.push(
      { name: `${p}attn_norm.weight`, dims: [spec.embedding], type: 0 },
      { name: `${p}attn_q.weight`, dims: [spec.embedding, spec.embedding], type: q },
      { name: `${p}attn_k.weight`, dims: [spec.embedding, kvDim], type: q },
      { name: `${p}attn_v.weight`, dims: [spec.embedding, kvDim], type: q },
      { name: `${p}attn_output.weight`, dims: [spec.embedding, spec.embedding], type: q },
      { name: `${p}ffn_norm.weight`, dims: [spec.embedding], type: 0 },
    );
    if (spec.expertCount > 0) {
      out.push(
        { name: `${p}ffn_gate_inp.weight`, dims: [spec.embedding, spec.expertCount], type: 0 },
        { name: `${p}ffn_gate_exps.weight`, dims: [spec.embedding, spec.expertFfn, spec.expertCount], type: q },
        { name: `${p}ffn_up_exps.weight`, dims: [spec.embedding, spec.expertFfn, spec.expertCount], type: q },
        { name: `${p}ffn_down_exps.weight`, dims: [spec.expertFfn, spec.embedding, spec.expertCount], type: q },
      );
      if ((spec.sharedExperts ?? 0) > 0) {
        out.push(
          { name: `${p}ffn_gate_shexp.weight`, dims: [spec.embedding, spec.expertFfn * (spec.sharedExperts ?? 1)], type: q },
          { name: `${p}ffn_up_shexp.weight`, dims: [spec.embedding, spec.expertFfn * (spec.sharedExperts ?? 1)], type: q },
          { name: `${p}ffn_down_shexp.weight`, dims: [spec.expertFfn * (spec.sharedExperts ?? 1), spec.embedding], type: q },
        );
      }
    } else {
      out.push(
        { name: `${p}ffn_gate.weight`, dims: [spec.embedding, spec.denseFfn], type: q },
        { name: `${p}ffn_up.weight`, dims: [spec.embedding, spec.denseFfn], type: q },
        { name: `${p}ffn_down.weight`, dims: [spec.denseFfn, spec.embedding], type: q },
      );
    }
  }
  out.push({ name: "output_norm.weight", dims: [spec.embedding], type: 0 }, { name: "output.weight", dims: [spec.embedding, spec.vocab], type: q });
  return out;
}

/** The bytes from the start of the file up to the start of the tensor data, plus where each tensor's data goes. */
export function buildGgufHeader(spec: SyntheticModelSpec, tensors: SyntheticTensor[] = syntheticTensors(spec), alignment = 32): { header: Uint8Array; dataOffset: number; offsets: number[]; dataSize: number } {
  const o = new Out();
  o.push(enc.encode("GGUF"));
  o.u32(3);
  const kv: Array<[string, "u32" | "str", number | string]> = [
    ["general.architecture", "str", spec.arch], ["general.name", "str", spec.name], ["general.alignment", "u32", alignment],
    [`${spec.arch}.block_count`, "u32", spec.layers], [`${spec.arch}.embedding_length`, "u32", spec.embedding],
    [`${spec.arch}.attention.head_count`, "u32", spec.heads], [`${spec.arch}.attention.head_count_kv`, "u32", spec.kvHeads],
    [`${spec.arch}.context_length`, "u32", spec.contextLength],
    ...(spec.expertCount > 0 ? ([[`${spec.arch}.expert_count`, "u32", spec.expertCount], [`${spec.arch}.expert_used_count`, "u32", spec.expertsUsed], [`${spec.arch}.expert_shared_count`, "u32", spec.sharedExperts ?? 0]] as Array<[string, "u32", number]>) : []),
  ];
  o.u64(tensors.length);
  o.u64(kv.length);
  for (const [k, t, v] of kv) {
    o.str(k);
    if (t === "u32") { o.u32(4); o.u32(v as number); } else { o.u32(8); o.str(v as string); }
  }
  const offsets: number[] = [];
  let at = 0;
  for (const t of tensors) {
    o.str(t.name);
    o.u32(t.dims.length);
    for (const d of t.dims) o.u64(d);
    o.u32(t.type ?? 8);
    at = Math.ceil(at / alignment) * alignment;
    offsets.push(at);
    o.u64(at);
    at += tensorByteSize(t);
  }
  const pad = (Math.ceil(o.length / alignment) * alignment) - o.length;
  o.push(new Uint8Array(pad));
  return { header: o.bytes(), dataOffset: o.length, offsets, dataSize: at };
}
