/**
 * A GGUF (v2 and v3) reader written from the file format: header, metadata, tensor table and data offset. It reads only what it needs
 * through a `ByteReader`, so a 400 GB file is inspected by reading its first few megabytes.
 */

export interface ByteReader {
  readonly size: number;
  read(offset: number, length: number): Promise<Uint8Array>;
}

export class GgufError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GgufError";
  }
}

/** Thrown inside the cursor when the buffered prefix ends mid-structure; the caller reads more and retries. */
class NeedMore extends Error {}

export type GgufValue = number | bigint | boolean | string | GgufValue[];

export interface GgufTensor {
  name: string;
  /** Innermost dimension first, as stored. */
  dims: number[];
  type: number;
  typeName: string;
  /** Offset from the start of the tensor data section. */
  offset: number;
  /** Offset from the start of the file. */
  fileOffset: number;
  /** Size in bytes; undefined for a quantisation this reader has no size for. */
  nbytes: number | undefined;
}

export interface GgufFile {
  version: number;
  tensorCount: number;
  metadata: Map<string, GgufValue>;
  tensors: GgufTensor[];
  alignment: number;
  /** Start of the tensor data section, in bytes from the start of the file. */
  dataOffset: number;
  fileSize: number;
}

/** `[block size in elements, bytes per block, name]` for the ggml tensor types this reader knows. */
export const GGML_TYPES: Record<number, readonly [number, number, string]> = {
  0: [1, 4, "F32"], 1: [1, 2, "F16"], 2: [32, 18, "Q4_0"], 3: [32, 20, "Q4_1"], 6: [32, 22, "Q5_0"], 7: [32, 24, "Q5_1"], 8: [32, 34, "Q8_0"], 9: [32, 36, "Q8_1"],
  10: [256, 84, "Q2_K"], 11: [256, 110, "Q3_K"], 12: [256, 144, "Q4_K"], 13: [256, 176, "Q5_K"], 14: [256, 210, "Q6_K"], 15: [256, 292, "Q8_K"],
  16: [256, 66, "IQ2_XXS"], 17: [256, 74, "IQ2_XS"], 18: [256, 98, "IQ3_XXS"], 19: [256, 50, "IQ1_S"], 20: [32, 18, "IQ4_NL"], 21: [256, 110, "IQ3_S"],
  22: [256, 82, "IQ2_S"], 23: [256, 136, "IQ4_XS"], 24: [1, 1, "I8"], 25: [1, 2, "I16"], 26: [1, 4, "I32"], 27: [1, 8, "I64"], 28: [1, 8, "F64"],
  29: [256, 56, "IQ1_M"], 30: [1, 2, "BF16"], 34: [256, 54, "TQ1_0"], 35: [256, 66, "TQ2_0"], 39: [32, 17, "MXFP4"],
};

/** Approximate bits per weight of a tensor type (for the planner's what-if sizing). */
export function bitsPerWeight(type: number): number | undefined {
  const t = GGML_TYPES[type];
  return t ? (t[1] * 8) / t[0] : undefined;
}

// Defensive limits: a header can claim anything, and these keep a hostile or corrupt file from exhausting memory.
const MAX_TENSORS = 2_000_000;
const MAX_KV = 2_000_000;
const MAX_STRING = 256 * 1024 * 1024;
const MAX_ARRAY = 200_000_000;
const MAX_DIMS = 8;
const MAX_ARRAY_DEPTH = 4;
const MAX_PREFIX = 1024 * 1024 * 1024;
const FIRST_READ = 4 * 1024 * 1024;

class Cursor {
  private readonly view: DataView;
  pos = 0;
  constructor(readonly bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  private need(n: number): void {
    if (this.pos + n > this.bytes.length) throw new NeedMore();
  }
  u8(): number { this.need(1); return this.view.getUint8(this.pos++); }
  i8(): number { this.need(1); return this.view.getInt8(this.pos++); }
  u16(): number { this.need(2); const v = this.view.getUint16(this.pos, true); this.pos += 2; return v; }
  i16(): number { this.need(2); const v = this.view.getInt16(this.pos, true); this.pos += 2; return v; }
  u32(): number { this.need(4); const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  i32(): number { this.need(4); const v = this.view.getInt32(this.pos, true); this.pos += 4; return v; }
  f32(): number { this.need(4); const v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; }
  f64(): number { this.need(8); const v = this.view.getFloat64(this.pos, true); this.pos += 8; return v; }
  u64(): bigint { this.need(8); const v = this.view.getBigUint64(this.pos, true); this.pos += 8; return v; }
  i64(): bigint { this.need(8); const v = this.view.getBigInt64(this.pos, true); this.pos += 8; return v; }
  /** A 64-bit count or offset that must fit a JavaScript safe integer. */
  safe(what: string): number {
    const v = this.u64();
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new GgufError(`${what} is larger than 2^53`);
    return Number(v);
  }
  string(): string {
    const n = this.safe("string length");
    if (n > MAX_STRING) throw new GgufError(`a string claims ${n} bytes`);
    this.need(n);
    const s = new TextDecoder("utf-8", { fatal: false }).decode(this.bytes.subarray(this.pos, this.pos + n));
    this.pos += n;
    return s;
  }
}

function readValue(c: Cursor, type: number, depth: number): GgufValue {
  switch (type) {
    case 0: return c.u8();
    case 1: return c.i8();
    case 2: return c.u16();
    case 3: return c.i16();
    case 4: return c.u32();
    case 5: return c.i32();
    case 6: return c.f32();
    case 7: return c.u8() !== 0;
    case 8: return c.string();
    case 9: {
      if (depth >= MAX_ARRAY_DEPTH) throw new GgufError("arrays are nested too deeply");
      const elem = c.u32();
      const n = c.safe("array length");
      if (n > MAX_ARRAY) throw new GgufError(`an array claims ${n} elements`);
      const out: GgufValue[] = [];
      for (let i = 0; i < n; i++) out.push(readValue(c, elem, depth + 1));
      return out;
    }
    case 10: return c.u64();
    case 11: return c.i64();
    case 12: return c.f64();
    default: throw new GgufError(`unknown metadata value type ${type}`);
  }
}

function tensorBytes(dims: number[], type: number): number | undefined {
  const t = GGML_TYPES[type];
  if (!t) return undefined;
  const [block, size] = t;
  const first = dims[0] ?? 1;
  if (first % block !== 0) return undefined;
  let n = (first / block) * size;
  for (let i = 1; i < dims.length; i++) n *= dims[i] as number;
  return Number.isSafeInteger(n) ? n : undefined;
}

function parseOnce(bytes: Uint8Array, fileSize: number): GgufFile {
  const c = new Cursor(bytes);
  if (bytes.length >= 4 && (bytes[0] !== 0x47 || bytes[1] !== 0x47 || bytes[2] !== 0x55 || bytes[3] !== 0x46)) throw new GgufError("not a GGUF file (bad magic)");
  c.pos = 4;
  const version = c.u32();
  if (version !== 2 && version !== 3) throw new GgufError(`unsupported GGUF version ${version}`);
  const tensorCount = c.safe("tensor count");
  const kvCount = c.safe("metadata count");
  if (tensorCount > MAX_TENSORS) throw new GgufError(`the file claims ${tensorCount} tensors`);
  if (kvCount > MAX_KV) throw new GgufError(`the file claims ${kvCount} metadata entries`);
  const metadata = new Map<string, GgufValue>();
  for (let i = 0; i < kvCount; i++) {
    const key = c.string();
    metadata.set(key, readValue(c, c.u32(), 0));
  }
  const raw: Array<{ name: string; dims: number[]; type: number; offset: number }> = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = c.string();
    const nd = c.u32();
    if (nd > MAX_DIMS) throw new GgufError(`tensor "${name}" has ${nd} dimensions`);
    const dims: number[] = [];
    for (let d = 0; d < nd; d++) dims.push(c.safe("tensor dimension"));
    raw.push({ name, dims, type: c.u32(), offset: c.safe("tensor offset") });
  }
  const a = metadata.get("general.alignment");
  const alignment = typeof a === "number" && a > 0 && a <= 1 << 20 ? a : 32;
  const dataOffset = Math.ceil(c.pos / alignment) * alignment;
  const tensors: GgufTensor[] = raw.map((t) => ({
    ...t, typeName: GGML_TYPES[t.type]?.[2] ?? `type${t.type}`, fileOffset: dataOffset + t.offset, nbytes: tensorBytes(t.dims, t.type),
  }));
  return { version, tensorCount, metadata, tensors, alignment, dataOffset, fileSize };
}

/** Reads the header, metadata and tensor table of a GGUF file. Refuses a tensor that points outside the file. */
export async function parseGguf(reader: ByteReader): Promise<GgufFile> {
  if (reader.size < 24) throw new GgufError("too small to be a GGUF file");
  let want = Math.min(reader.size, FIRST_READ);
  for (;;) {
    const bytes = await reader.read(0, want);
    try {
      const file = parseOnce(bytes, reader.size);
      validateBounds(file);
      return file;
    } catch (e) {
      if (!(e instanceof NeedMore)) throw e;
      if (want >= reader.size) throw new GgufError("the header runs past the end of the file");
      if (want >= MAX_PREFIX) throw new GgufError("the header is larger than 1 GiB");
      want = Math.min(reader.size, want * 4);
    }
  }
}

/** The check a loader needs before it trusts any offset: every tensor lies inside the file. */
export function validateBounds(file: GgufFile): void {
  if (file.dataOffset > file.fileSize) throw new GgufError("the tensor data starts after the end of the file");
  for (const t of file.tensors) {
    if (t.offset % file.alignment !== 0) throw new GgufError(`tensor "${t.name}" is not aligned to ${file.alignment}`);
    if (t.nbytes !== undefined && t.fileOffset + t.nbytes > file.fileSize) throw new GgufError(`tensor "${t.name}" extends past the end of the file`);
  }
}

export const metaNumber = (m: Map<string, GgufValue>, key: string): number | undefined => {
  const v = m.get(key);
  return typeof v === "number" ? v : typeof v === "bigint" ? Number(v) : undefined;
};
export const metaString = (m: Map<string, GgufValue>, key: string): string | undefined => {
  const v = m.get(key);
  return typeof v === "string" ? v : undefined;
};

/** A `ByteReader` over memory, for tests and for files already loaded. */
export function memoryReader(bytes: Uint8Array): ByteReader {
  return { size: bytes.length, read: async (offset, length) => bytes.subarray(offset, Math.min(bytes.length, offset + length)) };
}
