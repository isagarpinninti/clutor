import { describe, expect, it } from "vitest";
import { buildGgufHeader, buildLibrary, classifyFile, describeLibrary, pickImage, pickLlm, pickVision, scanFiles, tensorByteSize, syntheticTensors, type ByteReader, type Classified, type Entry, type SyntheticModelSpec } from "../src";

const GiB = 2 ** 30;
const reader = (head: Uint8Array, size: number): ByteReader => ({ size, read: async (o, n) => head.subarray(o, Math.min(head.length, o + n)) });

const spec = (over: Partial<SyntheticModelSpec>): SyntheticModelSpec => ({ arch: "qwen3moe", name: "Qwen3-30B-A3B", layers: 2, embedding: 256, heads: 4, kvHeads: 2, contextLength: 4096, expertCount: 8, expertsUsed: 2, expertFfn: 256, denseFfn: 0, vocab: 512, quantType: 12, ...over });
async function llm(path: string, over: Partial<SyntheticModelSpec>, bytes: number): Promise<Classified> {
  const s = spec(over);
  const tensors = syntheticTensors(s);
  const { header, dataOffset } = buildGgufHeader(s, tensors);
  const total = dataOffset + tensors.reduce((n, t) => n + tensorByteSize(t), 0);
  return classifyFile({ path, name: path.split("/").pop() as string, bytes }, reader(header, total));
}

/** A GGUF with string metadata and tensors, the way a ComfyUI file or a projector is (the data is not there; the reader claims it is). */
function bare(kv: Array<[string, string]>, tensors: Array<{ name: string; dims: number[]; type: number }>): ByteReader {
  const parts: number[] = [];
  const u32 = (v: number): void => { parts.push(v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255); };
  const u64 = (v: number): void => { u32(v); u32(Math.floor(v / 2 ** 32)); };
  const str = (s: string): void => { const b = new TextEncoder().encode(s); u64(b.length); parts.push(...b); };
  parts.push(0x47, 0x47, 0x55, 0x46); u32(3); u64(tensors.length); u64(kv.length);
  for (const [k, v] of kv) { str(k); u32(8); str(v); }
  let off = 0;
  for (const t of tensors) { str(t.name); u32(t.dims.length); for (const d of t.dims) u64(d); u32(t.type); u64(off); off += 1 << 24; }
  while (parts.length % 32) parts.push(0);
  const head = Uint8Array.from(parts);
  return reader(head, head.length + off + (1 << 24));
}

function safetensors(names: string[]): ByteReader {
  const json = new TextEncoder().encode(JSON.stringify(Object.fromEntries(names.map((n) => [n, { dtype: "BF16", shape: [1], data_offsets: [0, 2] }]))));
  const head = new Uint8Array(8 + json.length);
  new DataView(head.buffer).setBigUint64(0, BigInt(json.length), true);
  head.set(json, 8);
  return reader(head, head.length + 2);
}
const entry = (path: string, bytes: number): Entry => ({ path, name: path.split("/").pop() as string, bytes });

describe("what a file is", () => {
  it("reads a chat model, and says it is a mixture of experts", async () => {
    const f = await llm("/m/Qwen3-30B-A3B-Q4_K_M.gguf", {}, 17 * GiB);
    expect(f).toMatchObject({ kind: "chat-model", arch: "qwen3moe", moe: true, paramsB: 30 });
  });
  it("tells a picture-reading model from a plain one, and finds a projector", async () => {
    expect((await llm("/m/vl.gguf", { arch: "qwen3vlmoe", name: "Qwen3-VL-30B-A3B" }, 10 * GiB)).kind).toBe("vision-model");
    const p = await classifyFile(entry("/m/x.mmproj-f16.gguf", GiB), bare([["general.architecture", "clip"], ["general.type", "mmproj"], ["clip.projector_type", "qwen3vl_merger"]], []));
    expect(p).toMatchObject({ kind: "projector", arch: "qwen3vl_merger" });
  });
  it("recognises an image generator that states no architecture, by the names of its tensors", async () => {
    const f = await classifyFile(entry("/m/qwen-image-2.1-turbo-Q4_K_M.gguf", 4 * GiB), bare([], [
      { name: "model.diffusion_model.img_in.weight", dims: [64, 4096], type: 30 },
      { name: "model.diffusion_model.time_text_embed.timestep_embedder.linear_1.weight", dims: [256, 4096], type: 12 },
      { name: "model.diffusion_model.transformer_blocks.0.img_mlp.out.weight", dims: [12288, 4096], type: 12 },
    ]));
    expect(f).toMatchObject({ kind: "image-model", arch: "qwen-image", turbo: true });
  });
  it("recognises a text encoder and a VAE from safetensors headers", async () => {
    expect((await classifyFile(entry("/m/te.safetensors", 9 * GiB), safetensors(["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"]))).kind).toBe("text-encoder");
    expect((await classifyFile(entry("/m/v.safetensors", GiB), safetensors(["encoder.conv_in.weight", "decoder.conv_in.weight"]))).kind).toBe("vae");
    // the VAE of Qwen-Image (a video-style VAE) names its tensors differently: found on a real file
    expect((await classifyFile(entry("/m/qwen_image_2.1_vae_bf16.safetensors", 0.7 * GiB), safetensors(["conv1.weight", "conv2.weight", "decoder.conv1.weight", "encoder.conv1.weight", "decoder.head.2.weight", "decoder.middle.0.residual.2.weight"]))).kind).toBe("vae");
  });
  it("says why a file cannot be used instead of failing", async () => {
    const f = await llm("/m/gemma.gguf", { arch: "gemma3", name: "Gemma-3-12B" }, 7 * GiB);
    expect(f.kind).toBe("other");
    expect(f.note).toMatch(/not supported/);
    const junk = await classifyFile(entry("/m/junk.gguf", 100), reader(new Uint8Array(100), 100));
    expect(junk.kind).toBe("other");
  });
});

describe("a folder as a library", () => {
  const files = async (): Promise<Classified[]> => scanFiles([
    entry("/lib/chat/Qwen3-30B-A3B-Q4_K_M.gguf", 17 * GiB), entry("/lib/chat/Llama-3.2-1B.gguf", 0.8 * GiB), entry("/lib/chat/Coder-7B.gguf", 4.7 * GiB),
    entry("/lib/vl/Qwen3-VL-30B-A3B.i1-Q2_K.gguf", 10.5 * GiB), entry("/lib/vl/Qwen3-VL-30B-A3B.mmproj-f16.gguf", GiB),
    entry("/lib/img/qwen-image-2.1-turbo-Q4_K_M.gguf", 4 * GiB), entry("/lib/img/te-int8.safetensors", 8.7 * GiB), entry("/lib/img/te-bf16.safetensors", 16 * GiB), entry("/lib/img/vae.safetensors", 0.6 * GiB),
    entry("/lib/orphan/flux-dev-Q4.gguf", 6 * GiB),
  ], (e) => {
    const f = e.name;
    if (f.startsWith("Qwen3-30B")) return fake(spec({}), e);
    if (f.startsWith("Llama")) return fake(spec({ arch: "llama", name: "Llama-3.2-1B", expertCount: 0, expertsUsed: 0, denseFfn: 512 }), e);
    if (f.startsWith("Coder")) return fake(spec({ arch: "qwen2", name: "Qwen2.5-Coder-7B", expertCount: 0, expertsUsed: 0, denseFfn: 512 }), e);
    if (f.includes("mmproj")) return bare([["general.architecture", "clip"], ["general.type", "mmproj"]], []);
    if (f.startsWith("Qwen3-VL")) return fake(spec({ arch: "qwen3vlmoe", name: "Qwen3-VL-30B-A3B" }), e);
    if (f.startsWith("qwen-image") || f.startsWith("flux")) return bare([], [{ name: "model.diffusion_model.transformer_blocks.0.img_mlp.out.weight", dims: [64, 64], type: 12 }, { name: "model.diffusion_model.time_text_embed.x", dims: [64], type: 0 }]);
    if (f.startsWith("te")) return safetensors(["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"]);
    return safetensors(["encoder.conv_in.weight", "decoder.conv_in.weight"]);
  });
  function fake(s: SyntheticModelSpec, e: Entry): ByteReader {
    const t = syntheticTensors(s); const { header, dataOffset } = buildGgufHeader(s, t);
    void e; return reader(header, dataOffset + t.reduce((n, x) => n + tensorByteSize(x), 0));
  }

  it("groups files that belong together and notes what is missing", async () => {
    const lib = buildLibrary(await files());
    expect(lib.vision).toHaveLength(1);
    expect(lib.vision[0]!.projector.name).toMatch(/mmproj/);
    const turbo = lib.image.find((p) => p.diffusion.name.startsWith("qwen-image"))!;
    expect(turbo.complete).toBe(true);
    expect(turbo.textEncoder!.name).toBe("te-int8.safetensors");   // the tested variant, not the 16 GB one
    const orphan = lib.image.find((p) => p.diffusion.name.startsWith("flux"))!;
    expect(orphan.complete).toBe(false);
    expect(describeLibrary(lib).join(" ")).toMatch(/image generator/);
  });

  it("picks the most capable model that fits, for each task", async () => {
    const lib = buildLibrary(await files());
    const m16 = { ramBytes: 16 * GiB };
    expect(pickLlm(lib, "chat", m16).choice!.name).toMatch(/Qwen3-30B|Qwen3-VL/);     // a 30B mixture of experts beats a 1B and a 7B
    expect(pickLlm(lib, "code", m16).choice!.coder || pickLlm(lib, "code", m16).choice!.moe).toBe(true);
    expect(pickVision(lib, m16).choice!.model.name).toMatch(/VL/);
    expect(pickImage(lib, m16).choice!.diffusion.name).toMatch(/turbo/);
    expect(pickImage(lib, m16).reason).toMatch(/few steps/);
  });

  it("falls back when memory is small or a model is measured to be slow", async () => {
    const lib = buildLibrary(await files());
    const tiny = pickLlm(lib, "chat", { ramBytes: 4 * GiB });
    expect(tiny.choice).toBeDefined();
    const slow = pickLlm(lib, "chat", { ramBytes: 16 * GiB, measuredTps: (n) => (/Qwen3/.test(n) ? 0.4 : undefined) });
    expect(slow.choice!.name).not.toMatch(/Qwen3/);
    expect(pickImage(buildLibrary([]), { ramBytes: 16 * GiB }).reason).toMatch(/no image model/);
  });
});
