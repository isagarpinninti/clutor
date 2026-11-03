#!/usr/bin/env python3
"""A tiny runnable OLMoE GGUF with random K-quant weights, plus an independent pure-Python forward pass, for testing the C engine without a real model.

    tiny_olmoe.py write OUT.gguf [--seed N]
    tiny_olmoe.py check ENGINE OUT.gguf          run the engine on a few prompts and compare its logprobs with this file's own forward pass

The reference here shares no code with c/src: the GGUF writer, the Q4_K / Q6_K block layouts (from ggml's definitions), RMSNorm, QK-norm,
NeoX rotary embeddings, attention, the softmax router and the SwiGLU experts are written again in plain Python. The engine multiplies K-quant
weights against 8-bit-quantized activations (as llama.cpp does) while this reference works in doubles, so logprobs agree to a tolerance, not exactly.
"""
import json, math, random, struct, subprocess, sys

# Two tiny architectures. olmoe: full-width QK-norm, equal head counts, raw softmax weights. qwen3moe: grouped-query attention, head_dim larger than
# embedding / heads, per-head QK-norm with one shared weight, and chosen weights renormalised to sum to 1.
E, FF, NEXP, USED, LAYERS = 256, 256, 8, 2, 2
ARCH, HEADS, KVH, HD, PRE, EPS, THETA, CTX = "olmoe", 4, 4, 64, "olmo", 1e-5, 10000.0, 64
NQ = NKV = E
QKPH = NORMTOP = False
QKMODE, ROPEPAIRS, ATTN_SCALE, EMB_SCALE, RES_SCALE, LOGIT_SCALE = 1, False, None, 1.0, 1.0, 1.0   # QKMODE: 0 none, 1 whole vector, 2 per head
ADDBOS = ROPEFREQS = False   # llama: a BOS token is prepended; rope_freqs.weight divides each rotary frequency
HYBRID = False   # qwen35: Gated DeltaNet layers (3 of every 4) and gated full-attention layers, a dense feed-forward in every layer
SSM_S, SSM_HV, SSM_HK, SSM_CK, ROPE_DIM = 32, 8, 4, 4, 32   # qwen35: state size, value heads, key heads, convolution width, rotated dimensions per head
DENSE = QKVBIAS = False   # qwen2/llama: no experts (one plain feed-forward per layer), bias vectors on Q/K/V, output matrix tied to the embedding

def configure(arch):
    global ARCH, HEADS, KVH, HD, PRE, EPS, THETA, NQ, NKV, QKPH, NORMTOP, QKMODE, ROPEPAIRS, ATTN_SCALE, EMB_SCALE, RES_SCALE, LOGIT_SCALE, DENSE, QKVBIAS, FF, ADDBOS, ROPEFREQS, HYBRID, LAYERS
    ARCH = arch
    QKMODE, ROPEPAIRS, ATTN_SCALE, EMB_SCALE, RES_SCALE, LOGIT_SCALE = 1, False, None, 1.0, 1.0, 1.0
    DENSE = QKVBIAS = ADDBOS = ROPEFREQS = HYBRID = False
    LAYERS = 2
    if arch == "llama":   # dense Llama 3 style: GQA 4/2, head 64, adjacent-pair rotary embeddings with frequency factors, tied output, BOS prepended
        HEADS, KVH, HD, PRE, EPS, THETA, QKPH, NORMTOP, QKMODE, DENSE, QKVBIAS, FF = 4, 2, 64, "llama-bpe", 1e-5, 500000.0, False, False, 0, True, False, 512
        ROPEPAIRS, ADDBOS, ROPEFREQS = True, True, True
    elif arch == "qwen35":   # hybrid Qwen3.5 style: 3 Gated DeltaNet layers then 1 gated attention layer; the query projection also makes an output gate; rotary on half of each head
        HEADS, KVH, HD, PRE, EPS, THETA, QKPH, NORMTOP, QKMODE, DENSE, QKVBIAS, FF = 4, 2, 64, "qwen35", 1e-6, 10000000.0, True, False, 2, True, False, 512
        HYBRID, LAYERS = True, 4
    elif arch == "qwen2":   # dense Qwen2.5 style: GQA 4/2, head 64, biases on Q/K/V, tied output, a feed-forward twice the width of an expert
        HEADS, KVH, HD, PRE, EPS, THETA, QKPH, NORMTOP, QKMODE, DENSE, QKVBIAS, FF = 4, 2, 64, "qwen2", 1e-6, 1000000.0, False, False, 0, True, True, 512
    elif arch == "qwen3moe":
        HEADS, KVH, HD, PRE, EPS, THETA, QKPH, NORMTOP, QKMODE = 4, 2, 128, "qwen2", 1e-6, 1000000.0, True, True, 2
    elif arch == "granitemoe":   # no QK-norm, adjacent-pair rotary embeddings, granite's four multipliers, renormalised expert weights, the "refact" pre-tokenizer
        HEADS, KVH, HD, PRE, EPS, THETA, QKPH, NORMTOP, QKMODE = 4, 2, 64, "refact", 1e-6, 10000.0, False, True, 0
        ROPEPAIRS, ATTN_SCALE, EMB_SCALE, RES_SCALE, LOGIT_SCALE = True, 0.015625, 12.0, 0.22, 6.0
    else:
        HEADS, KVH, HD, PRE, EPS, THETA, QKPH, NORMTOP = 4, 4, 64, "olmo", 1e-5, 10000.0, False, False
    NQ, NKV = HEADS * HD, KVH * HD

def byte_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

B2U = byte_to_unicode()
MERGES = ["Ġ t", "h e", "Ġ a", "i n", "Ġt he", "e r", "o n", "Ġ s", "r e", "a t", "Ġ o", "e n", "Ġ w", "i s", "o r", "Ġ c"]

def vocab():
    toks = [B2U[b] for b in range(256)]
    for m in MERGES:
        a, b = m.split(" ")
        toks.append(a + b)
    toks += ["<|endoftext|>", "<|padding|>"]
    types = [1] * (len(toks) - 2) + [3, 3]
    return toks, types

# ---- GGUF writer ----
def s(x):
    b = x.encode(); return struct.pack("<Q", len(b)) + b

def kv(key, typ, val):
    out = s(key) + struct.pack("<I", typ)
    if typ == 4: out += struct.pack("<I", val)
    elif typ == 5: out += struct.pack("<i", val)
    elif typ == 6: out += struct.pack("<f", val)
    elif typ == 7: out += struct.pack("<?", val)
    elif typ == 8: out += s(val)
    elif typ == 9:
        et, items = val
        out += struct.pack("<IQ", et, len(items))
        for it in items: out += s(it) if et == 8 else struct.pack("<i", it)
    return out

F32, Q4K, Q5K, Q6K = 0, 12, 13, 14

def f16(x): return struct.pack("<e", x)

def rand_q4k(rng, nblocks):
    out = bytearray()
    for _ in range(nblocks):
        d = 0.01
        sc = [rng.randint(1, 8) for _ in range(8)]
        mn = [min(63, int(round(7.5 * v)) + rng.randint(-1, 1)) for v in sc]   # keeps weights near zero-mean
        q = bytearray(12)
        for j in range(4):
            q[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6)
            q[j + 4] = (mn[j] & 63) | ((mn[j + 4] >> 4) << 6)
            q[j + 8] = (sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4)
        out += f16(d) + f16(d) + bytes(q) + bytes(rng.randrange(256) for _ in range(128))
    return bytes(out)

def rand_q5k(rng, nblocks):
    out = bytearray()
    for _ in range(nblocks):
        d = 0.01
        sc = [rng.randint(1, 8) for _ in range(8)]
        mn = [min(63, int(round(15.5 * v)) + rng.randint(-1, 1)) for v in sc]   # 5-bit values average 15.5: keeps weights near zero-mean
        q = bytearray(12)
        for j in range(4):
            q[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6)
            q[j + 4] = (mn[j] & 63) | ((mn[j + 4] >> 4) << 6)
            q[j + 8] = (sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4)
        out += f16(d) + f16(d) + bytes(q) + bytes(rng.randrange(256) for _ in range(32 + 128))   # d, dmin, scales, qh, qs
    return bytes(out)

def rand_q6k(rng, nblocks):
    out = bytearray()
    for _ in range(nblocks):
        out += bytes(rng.randrange(256) for _ in range(128 + 64))
        out += bytes((rng.randint(-2, 2)) & 255 for _ in range(16))
        out += f16(0.003)
    return bytes(out)

def write(path, seed=1):
    rng = random.Random(seed)
    toks, types = vocab()
    V = len(toks)
    tensors = []   # (name, dims, type, bytes)
    def add(name, dims, typ, data): tensors.append((name, dims, typ, data))
    def f32(n, scale=0.0, base=1.0):
        return struct.pack("<%df" % n, *[base + scale * rng.uniform(-1, 1) for _ in range(n)])
    add("token_embd.weight", [E, V], Q4K, rand_q4k(rng, V))
    add("output_norm.weight", [E], F32, f32(E, 0.2))
    if ROPEFREQS:   # llama 3 style: one divisor (>= 1) per rotated pair
        add("rope_freqs.weight", [HD // 2], F32, struct.pack("<%df" % (HD // 2), *[1.0 + 7.0 * (i / (HD // 2)) ** 2 for i in range(HD // 2)]))
    if not DENSE or HYBRID:   # a dense qwen2 ties the output projection to the embedding and stores no output.weight
        add("output.weight", [E, V], Q6K, rand_q6k(rng, V))
    for l in range(LAYERS):
        p = "blk.%d." % l
        if HYBRID:
            CC = 2 * SSM_S * SSM_HK + SSM_S * SSM_HV
            add(p + "attn_norm.weight", [E], F32, f32(E, 0.2))
            add(p + "post_attention_norm.weight", [E], F32, f32(E, 0.2))
            if (l + 1) % 4:
                add(p + "attn_qkv.weight", [E, CC], Q5K, rand_q5k(rng, CC))   # Q5_K, as in the real file: the only K-quant the other architectures do not use
                add(p + "attn_gate.weight", [E, SSM_S * SSM_HV], Q4K, rand_q4k(rng, SSM_S * SSM_HV))
                add(p + "ssm_alpha.weight", [E, SSM_HV], Q4K, rand_q4k(rng, SSM_HV))
                add(p + "ssm_beta.weight", [E, SSM_HV], Q4K, rand_q4k(rng, SSM_HV))
                add(p + "ssm_conv1d.weight", [SSM_CK, CC], F32, f32(SSM_CK * CC, 0.6, 0.0))
                add(p + "ssm_dt.bias", [SSM_HV], F32, f32(SSM_HV, 1.0, 0.0))
                add(p + "ssm_a", [SSM_HV], F32, struct.pack("<%df" % SSM_HV, *[-math.exp(rng.uniform(-2.0, 0.7)) for _ in range(SSM_HV)]))
                add(p + "ssm_norm.weight", [SSM_S], F32, f32(SSM_S, 0.2))
                add(p + "ssm_out.weight", [SSM_S * SSM_HV, E], Q4K, rand_q4k(rng, E * (SSM_S * SSM_HV // 256)))
            else:
                add(p + "attn_q.weight", [E, 2 * NQ], Q4K, rand_q4k(rng, 2 * NQ))
                add(p + "attn_k.weight", [E, NKV], Q4K, rand_q4k(rng, NKV))
                add(p + "attn_v.weight", [E, NKV], Q6K, rand_q6k(rng, NKV))
                add(p + "attn_output.weight", [NQ, E], Q4K, rand_q4k(rng, E * (NQ // 256)))
                add(p + "attn_q_norm.weight", [HD], F32, f32(HD, 0.2))
                add(p + "attn_k_norm.weight", [HD], F32, f32(HD, 0.2))
            add(p + "ffn_gate.weight", [E, FF], Q4K, rand_q4k(rng, FF))
            add(p + "ffn_up.weight", [E, FF], Q4K, rand_q4k(rng, FF))
            add(p + "ffn_down.weight", [FF, E], Q6K, rand_q6k(rng, E * (FF // 256)))
            continue
        add(p + "attn_norm.weight", [E], F32, f32(E, 0.2))
        add(p + "attn_q.weight", [E, NQ], Q4K, rand_q4k(rng, NQ))
        add(p + "attn_k.weight", [E, NKV], Q4K, rand_q4k(rng, NKV))
        add(p + "attn_v.weight", [E, NKV], Q6K, rand_q6k(rng, NKV))
        add(p + "attn_output.weight", [NQ, E], Q4K, rand_q4k(rng, E * (NQ // 256)))
        if QKMODE:
            add(p + "attn_q_norm.weight", [HD if QKPH else NQ], F32, f32(HD if QKPH else NQ, 0.2))
            add(p + "attn_k_norm.weight", [HD if QKPH else NKV], F32, f32(HD if QKPH else NKV, 0.2))
        add(p + "ffn_norm.weight", [E], F32, f32(E, 0.2))
        if QKVBIAS:
            add(p + "attn_q.bias", [NQ], F32, f32(NQ, 0.5, 0.0))
            add(p + "attn_k.bias", [NKV], F32, f32(NKV, 0.5, 0.0))
            add(p + "attn_v.bias", [NKV], F32, f32(NKV, 0.5, 0.0))
        if DENSE:
            add(p + "ffn_gate.weight", [E, FF], Q4K, rand_q4k(rng, FF))
            add(p + "ffn_up.weight", [E, FF], Q4K, rand_q4k(rng, FF))
            add(p + "ffn_down.weight", [FF, E], Q6K, rand_q6k(rng, E * (FF // 256)))
        else:
            add(p + "ffn_gate_inp.weight", [E, NEXP], F32, f32(E * NEXP, 0.5, 0.0))
            add(p + "ffn_gate_exps.weight", [E, FF, NEXP], Q4K, rand_q4k(rng, FF * NEXP))
            add(p + "ffn_up_exps.weight", [E, FF, NEXP], Q4K, rand_q4k(rng, FF * NEXP))
            add(p + "ffn_down_exps.weight", [FF, E, NEXP], Q6K, rand_q6k(rng, E * NEXP))
    A = ARCH
    meta = [
        kv("general.architecture", 8, A), kv("general.name", 8, "tiny-%s-test" % A),
        kv(A + ".block_count", 4, LAYERS), kv(A + ".context_length", 4, CTX), kv(A + ".embedding_length", 4, E),
        kv(A + (".expert_feed_forward_length" if ARCH == "qwen3moe" else ".feed_forward_length"), 4, FF), kv(A + ".attention.head_count", 4, HEADS), kv(A + ".attention.head_count_kv", 4, KVH),
        kv(A + ".rope.freq_base", 6, THETA), kv(A + ".attention.layer_norm_rms_epsilon", 6, EPS),
    ] + ([] if DENSE else [kv(A + ".expert_used_count", 4, USED), kv(A + ".expert_count", 4, NEXP)]) + [
    ] + ([kv(A + ".attention.key_length", 4, HD), kv(A + ".attention.value_length", 4, HD)] if ARCH == "qwen3moe" else []) + (
        [kv(A + ".rope.dimension_count", 4, HD), kv(A + ".attention.scale", 6, ATTN_SCALE), kv(A + ".embedding_scale", 6, EMB_SCALE), kv(A + ".residual_scale", 6, RES_SCALE), kv(A + ".logit_scale", 6, LOGIT_SCALE)] if ARCH == "granitemoe" else [kv(A + ".rope.dimension_count", 4, HD)] if ARCH == "llama" else
        [kv(A + ".attention.key_length", 4, HD), kv(A + ".attention.value_length", 4, HD), kv(A + ".rope.dimension_count", 4, ROPE_DIM), kv(A + ".ssm.conv_kernel", 4, SSM_CK), kv(A + ".ssm.state_size", 4, SSM_S),
         kv(A + ".ssm.group_count", 4, SSM_HK), kv(A + ".ssm.time_step_rank", 4, SSM_HV), kv(A + ".ssm.inner_size", 4, SSM_S * SSM_HV), kv(A + ".full_attention_interval", 4, 4)] if ARCH == "qwen35" else []) + [
        kv("tokenizer.ggml.model", 8, "gpt2"), kv("tokenizer.ggml.pre", 8, PRE),
        kv("tokenizer.ggml.tokens", 9, (8, toks)), kv("tokenizer.ggml.token_type", 9, (5, types)), kv("tokenizer.ggml.merges", 9, (8, MERGES)),
        kv("tokenizer.ggml.bos_token_id", 4, V - 2), kv("tokenizer.ggml.eos_token_id", 4, V - 2), kv("tokenizer.ggml.padding_token_id", 4, V - 1),
    ] + ([] if ARCH == "llama" else [kv("tokenizer.ggml.add_bos_token", 7, ADDBOS)]) + [   # a real Llama 3 file has no add_bos_token key: the engine must default to adding BOS, as llama.cpp does
    ]
    infos, off = b"", 0
    for name, dims, typ, data in tensors:
        infos += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", typ, off)
        off += (len(data) + 31) // 32 * 32
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)) + b"".join(meta) + infos
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head)
        for name, dims, typ, data in tensors:
            f.write(data); f.write(b"\0" * ((-len(data)) % 32))

# ---- reference: parse our own file back, dequantize, run the model in doubles ----
def read_gguf(path):
    buf = open(path, "rb").read()
    pos = 4
    def u32():
        nonlocal pos; v = struct.unpack_from("<I", buf, pos)[0]; pos += 4; return v
    def u64():
        nonlocal pos; v = struct.unpack_from("<Q", buf, pos)[0]; pos += 8; return v
    def st():
        nonlocal pos; n = u64(); v = buf[pos:pos + n].decode(); pos += n; return v
    u32(); nt = u64(); nkv = u64()
    sz = {4: 4, 5: 4, 6: 4, 7: 1}
    for _ in range(nkv):
        st(); t = u32()
        if t == 8: st()
        elif t == 9:
            et = u32(); n = u64()
            for _ in range(n):
                if et == 8: st()
                else: pos += 4
        else: pos += sz[t]
    infos = []
    for _ in range(nt):
        name = st(); nd = u32(); dims = [u64() for _ in range(nd)]; typ = u32(); off = u64()
        infos.append((name, dims, typ, off))
    base = (pos + 31) // 32 * 32
    return buf, base, {n: (d, t, o) for n, d, t, o in infos}

def kscale(j, q):
    if j < 4: return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)

def deq_q4k(b):
    d, dm = struct.unpack("<ee", b[:4]); sc = b[4:16]; qs = b[16:144]; out = []
    for g in range(4):
        s1, m1 = kscale(2 * g, sc); s2, m2 = kscale(2 * g + 1, sc)
        blk = qs[32 * g:32 * g + 32]
        out += [d * s1 * (x & 15) - dm * m1 for x in blk]
        out += [d * s2 * (x >> 4) - dm * m2 for x in blk]
    return out

def deq_q5k(b):
    d, dm = struct.unpack("<ee", b[:4]); sc = b[4:16]; qh = b[16:48]; qs = b[48:176]; out = []
    for c in range(4):
        s1, m1 = kscale(2 * c, sc); s2, m2 = kscale(2 * c + 1, sc)
        out += [d * s1 * ((qs[32 * c + l] & 15) + (16 if qh[l] & (1 << (2 * c)) else 0)) - dm * m1 for l in range(32)]
        out += [d * s2 * ((qs[32 * c + l] >> 4) + (16 if qh[l] & (1 << (2 * c + 1)) else 0)) - dm * m2 for l in range(32)]
    return out

def deq_q6k(b):
    ql, qh = b[:128], b[128:192]
    sc = struct.unpack("<16b", b[192:208]); d = struct.unpack("<e", b[208:210])[0]; out = [0.0] * 256
    for h in range(2):
        for l in range(32):
            lo, hi = ql[64 * h + l], ql[64 * h + l + 32]; q = qh[32 * h + l]
            vals = [(lo & 15) | ((q & 3) << 4), (hi & 15) | (((q >> 2) & 3) << 4), (lo >> 4) | (((q >> 4) & 3) << 4), (hi >> 4) | (((q >> 6) & 3) << 4)]
            for k in range(4):
                e = 128 * h + 32 * k + l
                out[e] = d * sc[e // 16] * (vals[k] - 32)
    return out

def tensor_rows(g, name):
    buf, base, T = g
    dims, typ, off = T[name]
    cols = dims[0]; rows = 1
    for x in dims[1:]: rows *= x
    bs, bb = {F32: (1, 4), Q4K: (256, 144), Q5K: (256, 176), Q6K: (256, 210)}[typ]
    rb = cols // bs * bb
    out = []
    for r in range(rows):
        raw = buf[base + off + r * rb: base + off + (r + 1) * rb]
        if typ == F32: out.append(list(struct.unpack("<%df" % cols, raw)))
        else:
            row = []
            for k in range(cols // 256):
                blk = raw[k * bb:(k + 1) * bb]
                row += deq_q4k(blk) if typ == Q4K else deq_q5k(blk) if typ == Q5K else deq_q6k(blk)
            out.append(row)
    return out

def rnd(v):   # C's roundf: half away from zero
    return int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)

def q8k_roundtrip(v):
    """What the engine's 8-bit activation quantization does to a vector before every K-quant matrix product: blocks of 256, scale = max|x| / 127."""
    out = []
    for b in range(0, len(v), 256):
        blk = v[b:b + 256]; amax = max(abs(a) for a in blk); d = amax / 127.0; inv = 1.0 / d if d else 0.0
        out += [rnd(a * inv) * d for a in blk]
    return out

def dot(a, b): return sum(x * y for x, y in zip(a, b))
def matvec(W, x): return [dot(r, x) for r in W]
def rms(x, w): s = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + EPS); return [v * s * wi for v, wi in zip(x, w)]
def rope(v, pos, ff=None):
    if ROPEPAIRS:   # adjacent pairs (2i, 2i+1)
        o = v[:]
        for i in range(len(v) // 2):
            t = pos * THETA ** (-2.0 * i / len(v)) / (ff[i] if ff else 1.0); c, sn = math.cos(t), math.sin(t)
            o[2 * i] = v[2 * i] * c - v[2 * i + 1] * sn; o[2 * i + 1] = v[2 * i] * sn + v[2 * i + 1] * c
        return o
    h = len(v) // 2; o = v[:]
    for i in range(h):
        t = pos * THETA ** (-2.0 * i / len(v)) / (ff[i] if ff else 1.0); c, sn = math.cos(t), math.sin(t)
        o[i] = v[i] * c - v[i + h] * sn; o[i + h] = v[i] * sn + v[i + h] * c
    return o

class Ref:
    def mv(self, W, x):   # a product with K-quant weights: with qa=True the input goes through the engine's 8-bit rounding first
        return matvec(W, q8k_roundtrip(x) if self.qa else x)
    def reset(self):
        self.K = [[] for _ in range(LAYERS)]; self.V = [[] for _ in range(LAYERS)]
        if HYBRID:
            CC = 2 * SSM_S * SSM_HK + SSM_S * SSM_HV
            self.conv = [[[0.0] * (SSM_CK - 1) for _ in range(CC)] for _ in range(LAYERS)]
            self.state = [[[[0.0] * SSM_S for _ in range(SSM_S)] for _ in range(SSM_HV)] for _ in range(LAYERS)]
    def __init__(self, path, qa=False):
        self.qa = qa
        g = read_gguf(path); self.g = g
        if HYBRID: self.init_hybrid(g); return
        self.ff = tensor_rows(g, "rope_freqs.weight")[0] if ROPEFREQS else None
        self.emb = tensor_rows(g, "token_embd.weight"); self.out = self.emb if DENSE else tensor_rows(g, "output.weight"); self.on = tensor_rows(g, "output_norm.weight")[0]
        self.L = []
        for l in range(LAYERS):
            p = "blk.%d." % l; r = lambda n: tensor_rows(g, p + n)
            exps = lambda n: r(n)
            if DENSE:
                self.L.append(dict(an=r("attn_norm.weight")[0], q=r("attn_q.weight"), k=r("attn_k.weight"), v=r("attn_v.weight"), o=r("attn_output.weight"), fn=r("ffn_norm.weight")[0],
                                   bq=r("attn_q.bias")[0] if QKVBIAS else None, bk=r("attn_k.bias")[0] if QKVBIAS else None, bv=r("attn_v.bias")[0] if QKVBIAS else None, qn=None, kn=None, router=None,
                                   dgate=r("ffn_gate.weight"), dup=r("ffn_up.weight"), ddown=r("ffn_down.weight")))
                continue
            gate, up, down = exps("ffn_gate_exps.weight"), exps("ffn_up_exps.weight"), exps("ffn_down_exps.weight")
            self.L.append(dict(an=r("attn_norm.weight")[0], q=r("attn_q.weight"), k=r("attn_k.weight"), v=r("attn_v.weight"), o=r("attn_output.weight"),
                               qn=r("attn_q_norm.weight")[0] if QKMODE else None, kn=r("attn_k_norm.weight")[0] if QKMODE else None, fn=r("ffn_norm.weight")[0], router=r("ffn_gate_inp.weight"),
                               gate=[gate[e * FF:(e + 1) * FF] for e in range(NEXP)], up=[up[e * FF:(e + 1) * FF] for e in range(NEXP)],
                               down=[down[e * E:(e + 1) * E] for e in range(NEXP)]))
        self.K = [[] for _ in range(LAYERS)]; self.V = [[] for _ in range(LAYERS)]
    def init_hybrid(self, g):
        self.emb = tensor_rows(g, "token_embd.weight"); self.out = tensor_rows(g, "output.weight"); self.on = tensor_rows(g, "output_norm.weight")[0]
        self.L = []
        for l in range(LAYERS):
            p = "blk.%d." % l; r = lambda n: tensor_rows(g, p + n)
            d = dict(an=r("attn_norm.weight")[0], pn=r("post_attention_norm.weight")[0], gate=r("ffn_gate.weight"), up=r("ffn_up.weight"), down=r("ffn_down.weight"), rec=bool((l + 1) % 4))
            if d["rec"]:
                d.update(qkv=r("attn_qkv.weight"), z=r("attn_gate.weight"), al=r("ssm_alpha.weight"), be=r("ssm_beta.weight"), out=r("ssm_out.weight"), conv=r("ssm_conv1d.weight"),
                         a=r("ssm_a")[0], dt=r("ssm_dt.bias")[0], nm=r("ssm_norm.weight")[0])
            else:
                d.update(q=r("attn_q.weight"), k=r("attn_k.weight"), v=r("attn_v.weight"), o=r("attn_output.weight"), qn=r("attn_q_norm.weight")[0], kn=r("attn_k_norm.weight")[0])
            self.L.append(d)
        self.reset()

    def step_hybrid(self, tok):
        # Written from llama.cpp's qwen35 graph (src/models/qwen35.cpp, delta-net-base.cpp), independently of c/src/forward.c.
        silu = lambda a: a / (1 + math.exp(-a))
        S, HV, HK, CK = SSM_S, SSM_HV, SSM_HK, SSM_CK; KEYD = S * HK; CC = 2 * KEYD + S * HV
        x = self.emb[tok][:]; grp = HEADS // KVH
        pos = len(self.K[3])
        for l, L in enumerate(self.L):
            xn = rms(x, L["an"])
            if L["rec"]:
                qkv, z, be, al = self.mv(L["qkv"], xn), self.mv(L["z"], xn), self.mv(L["be"], xn), self.mv(L["al"], xn)
                beta = [1 / (1 + math.exp(-b)) for b in be]
                gdec = [(a + dt if a + dt > 20 else math.log(1 + math.exp(a + dt))) * A for a, dt, A in zip(al, L["dt"], L["a"])]
                out = []
                for c in range(CC):
                    win = self.conv[l][c] + [qkv[c]]
                    out.append(silu(sum(w * v for w, v in zip(L["conv"][c], win)))); self.conv[l][c] = win[1:]
                l2 = lambda v: [a / math.sqrt(sum(b * b for b in v) + EPS) for a in v]
                q = sum((l2(out[g * S:(g + 1) * S]) for g in range(HK)), []); k = sum((l2(out[KEYD + g * S:KEYD + (g + 1) * S]) for g in range(HK)), []); v = out[2 * KEYD:]
                att = []
                for h in range(HV):
                    kk = k[(h % HK) * S:(h % HK + 1) * S]; qq = [a / math.sqrt(S) for a in q[(h % HK) * S:(h % HK + 1) * S]]; vv = v[h * S:(h + 1) * S]
                    M = self.state[l][h]; eg = math.exp(gdec[h]); oh = []
                    for j in range(S):
                        M[j] = [a * eg for a in M[j]]
                        d = (vv[j] - dot(M[j], kk)) * beta[h]
                        M[j] = [a + b * d for a, b in zip(M[j], kk)]
                        oh.append(dot(M[j], qq))
                    att += [a * silu(b) for a, b in zip(rms(oh, L["nm"]), z[h * S:(h + 1) * S])]
                x = [a + b for a, b in zip(x, self.mv(L["out"], att))]
            else:
                qf, k, v = self.mv(L["q"], xn), self.mv(L["k"], xn), self.mv(L["v"], xn)
                qs = [rms(qf[h * 2 * HD:h * 2 * HD + HD], L["qn"]) for h in range(HEADS)]; gates = [qf[h * 2 * HD + HD:(h + 1) * 2 * HD] for h in range(HEADS)]
                ks = [rms(k[h * HD:(h + 1) * HD], L["kn"]) for h in range(KVH)]
                rot = lambda hv: rope(hv[:ROPE_DIM], pos) + hv[ROPE_DIM:]
                qs = [rot(a) for a in qs]; ks = [rot(a) for a in ks]
                self.K[l].append(sum(ks, [])); self.V[l].append(v)
                att = []
                for h in range(HEADS):
                    sl = slice((h // grp) * HD, (h // grp + 1) * HD)
                    sc = [dot(qs[h], kk[sl]) / math.sqrt(HD) for kk in self.K[l]]
                    m = max(sc); w = [math.exp(a - m) for a in sc]; zz = sum(w)
                    o = [sum(w[t] / zz * self.V[l][t][sl][i] for t in range(len(w))) for i in range(HD)]
                    att += [a / (1 + math.exp(-gt)) for a, gt in zip(o, gates[h])]
                x = [a + b for a, b in zip(x, self.mv(L["o"], att))]
            xn = rms(x, L["pn"]); gg, uu = self.mv(L["gate"], xn), self.mv(L["up"], xn)
            x = [a + b for a, b in zip(x, self.mv(L["down"], [silu(a) * b for a, b in zip(gg, uu)]))]
        lo = self.mv(self.out, rms(x, self.on)); m = max(lo); lse = m + math.log(sum(math.exp(a - m) for a in lo))
        return [a - lse for a in lo]

    def step(self, tok):
        if HYBRID: return self.step_hybrid(tok)
        x = [EMB_SCALE * a for a in self.emb[tok]]; pos = len(self.K[0]); grp = HEADS // KVH
        for l, L in enumerate(self.L):
            xn = rms(x, L["an"]); q = self.mv(L["q"], xn); k = self.mv(L["k"], xn); v = self.mv(L["v"], xn)
            if QKVBIAS:
                q = [a + b for a, b in zip(q, L["bq"])]; k = [a + b for a, b in zip(k, L["bk"])]; v = [a + b for a, b in zip(v, L["bv"])]
            if QKMODE == 2:   # one weight of head_dim, applied to every head separately
                q = sum((rms(q[h * HD:(h + 1) * HD], L["qn"]) for h in range(HEADS)), []); k = sum((rms(k[h * HD:(h + 1) * HD], L["kn"]) for h in range(KVH)), [])
            elif QKMODE == 1:
                q = rms(q, L["qn"]); k = rms(k, L["kn"])
            q = sum((rope(q[h * HD:(h + 1) * HD], pos, self.ff) for h in range(HEADS)), []); k = sum((rope(k[h * HD:(h + 1) * HD], pos, self.ff) for h in range(KVH)), [])
            self.K[l].append(k); self.V[l].append(v)
            att = []
            for h in range(HEADS):
                qs = q[h * HD:(h + 1) * HD]; ks = slice((h // grp) * HD, (h // grp + 1) * HD)
                sc = [dot(qs, kk[ks]) * (ATTN_SCALE if ATTN_SCALE else 1.0 / math.sqrt(HD)) for kk in self.K[l]]
                m = max(sc); w = [math.exp(a - m) for a in sc]; z = sum(w)
                att += [sum(w[t] / z * self.V[l][t][ks][i] for t in range(len(w))) for i in range(HD)]
            x = [a + RES_SCALE * b for a, b in zip(x, self.mv(L["o"], att))]
            if DENSE:
                xn = rms(x, L["fn"]); gg, uu = self.mv(L["dgate"], xn), self.mv(L["dup"], xn)
                hh = [a / (1 + math.exp(-a)) * b for a, b in zip(gg, uu)]
                x = [a + RES_SCALE * b for a, b in zip(x, self.mv(L["ddown"], hh))]
                continue
            xn = rms(x, L["fn"]); lg = matvec(L["router"], xn); m = max(lg); p = [math.exp(a - m) for a in lg]; z = sum(p); p = [a / z for a in p]
            top = sorted(range(NEXP), key=lambda e: (-p[e], e))[:USED]
            wts = {e: p[e] for e in top}
            if NORMTOP:
                zz = sum(wts.values()); wts = {e: w / zz for e, w in wts.items()}
            moe = [0.0] * E
            for e in top:
                gg, uu = self.mv(L["gate"][e], xn), self.mv(L["up"][e], xn)
                hh = [a / (1 + math.exp(-a)) * b for a, b in zip(gg, uu)]
                y = self.mv(L["down"][e], hh)
                moe = [a + wts[e] * b for a, b in zip(moe, y)]
            x = [a + RES_SCALE * b for a, b in zip(x, moe)]
        lo = [a / LOGIT_SCALE for a in self.mv(self.out, rms(x, self.on))]; m = max(lo); lse = m + math.log(sum(math.exp(a - m) for a in lo))
        return [a - lse for a in lo]

def run_engine(engine, args, float_math):
    import os
    env = dict(os.environ, CLUTOR_FLOAT_MATH="1" if float_math else "0")
    return subprocess.run([engine] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, env=env).stdout

def check(engine, path):
    """Two modes, each against a reference built for it.
    Float math (CLUTOR_FLOAT_MATH=1) must match the reference to float32 rounding: a bug in attention, rope, the router, biases, the dense or expert feed-forwards shows here.
    The default integer kernels (K-quant weights times 8-bit activations) are checked against the same reference with the engine's 8-bit activation rounding applied to every
    K-quant product, so a broken vector path shows as a real difference and not as noise; the agreement is limited only by rounding ties and float accumulation order."""
    toks, _ = vocab()
    prompts = ["the cat is in the room", "a rat on the mat", "we are in the car and on the road"]
    ref = Ref(path)
    bad = 0
    for text in prompts:
        ids = [int(t) for t in run_engine(engine, ["tokenize", path, text], False).split()]
        if ADDBOS: ids = [len(toks) - 2] + ids   # the engine prepends BOS (the generator makes <|endoftext|> the BOS token)
        # The recurrent state carries any 8-bit rounding flip forward, so the integer path of a hybrid model is checked with wider limits (measured worst case over seeds
        # and builds: mean 0.035, max 0.13); the float path, which has no such flips, keeps the strict limits and is the check of the logic.
        int_max, int_mean = (0.3, 0.08) if HYBRID else (0.05, 0.01)
        for float_math, max_tol, mean_tol, label in ((True, 1e-3, 1e-3, "float math  "), (False, int_max, int_mean, "integer dots")):
            ref.qa = not float_math; ref.reset()
            for t in ids: lp = ref.step(t)
            gap = sorted(lp, reverse=True)
            res = json.loads(run_engine(engine, ["logits", path, text, "--top", str(len(toks))], float_math))
            assert res["tokens"] == ids, (res["tokens"], ids)
            eng = {t["id"]: t["logprob"] for t in res["top"]}
            diffs = [abs(eng[i] - lp[i]) for i in range(len(toks))]
            top_ref = max(range(len(toks)), key=lambda i: lp[i]); top_eng = res["top"][0]["id"]
            ok = max(diffs) < max_tol and sum(diffs) / len(diffs) < mean_tol and (top_ref == top_eng or gap[0] - gap[1] < 0.3)
            bad += not ok
            print("OK  " if ok else "FAIL", label, "%-36r tokens=%2d  mean|dlogprob|=%.5f max=%.5f  (limits %g / %g)" % (text, len(ids), sum(diffs) / len(diffs), max(diffs), mean_tol, max_tol))
    print("failed:", bad)
    return 1 if bad else 0

if __name__ == "__main__":
    if "--arch" in sys.argv:
        configure(sys.argv[sys.argv.index("--arch") + 1])
    if sys.argv[1] == "write":
        write(sys.argv[2], int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1)
    elif sys.argv[1] == "check":
        sys.exit(check(sys.argv[2], sys.argv[3]))
