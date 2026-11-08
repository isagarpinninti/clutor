#!/usr/bin/env python3
"""
An independent numpy implementation of the Qwen3-VL image encoder, to check the C one (src/vision.c) against: `clutor-engine vision ... --out PREFIX` writes its outputs, this recomputes them
and prints the largest differences. It needs a real projector file (mmproj-*.gguf) and a picture (binary PPM); without them there is nothing to check.

    python3 test/vision_ref.py build/clutor-engine mmproj.gguf picture.ppm
"""
import struct, subprocess, sys, tempfile, os
import numpy as np

def read_gguf(path):
    f = open(path, "rb")
    def r(fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    f.read(4); r("<I"); nt, nk = r("<QQ")
    def rstr():
        n, = r("<Q"); return f.read(n).decode()
    def rval(t):
        m = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
        if t in m: return r(m[t])[0]
        if t == 7: return bool(r("<B")[0])
        if t == 8: return rstr()
        et, = r("<I"); n, = r("<Q"); return [rval(et) for _ in range(n)]
    kv = {}
    for _ in range(nk):
        k = rstr(); t, = r("<I"); kv[k] = rval(t)
    infos = []
    for _ in range(nt):
        n = rstr(); nd, = r("<I"); dims = [r("<Q")[0] for _ in range(nd)]; ty, = r("<I"); off, = r("<Q"); infos.append((n, dims, ty, off))
    align = kv.get("general.alignment", 32)
    base = (f.tell() + align - 1) // align * align
    T = {}
    for n, dims, ty, off in infos:
        cnt = int(np.prod(dims)); f.seek(base + off)
        dt = {0: np.float32, 1: np.float16}[ty]
        T[n] = np.frombuffer(f.read(cnt * np.dtype(dt).itemsize), dtype=dt).astype(np.float32).reshape(dims[::-1])   # ggml lists the fastest dimension first
    return kv, T

def ln(x, w, b, eps): m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + eps) * w + b
def gelu_tanh(x): return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
def gelu_erf(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x * 0.7071067811865476))

def encode(kv, T, img):
    H, W, _ = img.shape
    E = kv["clip.vision.embedding_length"]; depth = kv["clip.vision.block_count"]; heads = kv["clip.vision.attention.head_count"]; hd = E // heads
    eps = kv["clip.vision.attention.layer_norm_epsilon"]; ds_flags = kv["clip.vision.is_deepstack_layers"]
    gh, gw = H // 16, W // 16
    # patches in merge order: for each 2x2 block row-major, inside it (0,0),(0,1),(1,0),(1,1)
    order = [(br * 2 + mh, bc * 2 + mw) for br in range(gh // 2) for bc in range(gw // 2) for mh in range(2) for mw in range(2)]
    x = img.astype(np.float32) / 127.5 - 1.0
    P = np.stack([x[ph * 16:(ph + 1) * 16, pw * 16:(pw + 1) * 16, :].transpose(2, 0, 1).reshape(-1) for ph, pw in order])   # (c, y, x) order
    w = T["v.patch_embd.weight"] + T["v.patch_embd.weight.1"]            # [E, 3, 16, 16]
    h = P @ w.reshape(E, -1).T + T["v.patch_embd.bias"]
    # position embeddings: bilinear over the learned square grid, corners aligned
    pe = T["v.position_embd.weight"]; S = int(round(pe.shape[0] ** 0.5))
    hi = np.linspace(0, S - 1, gh); wi = np.linspace(0, S - 1, gw)
    pos = []
    for ph, pw in order:
        fh, fw = hi[ph], wi[pw]; h0, w0 = int(np.floor(fh)), int(np.floor(fw)); h1, w1 = min(h0 + 1, S - 1), min(w0 + 1, S - 1); dh, dw = fh - h0, fw - w0
        pos.append((1 - dh) * (1 - dw) * pe[h0 * S + w0] + (1 - dh) * dw * pe[h0 * S + w1] + dh * (1 - dw) * pe[h1 * S + w0] + dh * dw * pe[h1 * S + w1])
    h = h + np.stack(pos)
    # rotary angles
    half = hd // 2; qd = half // 2
    inv = 10000.0 ** (-(2 * np.arange(qd)) / half)
    ang = np.stack([np.concatenate([ph * inv, pw * inv]) for ph, pw in order])        # N x half
    cos = np.cos(ang)[:, None, :]; sin = np.sin(ang)[:, None, :]
    def rope(t):   # t: N x heads x hd, rotate_half layout
        a, b = t[..., :half], t[..., half:]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], -1)
    ds = []
    def merger(v, p, norm_first_name=None, norm_after=None):
        pass
    for l in range(depth):
        pre = f"v.blk.{l}."
        n1 = ln(h, T[pre + "ln1.weight"], T[pre + "ln1.bias"], eps)
        qkv = n1 @ T[pre + "attn_qkv.weight"].T + T[pre + "attn_qkv.bias"]
        q, k, v = [qkv[:, i * E:(i + 1) * E].reshape(-1, heads, hd) for i in range(3)]
        q, k = rope(q), rope(k)
        s = np.einsum("nhd,mhd->hnm", q, k) / np.sqrt(hd); s = s - s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
        a = np.einsum("hnm,mhd->nhd", p, v).reshape(-1, E)
        h = h + a @ T[pre + "attn_out.weight"].T + T[pre + "attn_out.bias"]
        n2 = ln(h, T[pre + "ln2.weight"], T[pre + "ln2.bias"], eps)
        f = gelu_tanh(n2 @ T[pre + "ffn_up.weight"].T + T[pre + "ffn_up.bias"])
        h = h + f @ T[pre + "ffn_down.weight"].T + T[pre + "ffn_down.bias"]
        if ds_flags[l]:
            g = h.reshape(-1, 4 * E)
            n = ln(g, T[f"v.deepstack.{l}.norm.weight"], T[f"v.deepstack.{l}.norm.bias"], eps)
            m = gelu_erf(n @ T[f"v.deepstack.{l}.fc1.weight"].T + T[f"v.deepstack.{l}.fc1.bias"])
            ds.append(m @ T[f"v.deepstack.{l}.fc2.weight"].T + T[f"v.deepstack.{l}.fc2.bias"])
    g = ln(h, T["v.post_ln.weight"], T["v.post_ln.bias"], eps).reshape(-1, 4 * E)
    m = gelu_erf(g @ T["mm.0.weight"].T + T["mm.0.bias"])
    return m @ T["mm.2.weight"].T + T["mm.2.bias"], ds

if __name__ == "__main__":
    engine, mmproj, ppm = sys.argv[1:4]
    with open(ppm, "rb") as f:
        assert f.readline().strip() == b"P6"; w, h = map(int, f.readline().split()); f.readline(); img = np.frombuffer(f.read(), np.uint8).reshape(h, w, 3)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "o")
        subprocess.run([engine, "vision", mmproj, ppm, "--out", out], check=True, stdout=subprocess.DEVNULL)
        kv, T = read_gguf(mmproj)
        emb, ds = encode(kv, T, img)
        names = [("emb", emb)] + [(f"ds{i}", x) for i, x in enumerate(ds)]
        worst = 0.0
        for nm, ref in names:
            got = np.fromfile(f"{out}.{nm}.f32", dtype=np.float32).reshape(ref.shape)
            err = np.abs(got - ref).max(); rel = err / (np.abs(ref).max() + 1e-9); worst = max(worst, rel)
            print(f"{nm:4s} max |C - numpy| = {err:.5f}  ({rel * 100:.3f}% of the largest value)  rms {np.sqrt((ref ** 2).mean()):.4f}")
        print("OK" if worst < 0.01 else "MISMATCH")
        sys.exit(0 if worst < 0.01 else 1)
