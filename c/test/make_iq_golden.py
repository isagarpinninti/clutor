#!/usr/bin/env python3
"""Writes test/iq_golden.bin: random blocks of every IQ type and what gguf-py (ggml's reference implementation in Python) dequantizes them to.

    make_iq_golden.py > test/iq_golden.bin        (needs: pip install gguf numpy)

The engine's test reads the file, so the check needs neither gguf nor numpy. File layout, repeated per type: u32 type, u32 number of blocks, the blocks, then the floats.
"""
import struct, sys
import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

TYPES = [(T.IQ2_XXS, 256, 66), (T.IQ2_XS, 256, 74), (T.IQ2_S, 256, 82), (T.IQ3_XXS, 256, 98), (T.IQ3_S, 256, 110), (T.IQ1_S, 256, 50), (T.IQ1_M, 256, 56), (T.IQ4_NL, 32, 18), (T.IQ4_XS, 256, 136)]
rng = np.random.default_rng(20260610)
N = 6
out = sys.stdout.buffer
for t, elems, nbytes in TYPES:
    blocks = rng.integers(0, 256, size=(N, nbytes), dtype=np.uint8)
    for b in range(N):
        d16 = np.float16(0.01 + 0.03 * rng.random()).view(np.uint16)
        if t == T.IQ1_M:   # the half-precision scale is spread over the top nibbles of four 16-bit words at the end
            for k in range(4):
                w = int(blocks[b, 48 + 2 * k]) | int(blocks[b, 49 + 2 * k]) << 8
                w = (w & 0x0FFF) | (((int(d16) >> (4 * k)) & 0xF) << 12)
                blocks[b, 48 + 2 * k] = w & 255; blocks[b, 49 + 2 * k] = w >> 8
        else:
            blocks[b, 0] = int(d16) & 255; blocks[b, 1] = int(d16) >> 8
    vals = dequantize(blocks, t).astype(np.float32)
    assert vals.shape == (N, elems), vals.shape
    out.write(struct.pack("<II", int(t), N)); out.write(blocks.tobytes()); out.write(vals.tobytes())
