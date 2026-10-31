#!/usr/bin/env python3
"""Compares the router's choices with llama.cpp's. Usage: compare_routing.py EVAL_CALLBACK_OUTPUT TRACE
EVAL_CALLBACK_OUTPUT comes from `llama-eval-callback -m MODEL -p PROMPT -n 1 -ngl 0`; TRACE (a v2 route trace of raw router scores) from `clutor-engine logits MODEL PROMPT --trace TRACE`.
llama-eval-callback prints only the first and last three of each token's eight experts (and, for a prompt of more than six tokens, only the first and last three tokens), so that is what is compared: the share of those
reference experts that are inside this engine's top-8 for the same token and layer."""
import re, sys
txt = open(sys.argv[1]).read().split("\n")
ref, i = {}, 0
while i < len(txt):
    m = re.search(r"ffn_moe_topk-(\d+) = ", txt[i])
    if m:
        L, rows, j = int(m.group(1)), [], i + 1
        while j < len(txt) and "sum =" not in txt[j]:
            nums = re.findall(r"\d+\.\d+", txt[j])
            if len(nums) == 6: rows.append([int(float(x)) for x in nums])
            j += 1
        ref[L] = rows
        i = j
    i += 1
mine = {}
for line in open(sys.argv[2]):
    if line.startswith("#"): continue
    f = line.split()
    scores = [float(v) for v in f[2:]]
    top8 = sorted(range(len(scores)), key=lambda e: (-scores[e], e))[:8]   # best first, ties to the lower id
    mine.setdefault(int(f[1]), {})[int(f[0])] = top8
hit = tot = 0
ntok = 1 + max(max(rows) for rows in mine.values())
for L, rows in sorted(ref.items()):
    # llama.cpp prints only the first and last three tokens when a prompt has more than six; the last layer keeps only the final token
    if len(rows) == ntok: idx = list(range(ntok))
    elif len(rows) == 6 and ntok > 6: idx = [0, 1, 2, ntok - 3, ntok - 2, ntok - 1]
    elif len(rows) == 1: idx = [ntok - 1]
    else: raise SystemExit(f"layer {L}: {len(rows)} printed token rows do not fit a {ntok}-token prompt")
    for r, t in zip(rows, idx):
        mm = set(mine[L][t])
        hit += sum(x in mm for x in r); tot += 6
print(f"{hit}/{tot} reference experts are in this engine's top-8 = {hit / tot:.1%}")
sys.exit(0 if hit / tot >= 0.9 else 1)
