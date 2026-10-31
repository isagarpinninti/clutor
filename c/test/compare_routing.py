#!/usr/bin/env python3
"""Compares the router's choices with llama.cpp's. Usage: compare_routing.py EVAL_CALLBACK_OUTPUT TRACE
EVAL_CALLBACK_OUTPUT comes from `llama-eval-callback -m MODEL -p PROMPT -n 1 -ngl 0`; TRACE from `clutor-engine logits MODEL PROMPT --trace TRACE`.
llama-eval-callback prints only the first and last three of each token's eight experts, so that is what is compared: the share of those
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
    p = list(map(int, line.split()))
    mine.setdefault(p[1], {})[p[0]] = p[2:]
hit = tot = 0
for L, rows in sorted(ref.items()):
    off = len(mine[L]) - len(rows)   # the last layer keeps only the final token
    for k, r in enumerate(rows):
        mm = set(mine[L][k + off])
        hit += sum(x in mm for x in r); tot += 6
print(f"{hit}/{tot} reference experts are in this engine's top-8 = {hit / tot:.1%}")
sys.exit(0 if hit / tot >= 0.9 else 1)
