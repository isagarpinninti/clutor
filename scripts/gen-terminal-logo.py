#!/usr/bin/env python3
"""Rasterizes the logo (assets/clutor-logo.svg: a honeycomb of six hexagons, the middle one faded) into the small grid that packages/cli/src/tui/welcome.ts draws with half-block characters.

    python3 scripts/gen-terminal-logo.py [pixel size in logo units, default 3.0]

Prints rows of '#' (solid), 'o' (faded) and '.' (empty); two rows make one terminal line."""
import re, sys, math, os
svg = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "clutor-logo.svg")).read()
polys = []
for m in re.finditer(r'<polygon points="([^"]+)"([^>]*)/>', svg):
    pts = [tuple(map(float, p.split(","))) for p in m.group(1).split()]
    polys.append((pts, "opacity" in m.group(2)))
def inside(pt, poly):
    x, y = pt; n = len(poly); c = False
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1: c = not c
    return c
def dist_edge(pt, poly):
    best = 1e9; x, y = pt
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = x2 - x1, y2 - y1
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best
GROW = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
px = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
xs = [p[0] for poly, _ in polys for p in poly]; ys = [p[1] for poly, _ in polys for p in poly]
x0, x1, y0, y1 = min(xs) - GROW, max(xs) + GROW, min(ys) - GROW, max(ys) + GROW
W, H = math.ceil((x1 - x0) / px), math.ceil((y1 - y0) / px)
H += H % 2
rows = []
for r in range(H):
    row = ""
    for c in range(W):
        pt = (x0 + (c + .5) * px, y0 + (r + .5) * px); ch = "."
        for poly, faded in polys:   # solid hexagons are drawn over the faded one
            if inside(pt, poly) or dist_edge(pt, poly) <= GROW:
                if not faded: ch = "#"
                elif ch == ".": ch = "o"
        row += ch
    rows.append(row)
print("\n".join(rows))
