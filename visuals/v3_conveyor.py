#!/usr/bin/env python3
"""
v3 — the address-not-unpack conveyor  (MUST-BUILD, Part 3).

The picture that replaces "unpacking": the model never de-mixes a packed channel at read
time. Upstream the question computes an ADDRESS (relation + entity); in the fact band an FFN
LOOKUP writes the value onto the belt already SEPARATED; the late layers just read it off and
commit. Contrast with the hand decoder, which does its work at READ time.
"""
import numpy as np
import matplotlib.patches as mp
import vstyle as v

BX0, BX1, BY, BH = 1.6, 14.4, 4.2, 0.9               # belt geometry
def xL(L): return BX0 + (BX1 - BX0) * L / 33.0        # layer -> x
FACT = (23, 28)


def box(ax, x, y, w, h, fc, ec, alpha=1.0, lw=1.6, z=5):
    ax.add_patch(mp.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 facecolor=fc, edgecolor=ec, alpha=alpha, lw=lw, zorder=z))


def belt(ax):
    # rollers + two belt lines
    for cx in (BX0, BX1):
        ax.add_patch(mp.Circle((cx, BY), BH / 2 + 0.12, facecolor="#1A1A22", edgecolor=v.DIM, lw=1.4, zorder=3))
    for yy in (BY - BH / 2, BY + BH / 2):
        ax.plot([BX0, BX1], [yy, yy], color=v.DIM, lw=1.6, zorder=2)
    # layer ticks
    for L in range(0, 34, 4):
        ax.plot([xL(L), xL(L)], [BY - BH / 2 - 0.12, BY - BH / 2 - 0.32], color=v.DIM, lw=1)
        ax.text(xL(L), BY - BH / 2 - 0.62, f"L{L}", color=v.DIM, fontsize=11, ha="center")
    # the fact band (shaded; the column header is drawn in frame() so it stacks with the lookup box)
    fb0, fb1 = xL(FACT[0]), xL(FACT[1])
    ax.add_patch(mp.Rectangle((fb0, BY - BH / 2 - 0.05), fb1 - fb0, BH + 0.1,
                 facecolor=v.HILITE, alpha=0.10, zorder=1))


def head(ax, title):
    ax.text(0.7, 8.4, title, color=v.FG, fontsize=23, weight="bold")
    ax.text(0.7, 0.5, "addressed UPSTREAM (when written) · the late layers just read it — never de-mixed at read time",
            color=v.DIM, fontsize=13)


def frame(stage):
    fig, ax = v.canvas()
    titles = ["the model builds an address — it does not unpack",
              "upstream: the question computes an ADDRESS",
              "the fact band: a lookup WRITES the value, already separated",
              "downstream: the late layers just READ it"]
    head(ax, titles[stage])
    belt(ax)
    # the question, upstream
    ax.text(xL(2), BY + 2.3, "“The capital of France is ___”", color=v.FG, fontsize=17, ha="left")
    ENT = "#C3A6FF"                                   # entity = its own colour (not currency-amber)
    if stage >= 1:                                    # address = relation + entity chips
        box(ax, xL(7), BY + 1.5, 2.3, 0.7, "#13202B", v.CAPITAL); ax.text(xL(7), BY + 1.5, "capital", color=v.CAPITAL, fontsize=15, ha="center", va="center", weight="bold", zorder=6)
        box(ax, xL(12), BY + 1.5, 2.3, 0.7, "#1C1630", ENT); ax.text(xL(12), BY + 1.5, "France", color=ENT, fontsize=15, ha="center", va="center", weight="bold", zorder=6)
        ax.text(xL(9.5), BY + 0.95, "address  =  relation  +  entity", color=v.DIM, fontsize=13, ha="center")
    if stage >= 2:                                    # FFN lookup stamps the value onto the belt
        lx = (xL(FACT[0]) + xL(FACT[1])) / 2
        ax.text(lx, BY + 2.55, "fact band L23–27", color=v.HILITE, fontsize=14, ha="center", weight="bold")
        box(ax, lx, BY + 1.85, 2.7, 0.78, "#241019", v.SUM); ax.text(lx, BY + 1.85, "FFN lookup", color=v.SUM, fontsize=14, ha="center", va="center", weight="bold", zorder=6)
        ax.annotate("", xy=(lx, BY + 0.5), xytext=(lx, BY + 1.42),
                    arrowprops=dict(arrowstyle="-|>", color=v.SUM, lw=2.2))
        tx = lx if stage == 2 else xL(31)             # the value box rides the belt to the read head
        box(ax, tx, BY, 1.7, 0.66, "#3A1426", v.SUM, z=6); ax.text(tx, BY, "Paris", color="#FFC2DA", fontsize=16, ha="center", va="center", weight="bold", zorder=7)
    if stage >= 3:                                    # read off + commit
        ax.annotate("", xy=(xL(31), BY + 1.4), xytext=(xL(31), BY + 0.4),
                    arrowprops=dict(arrowstyle="-|>", color=v.GOOD, lw=2.2))
        ax.text(xL(31), BY + 1.75, "answer:\nParis", color=v.GOOD, fontsize=15, ha="center", va="center", weight="bold")
    return fig


def main():
    paths = [v.save(frame(s), f"v3_conveyor_{s}.png") for s in range(4)]
    v.frames_to_mp4(paths, "v3_conveyor.mp4", fps=1.2)


if __name__ == "__main__":
    main()
