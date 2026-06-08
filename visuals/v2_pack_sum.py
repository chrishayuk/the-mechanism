#!/usr/bin/env python3
"""
v2 — three hand-placed vectors summing into one channel  (MUST-BUILD, Part 1 B-roll).

The numbers visual that sits under "to pack three facts into one, I add their coordinates,
column by column." The three real `pack` rows appear; the 4th column lights up and its three
values (-0.48, -0.59, -0.75) add down to -1.82; then the whole packed code row is revealed.
Same numbers as the `pack` terminal.
"""
import numpy as np
import vstyle as v

ROWS = [("capital", v.ATOMS[0], v.CAPITAL), ("currency", v.ATOMS[1], v.CURRENCY),
        ("language", v.ATOMS[2], v.LANGUAGE)]
COL_X = [4.6 + 1.62 * j for j in range(6)]            # x-centre of each of the 6 number columns
LABEL_X = 2.0
Y = {"capital": 6.7, "currency": 5.8, "language": 4.9, "rule": 4.35, "code": 3.7}
HCOL = 3                                              # the 4th column (index 3) is the called-out one


def num(ax, x, y, val, col, size=22, weight="bold", alpha=1.0):
    ax.text(x, y, f"{val:+.2f}", color=col, fontsize=size, weight=weight, ha="center", va="center",
            family="monospace", alpha=alpha)


def head(ax, stage):
    titles = ["PACK = add the coordinates, column by column",
              "PACK = add the coordinates, column by column",
              "watch ONE column: add it down",
              "watch ONE column: add it down",
              "three facts → one vector"]
    ax.text(0.7, 8.4, titles[stage], color=v.FG, fontsize=23, weight="bold")
    ax.text(0.7, 0.5, "storage = addition · 8 relations, 6 slots: the spots OVERLAP", color=v.DIM, fontsize=13)


def frame(stage):
    fig, ax = v.canvas()
    head(ax, stage)
    if stage >= 1:                                   # spotlight the 4th column
        x = COL_X[HCOL]
        ax.add_patch(plt_rect(x - 0.78, Y["code"] - 0.5, 1.56, Y["capital"] - Y["code"] + 1.0,
                              v.HILITE, 0.10))
    for name, vec, col in ROWS:
        ax.text(LABEL_X, Y[name], name, color=col, fontsize=20, weight="bold", ha="left", va="center")
        for j, val in enumerate(vec):
            hot = (stage >= 1 and j == HCOL)
            num(ax, COL_X[j], Y[name], val, v.HILITE if hot else col)
    ax.plot([LABEL_X, COL_X[-1] + 0.9], [Y["rule"], Y["rule"]], color=v.DIM, lw=1.2)
    if stage == 2:                                   # cue: this column adds DOWN into the code row
        ax.annotate("", xy=(COL_X[HCOL] - 0.95, Y["code"] + 0.05), xytext=(COL_X[HCOL] - 0.95, Y["capital"] - 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=v.HILITE, lw=2.0, alpha=0.8))
        ax.text(COL_X[HCOL], 2.9, "add the column down", color=v.HILITE, fontsize=15, ha="center")
    if stage >= 2:                                   # the packed code row (col 4 first, then the rest)
        ax.text(LABEL_X, Y["code"], "code", color=v.SUM, fontsize=20, weight="bold", ha="left", va="center")
        shown = range(6) if stage >= 3 else [HCOL]
        for j in shown:
            hot = (stage <= 3 and j == HCOL)
            num(ax, COL_X[j], Y["code"], v.CODE[j], v.HILITE if hot else v.SUM)
    if stage >= 4:
        ax.add_patch(plt_rect(LABEL_X - 0.2, Y["code"] - 0.45, COL_X[-1] - LABEL_X + 1.3, 0.9, v.SUM, 0.10))
        ax.text(COL_X[-1] + 1.4, Y["code"], "1 vector", color=v.SUM, fontsize=16, weight="bold", va="center")
    return fig


def plt_rect(x, y, w, h, color, alpha):
    import matplotlib.patches as mp
    return mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=0, facecolor=color, alpha=alpha, zorder=0)


def main():
    paths = [v.save(frame(s), f"v2_packsum_{s}.png") for s in range(5)]
    v.frames_to_mp4(paths, "v2_pack_sum.mp4", fps=1.3)


if __name__ == "__main__":
    main()
