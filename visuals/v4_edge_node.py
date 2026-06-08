#!/usr/bin/env python3
"""
v4 — edge-vs-node diagram  (MUST-BUILD, Part 4).

A fact is an EDGE: entity, relation -> value (France --capital-of--> Paris). Then the honest
limit that lands where the mechanism says it should: the RELATION index is clean and roomy
(generalises), while the ENTITY slots are crowded — you can read hundreds the model already
learned, but only WRITE about a dozen new ones before they collide.
"""
import numpy as np
import matplotlib.patches as mp
import vstyle as v

ENT = "#C3A6FF"


def node(ax, x, y, label, col, r=0.62):
    ax.add_patch(mp.Circle((x, y), r, facecolor="#15151C", edgecolor=col, lw=2.2, zorder=5))
    ax.text(x, y, label, color=col, fontsize=16, ha="center", va="center", weight="bold", zorder=6)


def edge_row(ax):
    y = 7.0
    node(ax, 3.0, y, "France", ENT)
    node(ax, 8.6, y, "Paris", v.SUM)
    ax.annotate("", xy=(7.95, y), xytext=(3.65, y),
                arrowprops=dict(arrowstyle="-|>", color=v.CAPITAL, lw=2.4))
    ax.text(5.8, y + 0.45, "capital-of", color=v.CAPITAL, fontsize=15, ha="center", weight="bold")
    ax.text(5.8, y - 0.62, "the relation is the address", color=v.DIM, fontsize=12, ha="center")
    ax.text(11.6, y, "entity, relation → value", color=v.FG, fontsize=18, ha="center", va="center")


def relation_panel(ax):
    x0, y0, w, h = 1.4, 1.2, 5.6, 4.2
    ax.add_patch(mp.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                 facecolor="#0E140E", edgecolor=v.GOOD, lw=1.4, alpha=0.9))
    ax.text(x0 + w / 2, y0 + h - 0.45, "RELATION index", color=v.GOOD, fontsize=17, ha="center", weight="bold")
    items = ["capital", "currency", "language", "seat → capital", "money → currency", "tongue → language"]
    for i, it in enumerate(items):
        yy = y0 + h - 1.05 - i * 0.47
        ax.plot(x0 + 0.7, yy, "o", color=v.GOOD, ms=12, mec=v.BG, mew=1.2)
        faded = "→" in it
        ax.text(x0 + 1.15, yy, it, color=(v.GOOD if not faded else "#5FA86C"), fontsize=14, va="center")
    ax.text(x0 + w / 2, y0 + 0.28, "clean · semantic · generalises", color=v.GOOD, fontsize=13, ha="center")


def entity_panel(ax):
    x0, y0, w, h = 9.0, 1.2, 5.6, 4.2
    ax.add_patch(mp.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                 facecolor="#140E10", edgecolor=v.BAD, lw=1.4, alpha=0.9))
    ax.text(x0 + w / 2, y0 + h - 0.45, "ENTITY slots  (one relation)", color="#FF8C8C", fontsize=17, ha="center", weight="bold")
    # a dense grid: most green (learned -> readable), ~a dozen amber (writable), a few red (collisions)
    cols, rows = 16, 7
    gx0, gy0, dx, dy = x0 + 0.55, y0 + 0.95, (w - 1.1) / (cols - 1), 2.55 / (rows - 1)
    n = cols * rows
    READ, WRITE = n - 12 - 5, 12
    rng = np.random.default_rng(3)
    for k in range(n):
        r, c = divmod(k, cols)
        jx, jy = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
        x, y = gx0 + c * dx + jx, gy0 + r * dy + jy
        if k < READ:
            ax.plot(x, y, "o", color=v.GOOD, ms=8, alpha=0.85)
        elif k < READ + WRITE:
            ax.plot(x, y, "o", color=v.HILITE, ms=9, mec=v.BG, mew=0.8)
        else:                                            # collisions: two markers crammed together
            ax.plot(x - 0.06, y, "o", color=v.BAD, ms=9)
            ax.plot(x + 0.06, y, "o", color=v.BAD, ms=9, alpha=0.8)
    ax.text(x0 + w / 2, y0 + 0.32, "crowded — ~a dozen new before they collide", color="#FF8C8C", fontsize=13, ha="center")


def main():
    fig, ax = v.canvas()
    ax.text(0.7, 8.4, "a fact is an EDGE — and the honest limit", color=v.FG, fontsize=23, weight="bold")
    edge_row(ax)
    relation_panel(ax)
    entity_panel(ax)
    ax.text(8.0, 0.5, "READ hundreds   ·   WRITE a dozen", color=v.FG, fontsize=18, ha="center", weight="bold")
    # legend dots between the panels
    for (yy, col, lab) in [(3.3, v.GOOD, "read"), (2.7, v.HILITE, "write"), (2.1, v.BAD, "collide")]:
        ax.plot(7.85, yy, "o", color=col, ms=10); ax.text(8.0, yy, lab, color=v.DIM, fontsize=11, va="center")
    v.save(fig, "v4_edge_node.png")


if __name__ == "__main__":
    main()
