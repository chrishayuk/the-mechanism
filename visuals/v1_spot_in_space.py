#!/usr/bin/env python3
"""
v1 — Part-1 "spot in space" establishing graphic  (MUST-BUILD).

The teaching frame: a fact is a SPOT; six numbers are its coordinates; a "direction" is
that spot. Build it up — one labelled point, then three (capital/currency/language), then
their vector sum = the packed channel. The 6D directions are projected to 2D with their own
top-2 SVD plane, so the picture is an honest shadow of the real `pack` vectors and the
tip-to-tail sum lands exactly on the projected code.
"""
import numpy as np
import vstyle as v

# project the real 6D directions into their best 2D plane (origin = the zero vector, no centring)
M = v.ATOMS[v.HAVE]                                   # capital, currency, language (3x6)
_, _, Vt = np.linalg.svd(M, full_matrices=False)
W = Vt[:2].T                                          # 6x2 orthonormal plane
P = M @ W                                             # 2D points for the three directions
Csum = v.CODE @ W                                     # 2D point for the packed code (= P.sum(0))

# auto-fit everything we'll draw (the spots + the tip-to-tail path) into the left 2/3 of frame
PATH = np.array([[0, 0], P[0], P[0] + P[1], Csum])
PTS = np.vstack([PATH, P])
mn, mx = PTS.min(0), PTS.max(0); ctr = (mn + mx) / 2
RX0, RX1, RY0, RY1 = 1.6, 10.6, 1.5, 7.5
SC = min((RX1 - RX0) / (mx[0] - mn[0] + 1e-9), (RY1 - RY0) / (mx[1] - mn[1] + 1e-9))
OX, OY = (RX0 + RX1) / 2, (RY0 + RY1) / 2
def xy(p): return (OX + SC * (p[0] - ctr[0]), OY + SC * (p[1] - ctr[1]))
LABELS = ["capital", "currency", "language"]
COLS = [v.CAPITAL, v.CURRENCY, v.LANGUAGE]
o = xy([0, 0])


def base(ax, title, sub):
    ax.plot([o[0] - 5.0, o[0] + 5.0], [o[1], o[1]], color=v.DIM, lw=0.8, alpha=0.30)
    ax.plot([o[0], o[0]], [o[1] - 3.6, o[1] + 3.6], color=v.DIM, lw=0.8, alpha=0.30)
    ax.plot(*o, "o", color=v.DIM, ms=4)
    ax.text(0.6, 8.4, title, color=v.FG, fontsize=23, weight="bold")
    ax.text(12.0, 4.9, sub, color=v.DIM, fontsize=15, va="center")
    ax.text(0.6, 0.5, "a fact is a spot · six numbers are its coordinates · a direction is that spot",
            color=v.DIM, fontsize=13)


def dot(ax, p, label, col, arrow=False, alpha=1.0):
    x, y = xy(p)
    if arrow:
        ax.annotate("", xy=(x, y), xytext=o, zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4, alpha=alpha, shrinkA=0, shrinkB=0))
    ax.plot(x, y, "o", color=col, ms=13, mec=v.BG, mew=1.5, zorder=5, alpha=alpha)
    dy = 0.5 if y >= o[1] else -0.78
    ax.text(x + 0.15, y + dy, label, color=col, fontsize=17, weight="bold", zorder=6, alpha=alpha)


def frame(stage):
    fig, ax = v.canvas()
    titles = ["one fact = one spot", "one fact = one spot", "three relations, three spots",
              "each spot is a direction", "add the directions → one packed spot"]
    subs = ["", "capital\nis one spot", "capital\ncurrency\nlanguage", "each is a\ndirection\nfrom zero",
            "their sum is\nONE spot —\nthe packed\nchannel"]
    base(ax, titles[stage], subs[stage])
    if stage == 4:                                           # tip-to-tail addition reads cleanest alone
        for i in range(3): dot(ax, P[i], LABELS[i], COLS[i], arrow=True, alpha=0.18)
        tail = np.array([0.0, 0.0])
        for i in range(3):
            a = xy(tail); tail = tail + P[i]; b = xy(tail)
            ax.annotate("", xy=b, xytext=a, zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=COLS[i], lw=2.6, shrinkA=0, shrinkB=0))
        cx, cy = xy(Csum)
        ax.plot(cx, cy, "*", color=v.SUM, ms=30, mec=v.BG, mew=1.5, zorder=7)
        ax.text(cx, cy + 0.75, "packed code", color=v.SUM, fontsize=16, weight="bold", ha="center", zorder=8)
        ax.text(8.0, 1.15, "[-0.17  0.81  0.37  -1.82  0.18  -0.53]", color=v.SUM, fontsize=15,
                weight="bold", ha="center", zorder=8)
        return fig
    if stage >= 1: dot(ax, P[0], LABELS[0], COLS[0], arrow=(stage >= 3))
    if stage >= 2:
        for i in (1, 2): dot(ax, P[i], LABELS[i], COLS[i], arrow=(stage >= 3))
    return fig


def main():
    paths = [v.save(frame(s), f"v1_spot_{s}.png") for s in range(5)]
    v.frames_to_mp4(paths, "v1_spot_in_space.mp4", fps=1.2)


if __name__ == "__main__":
    main()
