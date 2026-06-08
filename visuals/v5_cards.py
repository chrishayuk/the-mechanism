#!/usr/bin/env python3
"""
v5 — the text cards: title, closing, the "1 in 3" beat, and the Minsky–Papert XOR caption.
Plain monospace on black, matching the terminal captures. Pulled verbatim from SCRIPT.md.
"""
import vstyle as v


def card(name, lines, y0=5.6, gap=1.15, fontsize=34, color=None, weight="bold", sub=None):
    fig, ax = v.canvas()
    color = color or v.FG
    cols = color if isinstance(color, list) else [color] * len(lines)
    for i, ln in enumerate(lines):
        ax.text(8.0, y0 - i * gap, ln, color=cols[i], fontsize=fontsize, ha="center", va="center",
                weight=weight, family="monospace")
    if sub:
        ax.text(8.0, y0 - len(lines) * gap - 0.4, sub, color=v.DIM, fontsize=16, ha="center", va="center")
    v.save(fig, name)


def main():
    # cold-open title card
    card("card_title.png",
         ["The Mechanism.", "How a transformer reads", "its own knowledge."],
         y0=5.8, gap=1.25, fontsize=40,
         color=[v.FG, v.DIM, v.DIM])

    # Part 2 — the wall beat
    card("card_one_in_three.png",
         ["packed channel:", "1 in 3."],
         y0=5.3, gap=1.6, fontsize=54, color=[v.DIM, v.SUM],
         sub="one clean relation reads ~1.0 · the packed channel reads ~chance")

    # Part 3 — Minsky & Papert caption
    card("card_xor.png",
         ["a linear reader can't separate XOR", "— Minsky & Papert, 1969"],
         y0=5.2, gap=1.2, fontsize=30, color=[v.FG, v.DIM],
         sub="look-up rides the pack for free · joint computation does not")

    # closing card
    card("card_closing.png",
         ["The model packs its memory — but it never unpacks it. It addresses it.",
          "The address is the relation. The relation is an edge in a graph.",
          "Last video, you queried that graph. This one, you opened it up. Next, we take it out."],
         y0=5.6, gap=1.3, fontsize=21, color=[v.FG, v.GOOD, v.DIM])

    # thumbnail candidate
    card("card_thumb.png",
         ["It Doesn't Unpack", "Its Memory.", "It Addresses It."],
         y0=6.0, gap=1.5, fontsize=52, color=[v.FG, v.FG, v.SUM])


if __name__ == "__main__":
    main()
