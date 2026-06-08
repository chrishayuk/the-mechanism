#!/usr/bin/env python3
"""
vstyle — shared look + real numbers for the mechanism film's graphics.

Everything renders on-brand with the terminal captures: black background, monospace,
the SAME hand-placed numbers the `pack` capture prints. No new data is invented here —
the toy vectors are regenerated from pack.py's exact construction (rng seed 27), so the
graphics and the terminal show one set of numbers.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ----------------------------------------------------------------------------- palette
BG      = "#0B0B0F"   # near-black terminal background
FG      = "#EDEDED"   # off-white text
DIM     = "#6E6E78"   # muted labels / gridlines
CAPITAL = "#5CC8FF"   # cyan   — capital
CURRENCY= "#FFD166"   # amber  — currency
LANGUAGE= "#9BE564"   # green  — language
SUM     = "#FF6FA5"   # pink   — the packed code (the sum)
HILITE  = "#FFE08A"   # warm highlight (the column being added)
GOOD    = "#7CE38B"   # clean / relation index (green)
BAD     = "#FF5C5C"   # collision / wrong (red)
RELCOL  = {"capital": CAPITAL, "currency": CURRENCY, "language": LANGUAGE}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def _mono():
    """Pick a clean monospace that exists on this machine (prefer Menlo, fall back to bundled)."""
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Menlo", "DejaVu Sans Mono", "Andale Mono", "Courier New"):
        if name in have:
            return name
    return "monospace"


MONO = _mono()
plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": [MONO],
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.edgecolor": DIM,
    "axes.labelcolor": FG,
    "xtick.color": DIM,
    "ytick.color": DIM,
})


# --------------------------------------------------------------- the real toy numbers
# pack.py builds 8 relation directions in 6 slots with rng seed 27, unit-normalised rows.
# We regenerate the SAME ones so the graphics match the `pack` terminal exactly.
REL = ["capital", "currency", "language", "river", "mountain", "leader", "anthem", "motto"]


def toy_atoms():
    rng = np.random.default_rng(27)
    a = rng.standard_normal((8, 6))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


ATOMS = toy_atoms()
HAVE = [0, 1, 2]                       # 'Atlantis' has capital, currency, language
CODE = ATOMS[HAVE].sum(0)              # the packed channel = [-0.17 0.81 0.37 -1.82 0.18 -0.53]


def canvas(w=16, h=9, dpi=120):
    """A blank 16:9 black canvas (1920x1080 at dpi=120), no axes."""
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=fig.dpi)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")
    return path


def frames_to_mp4(frame_paths, name, fps=2):
    """Stitch PNG frames into an mp4 (ffmpeg) — falls back to a GIF if ffmpeg is missing."""
    import shutil, subprocess
    out = os.path.join(OUT, name)
    if shutil.which("ffmpeg"):
        # build a concat list so each still holds for 1/fps seconds
        listf = os.path.join(OUT, "_frames.txt")
        with open(listf, "w") as fh:
            for p in frame_paths:
                fh.write(f"file '{os.path.abspath(p)}'\nduration {1/fps:.3f}\n")
            fh.write(f"file '{os.path.abspath(frame_paths[-1])}'\n")   # hold last frame
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", listf, "-vf", "fps=30,format=yuv420p", out], check=True)
        os.remove(listf)
        print(f"  wrote {os.path.relpath(out)}")
        return out
    # fallback: GIF via pillow
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    gif = out.rsplit(".", 1)[0] + ".gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    print(f"  wrote {os.path.relpath(gif)} (ffmpeg not found)")
    return gif
