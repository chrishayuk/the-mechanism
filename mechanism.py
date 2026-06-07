#!/usr/bin/env python3
"""
mechanism — one entry point for the four captures of the mechanism film. Self-contained in
videos/the-mechanism/ (no dependency on fleet experiment dirs).

  python3 mechanism.py pack       Part 1   — pack by SUPERPOSITION (one channel) + manual OMP decoder (CPU)
  python3 mechanism.py wall       Part 2   — the model can't read the packed channel = chance (Gemma)
  python3 mechanism.py decode     Part 3.0a — the model PACKS too: one residual, 3 facts read out (polysemantic) (Gemma)
  python3 mechanism.py ladder     Part 3.0b — the compute ladder: read/count FREE, parity (joint compute) WALLS (CPU)
  python3 mechanism.py layers     Part 3.1 — watch the answer RESOLVE across layers (Sydney->Canberra @ L26) (Gemma)
  python3 mechanism.py trace      Part 3.2 — PROVE it: the value is WRITTEN by FFN lookups (absent early, ablate->collapse) (Gemma)
  python3 mechanism.py address    Part 3.3 — the address is the RELATION: probe generalises to synonyms (Gemma)
  python3 mechanism.py route      Part 3.4 — the ENTITY half: fuzzy, addressed by top-k candidates + rank (Gemma)
  python3 mechanism.py ffn        Part 4.0 — pack by ADDRESSING: a key->value FFN built by hand (the model's way) (CPU)
  python3 mechanism.py native     Part 4.1 — write at the address in real Gemma; reads natively (Gemma)
  python3 mechanism.py all        run all ten in order
  python3 wall_sweep.py           rigor    — the wall holds at every layer L4..L30 (run separately)
  python3 route_sweep.py          rigor    — the entity key builds across layers; L22 phrasing-trap (run separately)
"""
import sys, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CAPS = ["pack", "wall", "decode", "ladder", "layers", "trace", "address", "route", "ffn", "native"]


def run(name):
    print(f"\n{'='*70}\n# mechanism {name}\n{'='*70}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, f"{name}.py")], check=False)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        for c in CAPS: run(c)
    elif arg in CAPS:
        run(arg)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
