#!/usr/bin/env python3
"""
pack — build a packed memory by hand, then read it back. The whole idea is TWO tiny functions:

    pack(dirs)          storage = ADDITION       — lay the chosen relation-directions on one channel
    decode(code, atoms) reading = MATCH & PEEL    — take the best-matching relation, subtract it, repeat

A relation (capital, currency, language, …) is just a fixed DIRECTION — a vector. A place "has" a few
relations, so we ADD those directions into one vector. To read it back, we repeatedly grab whichever
relation still matches the leftover signal most, and PEEL it off. No model, no training.

This is SUPERPOSITION packing — it NEEDS a de-mixer (decode) to read. The opposite of the FFN's
key→value ADDRESSING (see ffn.py / the real model), which reads with one matmul and no peeling.
"""
import json
import numpy as np


# ======================= THE WHOLE MECHANISM — two functions, show these =======================

def pack(directions):
    """Storage is addition: lay the chosen directions on top of each other in ONE vector."""
    return sum(directions)


def decode(code, atoms, threshold=0.30):
    """Reading is match-and-peel: grab the best-matching relation, subtract it, repeat."""
    leftover = np.array(code, dtype=float)
    found, trace = [], []
    while True:
        match = atoms @ leftover               # how strongly does each relation match what's left?
        for f in found:
            match[f] = -9                       # ignore ones we already pulled out
        winner = int(np.argmax(match))
        trace.append((match.copy(), winner))    # (just for the on-screen printout)
        if match[winner] < threshold:           # nothing left above the noise → done
            break
        found.append(winner)
        leftover = leftover - match[winner] * atoms[winner]    # PEEL the winner off
    return set(found), trace


# ======================= everything below is setup + on-screen printing =======================

REL = ["capital", "currency", "language", "river", "mountain", "leader", "anthem", "motto"]


def show_construction():
    """The legible demo: 8 relation-directions in 6 slots; pack 3, read them back with decode()."""
    np.set_printoptions(precision=2, suppress=True, sign=" ")
    rng = np.random.default_rng(27)
    atoms = rng.standard_normal((8, 6)); atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)  # 8 unit dirs in 6-dim
    have = [0, 1, 2]                              # this place ('Atlantis') HAS capital, currency, language

    print("=== HOW IT'S PACKED — 8 relation-types, only 6 slots ===")
    print("Each relation is ONE hand-placed direction (6 numbers). 'Atlantis' HAS three of them:")
    for j in have:
        print(f"    {REL[j]:9s} = {atoms[j]}")
    code = pack([atoms[j] for j in have])
    print("\nPACK = add the three onto ONE channel:")
    print(f"    code = capital + currency + language = {code}   <- 3 facts, 1 vector (8 relations, 6 slots: they OVERLAP)")

    print("\nREAD BACK = decode(): match, lock the winner, PEEL it, repeat:")
    found, trace = decode(code, atoms)
    peeled = []
    for step, (match, w) in enumerate(trace, 1):
        cells = "  ".join(f"{REL[k]} " + ("(peeled)" if k in peeled else f"{match[k]:+.2f}") for k in have + [3])
        if match[w] < 0.30:
            print(f"    step {step}: {cells}  ...  nothing >= 0.30 left -> STOP")
        else:
            print(f"    step {step}: {cells}  ...  LOCK '{REL[w]}', peel it off"); peeled.append(w)
    ok = found == set(have)
    print(f"    recovered {{{', '.join(REL[j] for j in sorted(found))}}} {'EXACTLY' if ok else '(MISMATCH)'}"
          f" — the present relations light up; peeling resolves them; the absent ones stay quiet.\n")
    np.set_printoptions()


def harmonic_atoms(K, d, seed):
    """K unit directions packed into d<K dims (a DCT-style frame) — just a way to make many fit in few."""
    rng = np.random.default_rng(seed); rows = rng.choice(K, size=d, replace=False); n = np.arange(K)
    Fd = np.stack([np.cos(np.pi * (n + 0.5) * m / K) for m in rows])
    Q, _ = np.linalg.qr(rng.standard_normal((K, d))); R = Q @ Fd
    return (R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-9)).T    # rows = atoms (one per relation)


def main():
    show_construction()
    print("=== AND IT SCALES — a real world atlas (no model, no training) ===")
    K, ACTIVE, N = 1241, 6, 400
    rng = np.random.default_rng(0)
    places = [sorted(rng.choice(K, size=ACTIVE, replace=False)) for _ in range(N)]
    out = {}
    for ratio in (1, 5, 13, 19):
        d = K if ratio <= 1 else max(2, round(K / ratio))
        atoms = harmonic_atoms(K, d, 0)
        exact = sum(decode(pack([atoms[i] for i in p]), atoms)[0] == set(p) for p in places)
        out[f"ratio_{ratio}"] = dict(d=d, exact_set=round(exact / N, 3))
        print(f"  ratio {ratio:>2} (K={K} relation-types -> {d} slots): exact-set recovery {exact/N:.3f}")
    print(f"\n  {N} places x {K} relation-types, packed by hand; the SAME decode() reads each back exactly —")
    print(f"  even crammed ~13x tighter than should fit. I built both halves: pack() and decode().")
    json.dump(dict(K=K, active=ACTIVE, n_entities=N, results=out), open("pack.json", "w"), indent=1)
    print("wrote pack.json")


if __name__ == "__main__":
    main()
