#!/usr/bin/env python3
"""
ladder — the compute ladder: on a packed store, READING facts is a free look-up; COMPUTING a joint
function over them is not.

Pack 8 facts (bits) onto shared directions, then ask a LINEAR reader (a look-up — one matmul) to do
three jobs:
    READ   each fact back out         -> free  (every fact is linearly readable from the pack)
    COUNT  how many are on            -> free  (a sum is a linear aggregate)
    MAJORITY are most of them on      -> free  (a threshold on that sum — still linear)
    PARITY  is an odd number on (XOR) -> WALLS  (a look-up can't combine them; not linearly separable)
Then: the SAME parity on the RAW (unpacked) bits also walls -> the cost is the OPERATION, not the packing.
And: add one hidden layer (a little real computation) and parity becomes solvable -> the bits were always
there; a look-up just can't do the joint part.

A CPU/numpy toy in the spirit of fleet E17_compute_ladder. It is a LEARNABILITY statement about a bounded
linear reader -- NOT a claim about a real model (a deep model has the depth; the point is that the READ
is a look-up, and look-ups don't do joint computation).
"""
import numpy as np

N = 8            # facts (bits) per example
D = 8            # store width (8 shared directions)
M = 5000         # examples
def log(*a): print(*a, flush=True)


def main():
    rng = np.random.default_rng(0)
    atoms = rng.standard_normal((D, N)); atoms /= np.linalg.norm(atoms, axis=0)      # 8 hand-placed directions
    bits = (rng.random((M, N)) < 0.5).astype(float)                                  # random fact patterns
    code = bits @ atoms.T                                                            # PACK: superpose the active facts
    ntr = 3500; tr = slice(0, ntr); te = slice(ntr, M)

    from sklearn.linear_model import LogisticRegression

    def regress(X, y):                                 # a LINEAR reader (look-up) for a number: least squares
        W, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None); return X @ W

    def classify(X, y):                                # a LINEAR reader (look-up) for a yes/no: logistic
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr].ravel()); return float(np.mean(clf.predict(X[te]) == y[te].ravel()))

    s = bits.sum(1, keepdims=True)
    read_acc = float(np.mean((regress(code, bits)[te] > 0.5) == (bits[te] > 0.5)))     # recover every fact
    cnt_acc = float(np.mean(np.round(regress(code, s)[te]) == s[te]))                  # recover the count
    maj_acc = classify(code, (s >= N / 2).astype(int))                                 # most of them on? (linear threshold on the sum)
    par = (s % 2).astype(int)
    par_lin = classify(code, par)                                                      # odd number on? (XOR)
    par_raw = classify(bits, par)                                                      # unpacked control

    # one hidden layer (a little real computation) — does parity become solvable?
    try:
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=3000, random_state=0).fit(code[tr], par[tr].ravel())
        par_mlp = float(np.mean(clf.predict(code[te]) == par[te].ravel()))
    except Exception:
        par_mlp = float("nan")

    log("=== THE COMPUTE LADDER — pack 8 facts, ask a LINEAR reader (a look-up) four jobs ===")
    log(f"  (held-out accuracy; chance for the yes/no jobs = 0.50)")
    log(f"  READ      each fact back out          : {read_acc:.2f}   <- FREE: every fact is perfectly readable")
    log(f"  COUNT     how many are on             : {cnt_acc:.2f}   <- FREE: a sum is a linear aggregate")
    log(f"  MAJORITY  are most of them on?        : {maj_acc:.2f}   <- FREE: a threshold on that sum, still linear")
    log(f"  PARITY    odd number on? (XOR)        : {par_lin:.2f}   <- WALLS at chance: a look-up can't combine them")
    log(f"  ...same PARITY on the RAW bits        : {par_raw:.2f}   <- walls UNPACKED too -> the OPERATION, not the packing")
    log(f"  PARITY with one hidden layer (depth)  : {par_mlp:.2f}   <- add real computation -> solvable; the bits were always there")
    log("")
    log("  -> READ / COUNT / MAJORITY ride the pack for FREE (look-ups). PARITY does NOT -- combining the")
    log("     facts is a real computation, not a look-up. Look-up free; joint computation costs. And the")
    log("     wall is the OPERATION (raw bits wall too), not the packing.")
    import json
    json.dump(dict(N=N, D=D, read=round(read_acc, 3), count=round(cnt_acc, 3), majority=round(maj_acc, 3),
                   parity_linear=round(par_lin, 3), parity_raw=round(par_raw, 3), parity_depth=round(par_mlp, 3)),
              open("ladder.json", "w"), indent=1)
    log("wrote ladder.json")


if __name__ == "__main__":
    main()
