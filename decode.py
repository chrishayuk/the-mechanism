#!/usr/bin/env python3
"""
decode — one residual vector is POLYSEMANTIC: three linear decoders pull three different facts out of it.

A "linear decoder" is the simplest reader there is — fit a straight line (a linear map) from a vector to
a label. We aim three of them at the SAME L26 residual (the model's state at the answer position):

    VALUE    — the model's OWN unembedding (already a linear map) reads the answer token off the residual
    RELATION — a linear probe we train:  capital / currency / language
    ENTITY   — a linear probe we train:  which place

All three succeed → that one vector holds many facts at once = superposition / packed. (E17: packing
doesn't HIDE facts; they stay linearly readable.) So the model PACKS to store — and ADDRESSES to read
(trace.py). What it never does is iteratively de-mix a packed channel by query (wall.py).
Ground truth = the model's own single-token answers. Self-contained, Gemma-3-4b via MLX.
"""
import sys, json
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx
from chuk_lazarus.models_v2.loader import load_model, ModelDType

L_READ = 26
REL = {"capital": "The capital of {e} is", "currency": "The currency of {e} is", "language": "The language of {e} is"}
COUNTRIES = ["France","Germany","Italy","Spain","Portugal","Greece","Austria","Belgium","Netherlands",
 "Denmark","Norway","Sweden","Finland","Ireland","Poland","Hungary","Romania","Croatia","Serbia","Russia",
 "Turkey","Japan","China","India","Pakistan","Thailand","Vietnam","Indonesia","Malaysia","Nepal","Brazil",
 "Argentina","Chile","Peru","Colombia","Bolivia","Mexico","Cuba","Canada","Australia","Egypt","Morocco",
 "Kenya","Nigeria","Ghana","Ethiopia","Senegal","Mali","Iran","Iraq","Israel","Jordan","Lebanon","Yemen",
 "Oman","Qatar","Armenia","Georgia","Estonia","Latvia","Slovakia","Slovenia","Malta","Cyprus","Albania"]
def log(*a): print(*a, flush=True)


def linear_probe(Xtr, ytr, Xte, yte, ks=(1,)):
    """The simplest reader: fit a linear map vector->label, report held-out top-k accuracy."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    order = np.argsort(-clf.predict_proba(Xte), axis=1); cls = clf.classes_
    return {k: float(np.mean([yte[i] in cls[order[i, :k]] for i in range(len(yte))])) for k in ks}


def main():
    log("loading google/gemma-3-4b-it (bf16) ...")
    lm = load_model("google/gemma-3-4b-it", dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer; model.freeze()
    nL = len(model.model.layers); lidx = {id(model.model.layers[i]): i for i in range(nL)}
    CAP = {}; STATE = {"upto": None}
    blk = type(model.model.layers[0]); orig = blk.__call__
    def patched(self, x, mask=None, cache=None):                  # capture the L26 residual; optionally read it out early
        res = orig(self, x, mask=mask, cache=cache); i = lidx[id(self)]
        if i == L_READ: CAP["res"] = np.array(res.hidden_states[0, -1, :].astype(mx.float32))
        if STATE["upto"] is not None and i > STATE["upto"]:
            return type(res)(hidden_states=x, cache=res.cache)
        return res
    blk.__call__ = patched
    def answer_and_residual(prompt):
        STATE["upto"] = None
        ll = np.array(model(mx.array([tok.encode(prompt)])).logits[0, -1].astype(mx.float32))
        return int(np.argmax(ll)), CAP["res"].copy()
    def value_readout(prompt):                                    # the model's UNEMBED applied AT L26 (a linear readout)
        STATE["upto"] = L_READ
        ll = np.array(model(mx.array([tok.encode(prompt)])).logits[0, -1].astype(mx.float32))
        STATE["upto"] = None; return int(np.argmax(ll))

    # capture: one L26 residual per (place, relation), the model's answer, and the labels
    log(f"capturing L{L_READ} residuals for {len(COUNTRIES)} places x {len(REL)} relations ...")
    rel_names = list(REL); X, y_rel, y_ent, y_val = [], [], [], []
    for ei, e in enumerate(COUNTRIES):
        for ri, r in enumerate(rel_names):
            ans, res = answer_and_residual(REL[r].format(e=e))
            X.append(res); y_rel.append(ri); y_ent.append(ei); y_val.append(ans)
    X = np.array(X); y_rel = np.array(y_rel); y_ent = np.array(y_ent)
    N = len(X); rng = np.random.default_rng(0); perm = rng.permutation(N); ntr = int(0.7 * N)
    tr, te = perm[:ntr], perm[ntr:]
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Z = (X - mu) / sd

    # DECODER 1 — VALUE: the model's own unembed reads the answer off the L26 residual
    sample = [(e, r) for e in COUNTRIES[:20] for r in rel_names]
    value_acc = np.mean([value_readout(REL[r].format(e=e)) == answer_and_residual(REL[r].format(e=e))[0] for e, r in sample])
    # DECODER 2 — RELATION, DECODER 3 — ENTITY: linear probes on the SAME residuals, held-out
    rel = linear_probe(Z[tr], y_rel[tr], Z[te], y_rel[te])
    ent = linear_probe(Z[tr], y_ent[tr], Z[te], y_ent[te], ks=(1, 5))

    log(f"\n=== ONE RESIDUAL VECTOR @ L{L_READ} — THREE FACTS READ OUT (held-out {len(te)}/{N}) ===")
    log(f"  (1) VALUE    — the model's own unembed reads the answer off the residual : {value_acc:.2f}")
    log(f"  (2) RELATION — a 3-way linear probe (capital / currency / language)      : {rel[1]:.2f}   (chance 0.33)")
    log(f"  (3) ENTITY   — a linear probe: which of {len(COUNTRIES)} places           : top5 {ent[5]:.2f}   (chance {1/len(COUNTRIES):.02f})")
    log(f"\n  three DIFFERENT facts, ONE vector, all LINEARLY readable -> the residual is POLYSEMANTIC (packed).")
    log(f"  the model superposes many facts per direction. It PACKS to store, and ADDRESSES to read (trace.py).")
    json.dump(dict(L=L_READ, N=N, held=len(te), value_readable=round(float(value_acc), 3),
                   relation_acc=round(rel[1], 3), entity_top1=round(ent[1], 3), entity_top5=round(ent[5], 3)),
              open("decode.json", "w"), indent=1, default=float)
    log("\nwrote decode.json")


if __name__ == "__main__":
    main()
