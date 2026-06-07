#!/usr/bin/env python3
"""
decode — one residual vector is POLYSEMANTIC: three linear decoders pull three different facts out of it.

A "linear decoder" is the simplest reader there is — a straight-line map from a vector to a label. We aim
three of them at the SAME L26 residual (the model's state at the answer position):

    value     = the model's OWN unembedding, read at L26 (the logit-lens)  -> the answer word
    relation  = LogisticRegression().fit(R, rels)                          -> capital / currency / language
    entity    = LogisticRegression().fit(R, places)                        -> which place

All three succeed -> that one vector holds many facts at once = superposition / packed. (E17: packing
doesn't HIDE facts; they stay linearly readable.) So the model PACKS to store -- and ADDRESSES to read
(trace.py). What it never does is iteratively de-mix a packed channel by query (wall.py).
Ground truth = the model's own single-token answers. Self-contained, Gemma-3-4b via MLX.
"""
import sys, json
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx
from sklearn.linear_model import LogisticRegression
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


def topk(probe, X, y, k):                                          # held-out top-k accuracy of a fitted probe
    order = np.argsort(-probe.predict_proba(X), axis=1)
    return float(np.mean([y[i] in probe.classes_[order[i, :k]] for i in range(len(y))]))


def main():
    log("loading google/gemma-3-4b-it (bf16) ...")
    lm = load_model("google/gemma-3-4b-it", dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer; model.freeze()
    nL = len(model.model.layers); lidx = {id(model.model.layers[i]): i for i in range(nL)}
    CAP = {}; STATE = {"upto": None}
    blk = type(model.model.layers[0]); orig = blk.__call__
    def patched(self, x, mask=None, cache=None):
        res = orig(self, x, mask=mask, cache=cache); i = lidx[id(self)]
        if i == L_READ: CAP["res"] = np.array(res.hidden_states[0, -1, :].astype(mx.float32))
        if STATE["upto"] is not None and i > STATE["upto"]:        # cut the stack at L26 to read it out early
            return type(res)(hidden_states=x, cache=res.cache)
        return res
    blk.__call__ = patched

    def run(prompt, upto=None):                                    # one forward; upto=L26 -> the logit-lens read at L26
        STATE["upto"] = upto
        ll = np.array(model(mx.array([tok.encode(prompt)])).logits[0, -1].astype(mx.float32))
        STATE["upto"] = None; return int(np.argmax(ll)), CAP["res"].copy()

    # capture: one L26 residual per (place, relation), with the relation + place labels
    log(f"capturing L{L_READ} residuals for {len(COUNTRIES)} places x {len(REL)} relations ...")
    rel_names = list(REL); res, rels, places = [], [], []
    for ei, e in enumerate(COUNTRIES):
        for ri, r in enumerate(rel_names):
            res.append(run(REL[r].format(e=e))[1]); rels.append(ri); places.append(ei)
    res = np.array(res); rels = np.array(rels); places = np.array(places)
    R = (res - res.mean(0)) / (res.std(0) + 1e-6)                  # standardise the residuals
    rng = np.random.default_rng(0); perm = rng.permutation(len(R)); ntr = int(0.7 * len(R))
    tr, te = perm[:ntr], perm[ntr:]

    # ===================== THREE LINEAR DECODERS, all aimed at the SAME L26 residual =====================
    # 1) VALUE    — the model's OWN unembedding, read at L26 (the logit-lens): does it already say the answer?
    sample = [REL[r].format(e=e) for e in COUNTRIES[:20] for r in rel_names]
    value = float(np.mean([run(p, upto=L_READ)[0] == run(p)[0] for p in sample]))
    # 2) RELATION — a linear probe we fit on the residuals; 3) ENTITY — same, with top-k
    relation = LogisticRegression(max_iter=2000).fit(R[tr], rels[tr])
    entity   = LogisticRegression(max_iter=2000).fit(R[tr], places[tr])
    rel_acc  = float(np.mean(relation.predict(R[te]) == rels[te]))
    ent_top5 = topk(entity, R[te], places[te], 5)
    # =====================================================================================================

    log(f"\n=== ONE RESIDUAL VECTOR @ L{L_READ} — THREE FACTS READ OUT (held-out {len(te)}/{len(R)}) ===")
    log(f"  (1) VALUE    — the model's own unembed reads the answer off the residual : {value:.2f}")
    log(f"  (2) RELATION — a 3-way linear probe (capital / currency / language)      : {rel_acc:.2f}   (chance 0.33)")
    log(f"  (3) ENTITY   — a linear probe: which of {len(COUNTRIES)} places           : top5 {ent_top5:.2f}   (chance {1/len(COUNTRIES):.02f})")
    log(f"\n  three DIFFERENT facts, ONE vector, all LINEARLY readable -> the residual is POLYSEMANTIC (packed).")
    log(f"  the model superposes many facts per direction. It PACKS to store, and ADDRESSES to read (trace.py).")
    json.dump(dict(L=L_READ, N=len(R), held=len(te), value_readable=round(value, 3),
                   relation_acc=round(rel_acc, 3), entity_top5=round(ent_top5, 3)),
              open("decode.json", "w"), indent=1, default=float)
    log("\nwrote decode.json")


if __name__ == "__main__":
    main()
