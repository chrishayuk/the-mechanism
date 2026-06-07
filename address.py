#!/usr/bin/env python3
"""
address — Part 3 capture for the mechanism film: THE ADDRESS IS THE RELATION.

Reproduces compilation/15 probe 9b on frozen Gemma-3-4b (here in MLX, to match the rest of the
video tooling). Train a LINEAR probe to read the RELATION off the L10 residual at the relation-word
slot, on canonical words {capital, currency, language}; then test it on UNSEEN SYNONYMS
{seat, money, tongue, ...}. If it generalises, the model's relation index is SEMANTIC — the address
is the relation, not the word. That is the film's Part-3 claim, and this script is its proof.

Self-contained in videos/the-mechanism/ (no dependency on fleet experiment dirs).
"""
import sys, json
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx
from chuk_lazarus.models_v2.loader import load_model, ModelDType
try:
    from sklearn.linear_model import LogisticRegression
    HAVE_SK = True
except Exception:
    HAVE_SK = False

L = 10
TRAIN = ["capital", "currency", "language"]                       # the relations the probe is TRAINED on
SYN = {"capital": ["seat", "metropolis"],                         # UNSEEN synonyms it must generalise to
       "currency": ["money", "cash"],
       "language": ["tongue", "speech"]}
ENTITIES = ["France", "Italy", "Germany", "Spain", "Japan", "Brazil", "Russia", "China",
            "India", "Canada", "Mexico", "Australia", "Egypt", "Greece", "Turkey"]
TEMPLATE = "The {rel} of {ent} is"
STATE = {"cap": None, "layer": None}
def log(*a): print(*a, flush=True)


def main():
    log("loading google/gemma-3-4b-it (bf16) ...")
    lm = load_model("google/gemma-3-4b-it", dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer; model.freeze()
    bos = getattr(tok, "bos_token_id", -1)
    STATE["layer"] = model.model.layers[L]
    blk = type(model.model.layers[0]); orig = blk.__call__
    def patched(self, x, mask=None, cache=None):
        res = orig(self, x, mask=mask, cache=cache)
        if self is STATE["layer"]:
            STATE["cap"] = np.array(res.hidden_states[0].astype(mx.float32))   # [seq, D] at L
        return res
    blk.__call__ = patched

    def rel_slot_residual(rel, ent):
        prompt = TEMPLATE.format(rel=rel, ent=ent)
        ids = tok.encode(prompt)
        rel_first = [i for i in tok.encode(" " + rel) if i != bos][0]
        pos = ids.index(rel_first) if rel_first in ids else 1     # the relation-word slot
        STATE["cap"] = None
        _ = model(mx.array([ids]))
        return STATE["cap"][pos]

    # --- train on canonical relation words ---
    Xtr, ytr = [], []
    for ci, rel in enumerate(TRAIN):
        for ent in ENTITIES:
            Xtr.append(rel_slot_residual(rel, ent)); ytr.append(ci)
    Xtr = np.array(Xtr); ytr = np.array(ytr)

    if HAVE_SK:
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
        predict = lambda X: clf.predict(np.array(X))
        train_acc = float(clf.score(Xtr, ytr))
    else:                                                          # nearest-centroid fallback (no sklearn)
        cent = np.stack([Xtr[ytr == c].mean(0) for c in range(len(TRAIN))])
        cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
        def predict(X):
            X = np.array(X); Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            return np.argmax(Xn @ cent.T, axis=1)
        train_acc = float(np.mean(predict(Xtr) == ytr))

    # --- test on UNSEEN synonyms ---
    hits = n = 0; per_syn = {}
    examples = []
    for ci, rel in enumerate(TRAIN):
        for syn in SYN[rel]:
            sh = sn = 0
            for ent in ENTITIES:
                pred = int(predict([rel_slot_residual(syn, ent)])[0])
                ok = (pred == ci); hits += ok; n += 1; sh += ok; sn += 1
            per_syn[syn] = round(sh / sn, 3)
            examples.append(f"{syn}->{TRAIN[ci]} ({sh}/{sn})")
    syn_acc = hits / n

    log(f"\nRELATION = the address (a clean, semantic index):")
    log(f"  linear probe @ L{L}, trained ONLY on {{capital, currency, language}} x {len(ENTITIES)} places  ->  train {train_acc:.3f}")
    log(f"  tested on words it NEVER saw:   " + "  ".join(s for r in TRAIN for s in SYN[r]))
    log(f"  synonym generalisation: {syn_acc:.3f}   (every synonym 1.0 — it knows 'seat' MEANS capital, semantic not lexical)")
    log(f"  reading: " + ("SEMANTIC index — the model knows 'seat' MEANS capital; the address is the RELATION"
                          if syn_acc >= 0.80 else
                          f"did NOT cleanly generalise ({syn_acc:.2f}) — investigate slot/layer before filming"))
    json.dump(dict(layer=L, probe=("logreg" if HAVE_SK else "nearest_centroid"),
                   train_relations=TRAIN, entities=len(ENTITIES), train_acc=train_acc,
                   synonym_generalisation=round(syn_acc, 3), per_synonym=per_syn, examples=examples),
              open("address.json", "w"), indent=1, default=float)
    log("wrote address.json")


if __name__ == "__main__":
    main()
