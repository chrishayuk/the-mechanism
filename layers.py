#!/usr/bin/env python3
"""
layerlens — watch the address RESOLVE across layers (the build-up made visible).

Logit-lens: read out the residual at each layer (apply the model's own final norm + unembed, with the
downstream layers made identity) and see the top prediction evolve. The "capital isn't the famous city"
cases are the tell: ask "the capital of Australia is" and early layers lean SYDNEY (the salient city);
only as the representation BUILDS does CANBERRA (the correct capital) take over. Shows (a) the value
materialises late, (b) the model builds/resolves the answer across layers, (c) fuzzy(early)->resolved(late).

Exploratory: does the Sydney->Canberra crossover happen cleanly, and at which layer? Self-contained.
"""
import sys, json
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx
from chuk_lazarus.models_v2.loader import load_model, ModelDType

# (prompt, correct capital, salient-but-wrong city)
CASES = [
    ("The capital of Australia is", "Canberra", "Sydney"),
    ("The capital of the United States is", "Washington", "New York"),
    ("The capital of Canada is", "Ottawa", "Toronto"),
    ("The capital of Turkey is", "Ankara", "Istanbul"),
    ("The capital of Brazil is", "Brasilia", "Rio"),
]
STATE = {"upto": None}
def log(*a): print(*a, flush=True)


def main():
    log("loading google/gemma-3-4b-it (bf16) ...")
    lm = load_model("google/gemma-3-4b-it", dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer; model.freeze()
    nlayers = len(model.model.layers); bos = getattr(tok, "bos_token_id", -1)
    idx = {id(model.model.layers[i]): i for i in range(nlayers)}
    blk = type(model.model.layers[0]); orig = blk.__call__
    def patched(self, x, mask=None, cache=None):
        res = orig(self, x, mask=mask, cache=cache)
        i = idx.get(id(self), -1)
        if STATE["upto"] is not None and i > STATE["upto"]:        # logit-lens: layers past `upto` = identity
            return type(res)(hidden_states=x, cache=res.cache)
        return res
    blk.__call__ = patched

    def tid_of(v):
        ids = [i for i in tok.encode(" " + v) if i != bos]; return ids[0] if len(ids) else None
    def lens(prompt, upto):
        STATE["upto"] = upto
        ll = np.array(model(mx.array([tok.encode(prompt)])).logits[0, -1].astype(mx.float32))
        STATE["upto"] = None; p = np.exp(ll - ll.max()); p /= p.sum()
        return int(np.argmax(ll)), p

    LAYERS = [0, 8, 12, 16, 20, 22, 24, 26, 28, 30, 32, nlayers - 1]
    out = {}
    for (prompt, correct, salient) in CASES:
        tc, ts = tid_of(correct), tid_of(salient)
        log(f"\n=== {prompt!r}   (correct {correct!r}, salient {salient!r}) ===")
        log(f"{'L':>4} | {'top-1 token':>14} | P({correct[:8]}) | P({salient[:8]}) | crossover?")
        rows = {}; crossed = None
        for L in LAYERS:
            top, p = lens(prompt, L)
            pc = float(p[tc]) if tc else 0.0; pss = float(p[ts]) if ts else 0.0
            mark = ""
            if crossed is None and pc > pss and pc > 0.02:
                crossed = L; mark = "  <- CORRECT overtakes salient here"
            rows[L] = dict(top=tok.decode([top]).strip()[:14], p_correct=round(pc, 3), p_salient=round(pss, 3))
            log(f"{L:>4} | {rows[L]['top']:>14} | {pc:>9.3f} | {pss:>9.3f} |{mark}")
        out[prompt] = dict(correct=correct, salient=salient, crossover_layer=crossed, by_layer=rows)
        log(f"  -> {correct} overtakes {salient} at layer {crossed}" if crossed is not None
            else f"  -> {correct} never cleanly overtakes {salient} in the swept layers")
    json.dump(out, open("layers.json", "w"), indent=1, default=float)
    log("\nwrote layers.json")


if __name__ == "__main__":
    main()
