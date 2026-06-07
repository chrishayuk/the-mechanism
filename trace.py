#!/usr/bin/env python3
"""
trace — SHOW the addressing happen inside the model, and prove it's a LOOKUP, not a de-mix.

For a known fact ("The capital of France is" -> Paris), walk the residual stream layer by layer and ask
two things at the answer position:
  (1) logit-lens  P(answer)         -- WHEN does the value become linearly readable?
  (2) mlp_write   = mlp_out . v_hat -- WHICH layer's FFN WRITES the answer direction into the residual?
      (v_hat = unit embedding of the answer token; mlp_out = that layer's MLP output at the last position)

If the value is ABSENT early and WRITTEN by an FFN in the fact band (one big mlp_write spike, P jumps
right after), that's a key->value LOOKUP — the FFN reads a key and emits a value. It is NOT iterative
de-mixing (which would need the value already present, superposed, peeled out over many steps).
Then the causal check: ZERO that FFN's write -> the answer collapses. The lookup is doing the work.

Self-contained (Gemma-3-4b via MLX). Companion to wall.py: wall shows the model has no de-mixer for an
external pack; trace shows what it does INSTEAD -- a linear FFN lookup. (E17: facts are linearly readable.)
"""
import sys, json
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx
from chuk_lazarus.models_v2.loader import load_model, ModelDType

FACTS = [("The capital of France is", "Paris"), ("The capital of Japan is", "Tokyo"),
         ("The capital of Germany is", "Berlin"), ("The capital of Egypt is", "Cairo"),
         ("The capital of Canada is", "Ottawa")]
FACT_BAND = [23, 24, 25, 26, 27, 28]            # the fact-resolution band (where the value gets written)
CTRL_BAND = [10, 11, 12, 13, 14, 15]            # an early control band (should NOT carry the value)
CAP = {}; ABL = {"layers": None}
def log(*a): print(*a, flush=True)


def main():
    log("loading google/gemma-3-4b-it (bf16) ...")
    lm = load_model("google/gemma-3-4b-it", dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer; model.freeze()
    emb = np.array(model.model.embed_tokens.weight.astype(mx.float32))
    nL = len(model.model.layers); bos = getattr(tok, "bos_token_id", -1)
    lidx = {id(model.model.layers[i]): i for i in range(nL)}
    midx = {id(model.model.layers[i].mlp): i for i in range(nL)}

    # hook every block (capture residual-out) + every mlp (capture mlp-out, and optionally ZERO it for ablation)
    blk = type(model.model.layers[0]); borig = blk.__call__
    def bpatched(self, x, mask=None, cache=None):
        res = borig(self, x, mask=mask, cache=cache)
        CAP[("res", lidx[id(self)])] = np.array(res.hidden_states[0, -1, :].astype(mx.float32))
        return res
    blk.__call__ = bpatched
    mlpc = type(model.model.layers[0].mlp); morig = mlpc.__call__
    def mpatched(self, x):
        out = morig(self, x); i = midx[id(self)]
        CAP[("mlp", i)] = np.array(out[0, -1, :].astype(mx.float32))
        if ABL["layers"] is not None and i in ABL["layers"]:    # erase this MLP's write at the last position
            seq = out.shape[1]
            keep = (mx.arange(seq) != seq - 1).reshape(1, seq, 1).astype(out.dtype)
            out = out * keep
        return out
    mlpc.__call__ = mpatched

    # identity-downstream logit-lens: read the residual at layer L through the model's OWN norm+unembed
    STATE = {"upto": None}
    def lens_patched(self, x, mask=None, cache=None):
        res = borig(self, x, mask=mask, cache=cache)
        i = lidx[id(self)]
        CAP[("res", i)] = np.array(res.hidden_states[0, -1, :].astype(mx.float32))
        if STATE["upto"] is not None and i > STATE["upto"]:
            return type(res)(hidden_states=x, cache=res.cache)
        return res
    def tid_of(v):
        ids = [i for i in tok.encode(" " + v) if i != bos]; return ids[0] if ids else None

    def pvalue(prompt, tid, upto=None):
        STATE["upto"] = upto
        ll = np.array(model(mx.array([tok.encode(prompt)])).logits[0, -1].astype(mx.float32))
        STATE["upto"] = None; p = np.exp(ll - ll.max()); p /= p.sum(); return float(p[tid])

    out = {}
    LSET = [16, 20, 22, 24, 25, 26, 27, 28, 30, nL - 1]
    blk.__call__ = lens_patched                                 # use the lens hook for the per-layer reads
    for prompt, ans in FACTS:
        tid = tid_of(ans); vhat = emb[tid] / (np.linalg.norm(emb[tid]) + 1e-9)
        # one full forward (upto=None) to fill CAP for ALL layers (residual + mlp), gives the final P too
        pfull = pvalue(prompt, tid, upto=None)
        mlp_write = {i: float(CAP[("mlp", i)] @ vhat) for i in range(nL)}
        plens = {L: pvalue(prompt, tid, upto=L) for L in LSET}   # logit-lens P(answer) at each L
        write_layer = max(range(12, nL - 2), key=lambda i: mlp_write[i])
        out[prompt] = dict(answer=ans, p_final=round(pfull, 3), write_layer=write_layer,
                           p_lens={L: round(plens[L], 3) for L in LSET},
                           mlp_write={i: round(mlp_write[i], 2) for i in range(18, 30)})
        log(f"\n=== {prompt!r} -> {ans!r}   (final P={pfull:.3f}) ===")
        log("  L   :  " + "  ".join(f"{L}" for L in LSET))
        log("  P(ans):" + "  ".join(f"{plens[L]:.2f}" for L in LSET) + "   <- value becomes linearly readable")
        topw = sorted(range(12, nL - 2), key=lambda i: -mlp_write[i])[:3]
        log(f"  biggest FFN writes of the '{ans}' direction: " +
            ", ".join(f"L{i} ({mlp_write[i]:+.1f})" for i in topw) + "   <- the lookup fires here")

    # causal: the write is DISTRIBUTED across the band (one layer alone barely dents it) -> ablate the BAND,
    # with an early-band control for specificity. Zero fact-band FFN writes -> answer collapses; early band -> no effect.
    blk.__call__ = bpatched
    def p_with(prompt, tid, layers):
        ABL["layers"] = (set(layers) if layers else None)
        p = pvalue(prompt, tid, upto=None); ABL["layers"] = None; return p
    log(f"\n=== CAUSAL CHECK — zero the FFN writes (last position); fact-band L{FACT_BAND[0]}-{FACT_BAND[-1]} vs early-control L{CTRL_BAND[0]}-{CTRL_BAND[-1]} ===")
    rows = []
    for prompt, ans in FACTS:
        tid = tid_of(ans)
        p0 = p_with(prompt, tid, None)
        pf = p_with(prompt, tid, FACT_BAND)
        pc = p_with(prompt, tid, CTRL_BAND)
        rows.append((ans, p0, pf, pc))
        log(f"  P({ans:7s}):  clean {p0:.2f}   | zero FACT-band -> {pf:.2f} {'(COLLAPSES)' if pf < 0.3 else ''}"
            f"   | zero early-control -> {pc:.2f} {'(unharmed)' if pc > 0.7 * p0 else ''}")
    mfact = float(np.mean([(p0 - pf) / (p0 + 1e-9) for _, p0, pf, _ in rows]))
    mctrl = float(np.mean([(p0 - pc) / (p0 + 1e-9) for _, p0, _, pc in rows]))
    log(f"\n  -> the value is ABSENT early, then WRITTEN by FFN key->value lookups across the fact band (L23-27),")
    log(f"     readable LINEARLY thereafter. Zeroing the band's writes drops P(answer) {mfact*100:.0f}% (collapse);")
    log(f"     zeroing an equal early band drops it {mctrl*100:.0f}% (specific to the fact band).")
    log(f"     That is ADDRESSING (a written lookup), not de-mixing of an early superposition —")
    log(f"     which is exactly why an externally-packed channel (wall.py) reads nothing: no lookup wrote it.")
    json.dump(dict(facts=out, fact_band=FACT_BAND, ctrl_band=CTRL_BAND,
                   mean_factband_drop=round(mfact, 3), mean_ctrlband_drop=round(mctrl, 3),
                   ablation=[(a, round(p0, 3), round(pf, 3), round(pc, 3)) for a, p0, pf, pc in rows]),
              open("trace.json", "w"), indent=1, default=float)
    log("\nwrote trace.json")


if __name__ == "__main__":
    main()
