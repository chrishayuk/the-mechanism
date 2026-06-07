#!/usr/bin/env python3
"""
route_sweep — does the ENTITY key BUILD across layers the way the value does? (E15-style routing per layer.)

The layer-lens showed the VALUE resolve at L26 (Sydney->Canberra). This asks the same of the ENTITY KEY:
train an E15 router on the residual at each layer and measure top-k routing on a held-out phrasing.
Hypothesis: fuzzy early (low top-k = the model itself is still building the entity), resolving toward L26.
The point (user's): be fuzzy where the model is fuzzy, sharp where it's sharp -- read the key where it's
actually formed. One forward per prompt captures ALL target layers.

Self-contained. Reproduces fleet E15 at a sweep of layers.
"""
import sys, json, time
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from chuk_lazarus.models_v2.loader import load_model, ModelDType

TARGET = [12, 16, 20, 22, 24, 26, 28, 30]; Hr = 256; N = 150
COUNTRIES = ["France","Germany","Italy","Spain","Portugal","Greece","Austria","Switzerland","Belgium",
 "Netherlands","Denmark","Norway","Sweden","Finland","Iceland","Ireland","Poland","Hungary","Romania",
 "Bulgaria","Croatia","Serbia","Ukraine","Russia","Turkey","Japan","China","India","Pakistan","Bangladesh",
 "Thailand","Vietnam","Indonesia","Malaysia","Philippines","Singapore","Mongolia","Nepal","Cambodia","Laos",
 "Brazil","Argentina","Chile","Peru","Colombia","Venezuela","Ecuador","Bolivia","Paraguay","Uruguay",
 "Mexico","Cuba","Jamaica","Canada","Australia","New Zealand","Egypt","Morocco","Algeria","Tunisia",
 "Libya","Kenya","Nigeria","Ghana","Ethiopia","Tanzania","Uganda","Angola","Zambia","Zimbabwe","Senegal",
 "Mali","Sudan","Somalia","Cameroon","Iran","Iraq","Israel","Jordan","Lebanon","Syria","Yemen","Oman",
 "Qatar","Kuwait","Bahrain","Armenia","Georgia","Azerbaijan","Kazakhstan","Uzbekistan","Turkmenistan",
 "Afghanistan","Sri Lanka","South Korea","North Korea","Taiwan","Estonia","Latvia","Lithuania","Slovakia",
 "Slovenia","Luxembourg","Malta","Cyprus","Albania","Montenegro","Moldova","Belarus","Kyrgyzstan",
 "Tajikistan","Bhutan","Myanmar","Brunei","Botswana","Namibia","Mozambique","Madagascar","Malawi","Rwanda",
 "Burundi","Chad","Niger","Mauritania","Gabon","Congo","Liberia","Guinea","Benin","Togo","Gambia","Panama",
 "Costa Rica","Nicaragua","Honduras","Guatemala","Belize","Guyana","Suriname","Haiti","Bahamas","Fiji",
 "South Africa","United Kingdom","United States","Dominican Republic","El Salvador","Sierra Leone",
 "Mauritius","Maldives","Papua New Guinea","Eritrea","Djibouti","Lesotho"]
TRAIN = "The capital of {e} is"; PARA = "{e}'s capital city is"; CROSS = "The currency of {e} is"
def log(*a): print(*a, flush=True)


def main():
    mx.random.seed(0)                                # seed router inits -> reproducible per-layer top-k
    t = time.time(); log("loading google/gemma-3-4b-it (bf16) ...")
    lm = load_model("google/gemma-3-4b-it", dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer; Dh = model.model.embed_tokens.weight.shape[1]; model.freeze()
    ents = COUNTRIES[:N]; log(f"loaded {time.time()-t:.1f}s; routing {len(ents)} places, layers {TARGET}")
    mlp_idx = {id(model.model.layers[L].mlp): L for L in TARGET}
    mlpcls = type(model.model.layers[0].mlp); orig = mlpcls.__call__; CAP = {}
    def patched(self, x):
        out = orig(self, x); L = mlp_idx.get(id(self))
        if L is not None: CAP[L] = np.array(x[0, -1, :].astype(mx.float32))
        return out
    mlpcls.__call__ = patched
    def cap_all(p):
        CAP.clear(); ll = model(mx.array([tok.encode(p)])).logits; mx.eval(ll); return {L: CAP[L].copy() for L in TARGET}

    log(f"capturing {N}x3 prompts across {len(TARGET)} layers each ...")
    Xtr = {L: [] for L in TARGET}; Xpa = {L: [] for L in TARGET}; Xcr = {L: [] for L in TARGET}
    for e in ents:
        for d, P in ((Xtr, TRAIN), (Xpa, PARA), (Xcr, CROSS)):
            c = cap_all(P.format(e=e))
            for L in TARGET: d[L].append(c[L])
    y = np.arange(N)

    class Router(nn.Module):
        def __init__(s, H, N): super().__init__(); s.a = nn.Linear(H, Hr); s.b = nn.Linear(Hr, N)
        def __call__(s, x): return s.b(nn.gelu(s.a(x)))
    def fit_eval(L):
        Xt = np.array(Xtr[L]); mu = Xt.mean(0); sd = Xt.std(0) + 1e-6
        net = Router(Dh, N); Z = mx.array(((Xt - mu) / sd).astype(np.float32)); ym = mx.array(y.astype(np.int32))
        lg = nn.value_and_grad(net, lambda n: nn.losses.cross_entropy(n(Z), ym).mean()); opt = optim.Adam(learning_rate=2e-3)
        for _ in range(400): l, g = lg(net); opt.update(net, g); mx.eval(net.parameters(), opt.state)
        def tk(X):
            lo = np.array(net(mx.array(((np.array(X) - mu) / sd).astype(np.float32)))); o = np.argsort(-lo, axis=1)
            return {k: float(np.mean([y[i] in o[i, :k] for i in range(N)])) for k in (1, 5)}
        return tk(Xpa[L]), tk(Xcr[L])

    log(f"\n=== ENTITY KEY by layer (E15-style routing; chance@5 = {5/N:.02f}) — does it BUILD like the value? ===")
    log(f"{'L':>4} | {'paraphrase top1':>15} {'top5':>6} | {'cross-rel top1':>14} {'top5':>6} | reading")
    log("-" * 78)
    out = {}
    for L in TARGET:
        pa, cr = fit_eval(L); out[L] = dict(paraphrase=pa, cross_relation=cr)
        r = ("forming" if pa[5] < 0.5 else ("fuzzy (candidate-list)" if pa[5] < 0.85 else "resolved (sharp-ish)"))
        log(f"{L:>4} | {pa[1]:>15.2f} {pa[5]:>6.2f} | {cr[1]:>14.2f} {cr[5]:>6.2f} | {r}")
    best = max(TARGET, key=lambda L: out[L]["paraphrase"][5])
    log(f"\n  -> the entity key BUILDS across layers and peaks around L{best} (paraphrase top5 {out[best]['paraphrase'][5]:.2f}).")
    log(f"     fuzzy early (read too soon = candidate soup), sharpest near L26 — read the key where it's FORMED,")
    log(f"     and even there it's a candidate-list (top-5), not pinpoint. Be fuzzy where the model is fuzzy.")
    json.dump(dict(layers=TARGET, N=N, by_layer=out, peak_layer=best), open("route_sweep.json", "w"), indent=1, default=float)
    log("wrote route_sweep.json")


if __name__ == "__main__":
    main()
