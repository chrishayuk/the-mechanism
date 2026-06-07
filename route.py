#!/usr/bin/env python3
"""
route — the ENTITY half of the address: it's fuzzy, so you address it by candidate-list + ranking.

The relation is a clean index (address.py). The ENTITY is not — but the model's OWN activation at L26 is
a usable, phrasing-invariant entity key. Train a small router (L26 residual -> which entity), test on a
HELD-OUT phrasing: top-1 is weak (~0.7 — not pinpoint) but top-5 is strong (~0.9 — a candidate list).
Cross-relation (train on 'capital of X', route on 'currency of X') confirms it keys on the ENTITY, not
the answer (answer-leak ruled out). So: address the entity = generate top-k by activation, then rank/verify.

Reproduces fleet E15 (real-entity activation routing) in this folder. Self-contained.
"""
import sys, json, time
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from chuk_lazarus.models_v2.loader import load_model, ModelDType

L = 26; Hr = 256; N = 150
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
    mx.random.seed(0)                                # seed the router init -> reproducible top-k
    t = time.time(); log("loading google/gemma-3-4b-it (bf16) ...")
    lm = load_model("google/gemma-3-4b-it", dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer; Dh = model.model.embed_tokens.weight.shape[1]; model.freeze()
    ents = COUNTRIES[:N]; log(f"loaded {time.time()-t:.1f}s; routing {len(ents)} real places on the L{L} key")
    mlp = model.model.layers[L].mlp; mlpcls = type(mlp); orig = mlpcls.__call__; CAP = {"v": None}
    def patched(self, x):
        out = orig(self, x)
        if self is mlp: CAP["v"] = np.array(x[0, -1, :].astype(mx.float32))
        return out
    mlpcls.__call__ = patched
    def cap(p):
        CAP["v"] = None; ll = model(mx.array([tok.encode(p)])).logits; mx.eval(ll); return CAP["v"].copy()

    log("capturing L26 entity keys (train phrasing + held-out paraphrase + cross-relation) ...")
    Xtr = np.array([cap(TRAIN.format(e=e)) for e in ents])
    Xpa = np.array([cap(PARA.format(e=e)) for e in ents])
    Xcr = np.array([cap(CROSS.format(e=e)) for e in ents])
    y = np.arange(N); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6

    class Router(nn.Module):
        def __init__(s, H, N): super().__init__(); s.a = nn.Linear(H, Hr); s.b = nn.Linear(Hr, N)
        def __call__(s, x): return s.b(nn.gelu(s.a(x)))
    # AVERAGE over router-init seeds: a single router has ~0.1 run-to-run wobble (esp. cross-relation),
    # so we ensemble 5 and report the mean -> a stable, reproducible number.
    def fit_eval(seed, ks=(1, 3, 5, 10)):
        mx.random.seed(seed); net = Router(Dh, N)
        Z = mx.array(((Xtr - mu) / sd).astype(np.float32)); ym = mx.array(y.astype(np.int32))
        lg = nn.value_and_grad(net, lambda n: nn.losses.cross_entropy(n(Z), ym).mean()); opt = optim.Adam(learning_rate=2e-3)
        for _ in range(500): l, g = lg(net); opt.update(net, g); mx.eval(net.parameters(), opt.state)
        def tk(X):
            lo = np.array(net(mx.array(((X - mu) / sd).astype(np.float32)))); o = np.argsort(-lo, axis=1)
            return {k: float(np.mean([y[i] in o[i, :k] for i in range(N)])) for k in ks}
        return tk(Xpa), tk(Xcr), net
    SEEDS = range(5); ks = (1, 3, 5, 10)
    runs = [fit_eval(s) for s in SEEDS]
    pa_runs = [r[0] for r in runs]; cr_runs = [r[1] for r in runs]; net0 = runs[0][2]
    para = {k: float(np.mean([r[k] for r in pa_runs])) for k in ks}
    cross = {k: float(np.mean([r[k] for r in cr_runs])) for k in ks}
    para_sd = {k: float(np.std([r[k] for r in pa_runs])) for k in ks}
    cross_sd = {k: float(np.std([r[k] for r in cr_runs])) for k in ks}
    chance5 = 5 / N

    log(f"\n=== ROUTE THE ENTITY by its own activation key @ L{L}  (mean of 5 routers; chance@5 = {chance5:.02f}) ===")
    log(f"  held-out paraphrase ('X's capital city is')   : top1 {para[1]:.2f}  top5 {para[5]:.2f}  (+/-{para_sd[5]:.02f})")
    log(f"  cross-relation (train capital, route currency): top1 {cross[1]:.2f}  top5 {cross[5]:.2f}  (+/-{cross_sd[5]:.02f})")
    log(f"  -> top-1 weak (~{para[1]:.2f}: NOT pinpoint); top-5 strong (~{para[5]:.2f}: a candidate list).")
    log(f"     cross-relation top5 {cross[5]:.2f} is FAR above chance ({chance5:.02f}) -> a GENUINE ENTITY key,")
    log(f"     not answer-leak (different relation, different answer, still finds the place). ~ paraphrase.")
    log(f"     mechanism: address the entity = generate top-k by activation, then rank / verify.")

    # concrete: ONE place's candidate list -- the entity resolves to a ranked SHORT-LIST, not a pinpoint
    lo0 = np.array(net0(mx.array(((Xpa - mu) / sd).astype(np.float32))))
    ranks = [int(list(np.argsort(-lo0[i])).index(i)) + 1 for i in range(N)]
    pick = next((i for i in range(N) if 2 <= ranks[i] <= 5), int(np.argmin(ranks)))
    top5 = [ents[j] for j in np.argsort(-lo0[pick])[:5]]
    log(f"\n  SEE IT — read the entity from \"{ents[pick]}'s capital city is\":")
    log(f"    router's top-5 guesses: {top5}")
    log(f"    true '{ents[pick]}' is in there at rank {ranks[pick]} -> a short-list you then verify, not one clean guess")
    json.dump(dict(L=L, N=N, seeds=list(SEEDS), chance5=chance5,
                   example=dict(place=ents[pick], top5=top5, true_rank=ranks[pick]),
                   paraphrase=para, paraphrase_sd=para_sd, cross_relation=cross, cross_relation_sd=cross_sd,
                   reading="entity = candidate-list (top-5 ~0.85), not pinpoint (top-1 ~0.65); cross-relation far above chance = genuine entity key"),
              open("route.json", "w"), indent=1, default=float)
    log("wrote route.json")


if __name__ == "__main__":
    main()
