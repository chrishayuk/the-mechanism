#!/usr/bin/env python3
"""
wall — Part 2 capture: ONE relation reads; the PACKED channel does not. The model has no decoder.

Frozen Gemma-3-4b (MLX). Reproduces E5's C1/C2 on NOVEL entities (so it's the injection, not prior
knowledge): inject ONE relation's answer direction -> the model reads it (~1.0); inject the SUM of an
entity's relations (the packed channel) -> query-conditioned read collapses to ~chance (1/3), and it's
amplify-strongest (one answer wins every question). Inject at L20 — the relation-differentiation /
BUILDING band (Lazarus L14-25), before the L26+ fact-explosion — so the model gets the whole downstream
stack to build/unpack the injected code. It still walls (purest here). wall_sweep.py shows it walls at
EVERY layer L4..L30, so it is not a late-inject artifact.

Self-contained in videos/the-mechanism/.
"""
import sys, json
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx
from chuk_lazarus.models_v2.loader import load_model, ModelDType

L = 20; ALPHA = 7000.0; SEED = 0; N = 40   # L20 = the relation-differentiation / BUILDING band (Lazarus L14-25),
#                                          # BEFORE the L26+ fact-explosion. Injecting here gives the model the whole
#                                          # downstream stack to "build/unpack" the code -- the fairest test -- and it
#                                          # STILL walls (purest: C2~chance, query-indep highest). The wall holds at
#                                          # EVERY layer L4..L30 (wall_sweep.py) -- not a late-inject artifact.
REL_PROMPT = {"capital": "The capital of {e} is", "currency": "The currency of {e} is",
              "language": "The language of {e} is"}
REL = list(REL_PROMPT.keys())
CANDS = {"capital": ["Oslo","Lima","Cairo","Dublin","Bern","Tokyo","Rome","Vienna","Prague","Athens","Lisbon","Quito"],
         "currency": ["Yen","Rand","Baht","Won","Krona","Dinar","Rupee","Real","Lira","Naira","Dong","Cedi"],
         "language": ["Dutch","Greek","Welsh","Thai","Czech","Polish","Danish","Finnish","Hindi","Korean","Tamil","Latin"]}
STATE = {"inj": None, "layer": None}
def log(*a): print(*a, flush=True)


def main():
    log("loading google/gemma-3-4b-it (bf16) ...")
    lm = load_model("google/gemma-3-4b-it", dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer; model.freeze()
    emb = np.array(model.model.embed_tokens.weight.astype(mx.float32))
    bos = getattr(tok, "bos_token_id", -1)
    STATE["layer"] = model.model.layers[L]
    blk = type(model.model.layers[0]); orig = blk.__call__
    def patched(self, x, mask=None, cache=None):
        res = orig(self, x, mask=mask, cache=cache)
        if self is STATE["layer"] and STATE["inj"] is not None:
            h = res.hidden_states; seq = h.shape[1]
            last = (mx.arange(seq) == seq - 1).reshape(1, seq, 1).astype(h.dtype)
            res = type(res)(hidden_states=h + last * mx.array(STATE["inj"].astype(np.float32)).reshape(1, 1, -1), cache=res.cache)
        return res
    blk.__call__ = patched

    def tid_of(v):
        ids = [i for i in tok.encode(" " + v) if i != bos]; return ids[0] if len(ids) == 1 else None
    pools = {r: [(v, tid_of(v)) for v in CANDS[r] if tid_of(v)] for r in REL}
    edir = lambda tid: emb[tid]
    rng = np.random.default_rng(SEED)
    C = "bdfgklmnprstvz"; V = "aeiou"; names = set()
    while len(names) < N:
        names.add("".join((C[rng.integers(len(C))] + V[rng.integers(len(V))]) for _ in range(rng.integers(2, 4))).capitalize())
    ents = [{"name": nm, "facts": {r: pools[r][rng.integers(len(pools[r]))] for r in REL}} for nm in names]

    def probs(prompt, vec):
        STATE["inj"] = vec
        ll = np.array(model(mx.array([tok.encode(prompt)])).logits[0, -1].astype(mx.float32))
        STATE["inj"] = None; p = np.exp(ll - ll.max()); return p / p.sum()

    # C0 novel (no inject), C1 single relation, C2 packed channel
    c0 = c1 = c2 = nq = 0; per_ent_picks = []
    for e in ents:
        tids = [e["facts"][r][1] for r in REL]
        packed = ALPHA * np.sum([edir(t) for t in tids], axis=0)
        picks = []
        for ri, r in enumerate(REL):
            tid = e["facts"][r][1]; pr = REL_PROMPT[r].format(e=e["name"])
            c0 += (int(np.argmax(probs(pr, None))) == tid)                                  # novel: knows nothing
            single = ALPHA * edir(tid)
            p1 = probs(pr, single); c1 += (tids[int(np.argmax([p1[t] for t in tids]))] == tid)   # one relation
            p2 = probs(pr, packed); pk = int(np.argmax([p2[t] for t in tids]))
            c2 += (tids[pk] == tid); picks.append(pk); nq += 1                                # packed channel
        per_ent_picks.append(picks)
    c0, c1, c2 = c0/nq, c1/nq, c2/nq
    qi = float(np.mean([1.0 if len(set(p)) == 1 else 0.0 for p in per_ent_picks]))          # amplify-strongest signature

    # concrete demo: a FIXED place (name + facts hand-chosen, verified amplify-strongest) so the on-screen
    # NAME is locked run-to-run. The 40-entity tally above is the statistical claim; this is the illustration.
    DEMO_NAME = "Marn"; DEMO = {"capital": "Cairo", "currency": "Rand", "language": "Tamil"}
    dtid = {r: tid_of(DEMO[r]) for r in REL}
    dpacked = ALPHA * np.sum([edir(dtid[r]) for r in REL], axis=0)
    qw = max(len(f"'{rr} of {DEMO_NAME}?'") for rr in REL)
    log("\nWATCH ONE PLACE — packed channel injected, ask 3 different questions:")
    log(f"  '{DEMO_NAME}' packed with   capital={DEMO['capital']}   currency={DEMO['currency']}   language={DEMO['language']}")
    for r in REL:
        p2 = probs(REL_PROMPT[r].format(e=DEMO_NAME), dpacked)
        pk = int(np.argmax([p2[dtid[rr]] for rr in REL])); said = DEMO[REL[pk]]; truth = DEMO[r]
        tag = "      <- same answer every time" if r == REL[-1] else ""
        q = f"'{r} of {DEMO_NAME}?'"
        log(f"    ask {q:<{qw}}  ->  '{said}'   ({'right' if said == truth else 'WRONG'}){tag}")

    log(f"\nTHE TALLY — {N} novel places, inject @ L{L} (the BUILDING band; every advantage):")
    log(f"  C0  clean baseline             :  {c0:.3f}")
    log(f"  C1  one clean relation alone   :  {c1:.3f}   (reads a single clean signal perfectly)")
    log(f"  C2  packed channel, all at once:  {c2:.3f}   (chance ~ 1/3 — it CAN'T unpack)")
    log(f"  query-independence             :  {qi:.3f}   (amplify-strongest — same answer regardless of question)")
    json.dump(dict(L=L, alpha=ALPHA, n=N, C0_novel=round(c0,3), C1_single=round(c1,3),
                   C2_packed=round(c2,3), query_independence=round(qi,3)),
              open("wall.json", "w"), indent=1, default=float)
    log("wrote wall.json")


if __name__ == "__main__":
    main()
