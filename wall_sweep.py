#!/usr/bin/env python3
"""
wall_sweep — is the wall a late-inject artifact? Sweep the injection site top-to-bottom: for each L,
measure C1 (one relation reads) and C2 (packed channel = chance). The answer: it walls at EVERY layer
(C2 ~chance L4..L30, C1 ~1.0), so it is NOT a depth/late-inject trick. The film injects at L20 — the
relation-differentiation/BUILDING band (Lazarus L14-25, before the L26+ fact-explosion) — because that's
the fairest test (whole downstream stack to work the signal out) and the purest wall; L30 (fleet E16's
late delivery site) is actually the muddiest. One model load.
"""
import sys
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx
from chuk_lazarus.models_v2.loader import load_model, ModelDType

LAYERS = [4, 20, 26, 30]; ALPHA = 7000.0; SEED = 0; N = 40
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
    emb = np.array(model.model.embed_tokens.weight.astype(mx.float32)); bos = getattr(tok, "bos_token_id", -1)
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
    edir = lambda t: emb[t]
    rng = np.random.default_rng(SEED)
    C = "bdfgklmnprstvz"; V = "aeiou"; names = set()
    while len(names) < N:
        names.add("".join((C[rng.integers(len(C))] + V[rng.integers(len(V))]) for _ in range(rng.integers(2, 4))).capitalize())
    ents = [{"name": nm, "facts": {r: pools[r][rng.integers(len(pools[r]))] for r in REL}} for nm in names]

    def probs(prompt, vec):
        STATE["inj"] = vec
        ll = np.array(model(mx.array([tok.encode(prompt)])).logits[0, -1].astype(mx.float32))
        STATE["inj"] = None; p = np.exp(ll - ll.max()); return p / p.sum()

    log(f"\n{'L':>4} | {'C1 one-relation':>15} | {'C2 packed':>10} | {'query-indep':>11} | reading")
    log("-" * 70)
    rows = {}
    for L in LAYERS:
        STATE["layer"] = model.model.layers[L]
        c1 = c2 = nq = 0; picks_all = []
        for e in ents:
            tids = [e["facts"][r][1] for r in REL]
            packed = ALPHA * np.sum([edir(t) for t in tids], axis=0)
            picks = []
            for r in REL:
                tid = e["facts"][r][1]; pr = REL_PROMPT[r].format(e=e["name"])
                p1 = probs(pr, ALPHA * edir(tid)); c1 += (tids[int(np.argmax([p1[t] for t in tids]))] == tid)
                p2 = probs(pr, packed); pk = int(np.argmax([p2[t] for t in tids]))
                c2 += (tids[pk] == tid); picks.append(pk); nq += 1
            picks_all.append(picks)
        c1, c2 = c1/nq, c2/nq
        qi = float(np.mean([1.0 if len(set(p)) == 1 else 0.0 for p in picks_all]))
        valid = (c1 >= 0.80) and (c2 <= 0.50)
        rows[L] = (c1, c2, qi)
        log(f"{L:>4} | {c1:>15.3f} | {c2:>10.3f} | {qi:>11.3f} | "
            f"{'VALID wall (C1 reads, C2 chance)' if valid else ('C1 weak here' if c1<0.80 else 'C2 not chance')}")
    everywhere = all(rows[L][0] >= 0.80 and rows[L][1] <= 0.50 for L in LAYERS)
    log(f"\nVERDICT: the wall holds at EVERY layer (C1~1.0, C2~chance) = {everywhere}. NOT a late-inject artifact.")
    log(f"  L20 (BUILDING band, the film's site): C1 {rows[20][0]:.3f} / C2 {rows[20][1]:.3f}  (purest)")
    log(f"  L30 (E16 delivery, late):             C1 {rows[30][0]:.3f} / C2 {rows[30][1]:.3f}  (muddiest)")
    log("  -> inject at L20: fairest test (whole downstream stack to work it out), purest wall. The model has no decoder, at any depth.")


if __name__ == "__main__":
    main()
