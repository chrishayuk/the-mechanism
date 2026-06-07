#!/usr/bin/env python3
"""
ffn — build a transformer FFN BY HAND as a key->value memory, and read it by ADDRESSING (one matmul).

This is the OTHER way to pack, and it's the model's way (Geva et al.: an FFN layer IS key->value memory).
  pack.py:  superposition — all facts ADDED onto one channel; reading needs iterative de-mixing (OMP).
  ffn.py:   addressing    — each fact is a (key -> value) ROW; a query matches a key, the value FIRES.
            no iteration, no interference between facts: capacity = number of neurons, read = one matmul.

An FFN is:  h = act(W_in @ x);  out = W_out^T @ h.   We PLANT memories by hand: row i of W_in is an
address (a key), row i of W_out is the answer at that address (a value). Feed the address, the matching
neuron lights up, and it writes its value. That's exactly what trace.py shows the real model doing.
Self-contained, CPU/numpy, no training.
"""
import numpy as np

d = 24                                          # model width (tiny, for legibility)
FACTS = [("capital  of Atlantis", "Paris"), ("currency of Atlantis", "Euro"),  ("language of Atlantis", "Latin"),
         ("capital  of Zerivia",  "Cairo"), ("currency of Zerivia",  "Rand"),  ("language of Zerivia",  "Tamil")]


def unit(v): return v / (np.linalg.norm(v) + 1e-9)


def main():
    rng = np.random.default_rng(0)
    addrs = [a for a, _ in FACTS]; answers = [b for _, b in FACTS]
    vocab = sorted(set(answers))
    # each ADDRESS -> a key direction; each ANSWER WORD -> a value direction (shared across facts with same answer)
    key = {a: unit(rng.standard_normal(d)) for a in addrs}
    val = {w: unit(rng.standard_normal(d)) for w in vocab}

    # BUILD THE FFN BY HAND: W_in rows = keys (address detectors), W_out rows = values (what each writes)
    W_in = np.stack([key[a] for a in addrs])                 # (n_facts, d)  -- each neuron detects one address
    W_out = np.stack([val[answers[i]] for i in range(len(FACTS))])  # (n_facts, d)  -- and writes that answer

    def ffn(x):
        h = np.maximum(W_in @ x, 0.0)                        # which address-neuron fires? (ReLU key match)
        return h, W_out.T @ h                                # the firing neuron writes its value into the output

    def read_word(out):                                      # decode the FFN output back to an answer word
        return max(vocab, key=lambda w: out @ val[w])

    print("=== A TRANSFORMER FFN, BUILT BY HAND  (h = relu(W_in @ x);  out = W_out^T @ h) ===")
    print(f"  W_in : {W_in.shape[0]} rows x {W_in.shape[1]} dims  -- each ROW is an ADDRESS the neuron detects")
    print(f"  W_out: {W_out.shape[0]} rows x {W_out.shape[1]} dims  -- each ROW is the VALUE that neuron writes")
    print(f"  planted {len(FACTS)} facts, one neuron each:")
    for i, (a, b) in enumerate(FACTS):
        print(f"     neuron {i}:  detect [{a}]  ->  write '{b}'")

    print("\n=== READ IT BY ADDRESSING — feed an address, ONE matmul, the value fires ===")
    correct = 0
    for a, b in FACTS:
        h, out = ffn(key[a])
        fired = int(np.argmax(h)); got = read_word(out)
        correct += (got == b)
        bar = "".join("#" if h[j] > 0.5 else ("." if h[j] > 1e-6 else " ") for j in range(len(FACTS)))
        print(f"  ask [{a}]  ->  neurons [{bar}] fire (#{fired})  ->  out reads '{got}'   {'OK' if got==b else 'X'}")
    print(f"\n  recovered {correct}/{len(FACTS)} by a single key->value lookup — NO iteration, NO de-mixing.")
    print("  to add a fact: add a neuron (a row). facts don't interfere — capacity is # neurons, not de-mix budget.")
    print("\n  pack.py packs by ADDING onto one channel (needs an OMP de-mixer to read).")
    print("  THIS packs by ADDRESSING (key->value rows; read = one matmul) — and it's how the model does it")
    print("  (Geva et al.: FFN = key->value memory). trace.py shows the real model writing+reading exactly this way.")


if __name__ == "__main__":
    main()
