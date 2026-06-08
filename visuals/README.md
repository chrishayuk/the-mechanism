# Visuals — the mechanism film

Code-generated graphics for the non-terminal **VISUAL** cues in `../SCRIPT.md`. Same ethos as
the terminal captures: built by hand in Python, black background, monospace, and the **same
hand-placed numbers** the `pack` capture prints (regenerated from rng seed 27 in `vstyle.py`,
so the graphics and the terminal never disagree).

```bash
python3 make_visuals.py        # generate everything into out/
python3 v1_spot_in_space.py    # or one at a time
```

Each "build" visual writes numbered PNG frames **and** an `.mp4` (ffmpeg; falls back to GIF).

| File | SCRIPT.md cue | What it shows |
|------|---------------|---------------|
| `v1_spot_in_space` | Part 1, "spot in space" establishing graphic *(must-build)* | a fact = a spot; three spots (capital/currency/language) as directions; their tip-to-tail sum = the packed code `[-0.17 0.81 0.37 -1.82 0.18 -0.53]`. 5 frames. |
| `v2_pack_sum` | Part 1, "three rows sum into one; the 4th column highlights" *(B-roll)* | the three real `pack` rows; column 4 (`-0.48 -0.59 -0.75`) adds down to `-1.82`; the packed code row revealed. 5 frames. |
| `v3_conveyor` | Part 3, "address-not-unpack conveyor" *(must-build)* | the question computes an address (relation+entity); the fact-band FFN lookup writes `Paris` onto the belt already separated; late layers read it. 4 frames. |
| `v4_edge_node` | Part 4, "edge-vs-node diagram" *(must-build)* | `France --capital-of--> Paris`; the relation index clean/roomy (generalises) vs entity slots crowded (read hundreds, write a dozen, then collide). Still. |
| `card_title` | Cold open title card | "The Mechanism. / How a transformer reads / its own knowledge." |
| `card_one_in_three` | Part 2 beat | "packed channel: 1 in 3." |
| `card_xor` | Part 3 caption | "a linear reader can't separate XOR — Minsky & Papert, 1969." |
| `card_closing` | Closing card | the three-line close. |
| `card_thumb` | Thumbnail candidate | "It Doesn't Unpack Its Memory. It Addresses It." |

**Palette** (`vstyle.py`): capital = cyan, currency = amber, language = green, packed code = pink,
relation-clean = green, collision = red. 1920×1080 @ dpi 120.

Still open from the script's checklist and not built here (they need external footage, not
code): the Video-3 recap flash (`larql> France capital-of -> Paris`) and the warm-up B-roll.
