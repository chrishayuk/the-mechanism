#!/usr/bin/env python3
"""
make_visuals — generate every graphic for the mechanism film in one go.

Same ethos as the captures: built by hand in code, black background, monospace, and the
SAME hand-placed numbers the `pack` terminal prints. Writes PNG frames + mp4 builds to
visuals/out/. Run:  python3 make_visuals.py
"""
import importlib

MODULES = ["v1_spot_in_space", "v2_pack_sum", "v3_conveyor", "v4_edge_node", "v5_cards"]


def main():
    for name in MODULES:
        print(f"\n# {name}")
        importlib.import_module(name).main()
    print("\nall visuals written to visuals/out/")


if __name__ == "__main__":
    main()
