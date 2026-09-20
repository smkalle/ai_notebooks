#!/usr/bin/env python3
"""Generate a FlyLM split.

    python3 make_dataset.py --split train   --per-cell 20 --out data/train.jsonl
    python3 make_dataset.py --split holdout --per-cell 12 --out data/holdout.jsonl --seed 101

Exact duplicates are dropped and reported rather than silently shipped: a corpus whose
"60K rows" are 12K rows repeated five times trains a model that has seen 12K rows.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

from flylm.generate import generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "holdout"], default="train")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-cell", type=int, default=8,
                    help="paraphrases per (fruit, stage, intent) cell")
    ap.add_argument("--world-per-template", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    if a.split == "holdout" and a.seed == 7:
        a.seed = 101                      # different stream from train, by default

    rows = generate(a.split, seed=a.seed, per_cell=a.per_cell,
                    world_per_template=a.world_per_template)

    seen, unique, dupes = set(), [], 0
    for r in rows:
        key = (r["prompt"], r["response"])
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        unique.append(r)

    by_cell = collections.Counter(
        (r["intent"], r["fruit"], r["stage"]) for r in unique if r["fruit"])
    short = [c for c, n in by_cell.items() if n < a.per_cell]

    print(f"split={a.split} seed={a.seed} per_cell={a.per_cell}")
    print(f"  generated {len(rows)}  unique {len(unique)}  dropped {dupes} exact duplicates")
    print(f"  cells {len(by_cell)}   under-target cells {len(short)}")
    if short:
        worst = min(by_cell[c] for c in short)
        print(f"  note: {len(short)} cells fell short of --per-cell (min {worst}). the phrase")
        print(f"        inventory is the ceiling, not the loop: add sensory/verdict clauses in")
        print(f"        flylm/ontology.py and flylm/generate.py rather than raising --per-cell.")
    for k in ("intent", "stage", "family", "verdict"):
        c = collections.Counter(r[k] for r in unique)
        print(f"  {k:8} " + "  ".join(f"{v}:{n}" for v, n in sorted(
            c.items(), key=lambda x: (x[0] is None, x[0]))))

    if a.stats_only:
        return 0
    out = pathlib.Path(a.out or f"data/{a.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unique))
    print(f"  -> {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
