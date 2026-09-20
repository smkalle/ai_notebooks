#!/usr/bin/env python3
"""The five disjointness checks from SPEC section 5.3. Exits non-zero on any failure.

Run this before every training run. A held-out set that overlaps the training set does
not measure generalization, it measures memory - and it will not announce itself.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from flylm.ontology import HOLDOUT_FRUITS, HOLDOUT_MARK, SENSORY, FRUIT_BY_NAME

NGRAM = 8


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def ngrams(s: str, n: int = NGRAM):
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--golden", default="../evals/golden/held_out_golden.jsonl")
    ap.add_argument("--holdout", default="data/holdout.jsonl")
    a = ap.parse_args()

    train = load(a.train)
    if train is None:
        print(f"no train split at {a.train} - run make_dataset.py first")
        return 2
    golden = load(a.golden) or []
    holdout = load(a.holdout) or []
    evalrows = golden + holdout
    fails = []

    # 1 - fruit disjointness
    hold_names = {f.name for f in HOLDOUT_FRUITS}
    leaked = sorted({r["fruit"] for r in train if r.get("fruit") in hold_names}
                    | {n for n in hold_names for r in train if n in r["prompt"] or n in r["response"]})
    print(f"1 fruit disjointness      : {'FAIL ' + str(leaked) if leaked else 'ok (0 of 7 holdout fruits in train)'}")
    fails += ["fruit"] if leaked else []

    # 2 - template / sensory disjointness
    reserved = {p[: -len(HOLDOUT_MARK)] for st in SENSORY.values() for ph in st.values()
                for p in ph if p.endswith(HOLDOUT_MARK)}
    used = sorted({p for p in reserved for r in train if p in r["response"]})
    print(f"2 reserved-phrase leakage : {'FAIL ' + str(used[:3]) if used else f'ok (0 of {len(reserved)} reserved clauses in train)'}")
    fails += ["template"] if used else []

    # 3 - exact prompt disjointness
    tp = {norm(r["prompt"]) for r in train} - {""}
    overlap = sorted({r["prompt"] for r in evalrows
                      if norm(r["prompt"]) and norm(r["prompt"]) in tp})
    print(f"3 exact-prompt overlap    : {'FAIL ' + str(len(overlap)) + ' prompts, e.g. ' + repr(overlap[0]) if overlap else 'ok (0)'}")
    fails += ["exact"] if overlap else []

    # 4 - n-gram disjointness
    tg = set()
    for r in train:
        tg |= ngrams(r["prompt"])
    hits = [r["id"] for r in evalrows if ngrams(r["prompt"]) & tg]
    print(f"4 {NGRAM}-gram prompt overlap  : {'FAIL ' + str(len(hits)) + ' cases, e.g. ' + hits[0] if hits else 'ok (0)'}")
    fails += ["ngram"] if hits else []

    # 5 - response leakage
    tr = {norm(r["response"]) for r in train}
    leaks = [r["id"] for r in evalrows if norm(r.get("reference", r.get("response", ""))) in tr]
    print(f"5 reference-response leak : {'FAIL ' + str(len(leaks)) + ' cases, e.g. ' + leaks[0] if leaks else 'ok (0)'}")
    fails += ["response"] if leaks else []

    print(f"\ntrain {len(train)} rows | golden {len(golden)} | generated holdout {len(holdout)}")
    if fails:
        print(f"DISJOINTNESS FAILED: {', '.join(fails)} - do not train on this split")
        return 1
    print("all five checks pass - the held-out set measures generalization, not memory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
