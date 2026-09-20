# FlyLM prototype

The runnable half of [`../SPEC.md`](../SPEC.md): the ontology, the voice contract, the
graders, the seeded generator, the Claude judge, and the harness that ties them together.

Pure standard library. No API key needed for anything below except the judge.

```bash
cd flylm/prototype

# 1. validate the harness BEFORE generating data or training anything
python3 run_eval.py --adapter null        # must fail every slice and every hard gate
python3 run_eval.py --adapter constant    # must pass voice, must fail safety
python3 run_eval.py --adapter oracle      # must pass everything

# 2. only then generate a corpus
python3 make_dataset.py --split train   --per-cell 20 --out data/train.jsonl
python3 make_dataset.py --split holdout --per-cell 12 --out data/holdout.jsonl

# 3. prove the held-out set measures generalization, not memory
python3 verify_holdout.py

# 4. the graders' own known-good / known-bad suite
python3 -m unittest discover -s tests -v
```

## Current state

Reproduced on a clean checkout:

| step | result |
|---|---|
| `--adapter null` | 0.000 on all nine slices; 154/154 rows fail hard gates |
| `--adapter constant` | voice 1.000, safety 0.000, containment 0.000, curve cliff 0.000 |
| `--adapter oracle` | **1.000 on all nine slices**, hard gates 1.000, soft 0.991 |
| `verify_holdout.py` | all five checks pass |
| `unittest` | 27 tests, all pass |

The oracle run also reports `tier-1 classifier abstained 0.117` — 12% of rows the phrase
classifier will not guess at. Those are the judge's share, and they are reported rather
than resolved by guesswork.

## What the validation runs actually caught

They are in the spec because they work. Building this, they caught, in order:

1. **A flat curve counted as monotone.** The null adapter scored 1.000 on
   `monotone_groups` — every enthusiasm was 0, and "non-decreasing" is trivially true of
   all zeros. `grade_group` now requires the curve to actually rise.
2. **A cheerful constant passed containment.** `"it smells sweet. i love it."` routes
   through the nose and leaks nothing, so it beat the refusal slice. Containment now also
   requires the answer to name its limit or ask — which is what makes the refusal honest
   rather than merely on-topic.
3. **Empty output left hard gates "not measured" instead of failed.** A model that
   answers nothing looked like a model nobody checked.
4. **V4 (first person) was a bad rule.** The hand-written golden references pass it only
   44.8% of the time — `"sweet and wide open. yes. this is the good day."` is perfect fly
   and contains no "i". It is now measured as a corpus-level rate with a band, not scored
   per row. Without the oracle run this would have penalised a correct model on nearly
   half the suite.
5. **V6 flagged "that one" and "the good day".** The rule is about units a fly cannot
   smell, not about numerals; the standalone numeral test only produced false positives.
6. **The classifier read a bare "yes" as a ripeness verdict.** `"the cold place, yes"`
   answers the storage question. Weak tokens now defer to the odor markers.
7. **`avoid` zeroed the enthusiasm of a model excited about mould.** `"wet ground and
   yeast. i live here now."` scored a clean cliff. Caught by a unit test, not by a run.
8. **The generator emitted 14 golden references verbatim.** A golden set and a generator
   written by the same author share a vocabulary. The generator now routes around the
   golden set structurally (`generate.collides`), so `verify_holdout.py` checks 3–5 are
   regression guards rather than things you discover after a training run.

Every one of these would have produced confident numbers pointing the wrong way.

## What is not built here

Stated plainly so the spec's budget is not mistaken for shipped code:

- **`compare`, `rank` and the 2-turn `trajectory` families are not implemented.** The
  generator covers 12 of the 14 intents: six fruit-dependent single-item intents and six
  world intents. The golden set *does* contain compare/rank-shaped safety cases, so the
  graders handle them; the generator does not yet emit them.
- **The phrase inventory is a reduced sample.** Three sensory clauses per (family, stage)
  and three verdict clauses per (intent, stage), one of each reserved for the held-out
  split. That supports ~12 distinct responses per cell against the spec's 45, so
  `--per-cell 20` under-delivers on most cells and says so. Extending the inventory in
  `ontology.SENSORY` and `generate.VERDICTS` is the work, not raising the flag.
- **The generated held-out tier reaches ~540 rows, not 3,000**, for the same reason: the
  reserved `@h` pool is one clause per slot. Until the inventory grows, the 154-case
  golden set is the suite that matters and its noise floor (±8 points at one rep) is the
  honest one to quote.
- **No training code.** Tracks A/B/C in SPEC §8 are the notebook's job. `adapters.HFAdapter`
  is the seam they plug into, and its generation config is pinned there so every track is
  scored under identical sampling.
- **The judge has never been run.** `flylm/judge.py` is complete and import-safe without
  credentials, but no call has been made and it has not been calibrated. Until
  `judge.calibrate()` clears ~0.90 agreement against human labels, its scores should not
  steer a decision.

## Layout

```
flylm/ontology.py    three axes, families, stages, intents, the phrase lexicons
flylm/voice.py       V1-V10, deterministic, zero cost
flylm/graders.py     the tier-1 classifier and one grader per slice
flylm/generate.py    the seeded generator and its six hard invariants
flylm/judge.py       the Claude judge: schema, untrusted wrapper, batch path, calibration
flylm/adapters.py    null / constant / oracle / hf
flylm/harness.py     run, record, aggregate, Wilson intervals, threshold gates
```

`runs/` and `data/` are generated and git-ignored. Everything in them is reproducible
from a seed plus the golden hash.
