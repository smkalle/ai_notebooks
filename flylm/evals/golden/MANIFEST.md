# Golden held-out set — manifest

**154 cases. Hand-authored. Frozen.**

`held_out_golden.jsonl`, one case per line, conforming to `../eval_case.schema.json`.

## Provenance

`gold_source` is `human` on every case, and that field is not decoration. The reference
answers were written by a person before the corpus generator existed, which is the only
reason this tier can catch a generator bug. If the golden references were ever
regenerated from a model, reference-similarity scoring would start rewarding *imitation
of that model* instead of correctness, and a comparison between two fine-tunes would be
biased toward whichever one resembled the generator. So:

- Never regenerate this file from a model.
- Never add a case whose reference came from a model, even a good one.
- New cases are appended with a new id; existing cases are not silently edited. A
  reference that has to change means the ontology changed, and scores from before and
  after are not comparable — bump the freeze hash and say so in the results.

## Freeze

```
sha256(held_out_golden.jsonl) = 6931e8ac3f2192aff8f77db292ac816634d5a38fe413134e1e7097ec8d08b1a0
→ recorded in every results.json as `golden_hash`
```

Compute with `python3 -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('held_out_golden.jsonl').read_bytes()).hexdigest())"`.
Every scored run records the hash it ran against. Two runs with different hashes are two
different evals and their numbers do not go in the same table.

## Composition

| slice | cases |
|---|---|
| `stage_accuracy` | 42 |
| `safety_geosmin` | 20 |
| `enthusiasm_curve` | 18 (3 groups × 6 stages) |
| `containment` | 18 |
| `unseen_fruit` | 18 |
| `indifference` | 12 |
| `robustness` | 10 |
| `abstention` | 10 |
| `voice` | 6 |

Stage coverage: `0:14  1:15  2:27  3:18  4:14  5:33  n/a:33`.

Stage 5 is deliberately oversampled across the suite because it is the consequential
class and the one a mean would hide. Within `stage_accuracy` the distribution is flat —
7 cases per stage — so the majority-class baseline for that slice's confusion matrix is
1/6 ≈ 0.167 and nothing about the imbalance elsewhere inflates it.

Family coverage: all 5 families, plus all 7 held-out fruits (18 `unseen_fruit` cases).

## Both-directions coverage

Deliberate, and the reason several cases look redundant:

- `danger` where the answer is **avoid** (S4, 20 cases) *and* `danger` where a worried
  human is holding a perfectly good fruit (`S9-abstain-05`, `S9-abstain-06`: white bloom
  on grapes, sugar crystals on dates). A suite with only the first is scored 100% by a
  model that always says no.
- Fruits the fly **loves** *and* fruits it is **indifferent** to (S3, 12 cases), because
  a model excited about everything is a broken sensor and the enthusiasm curve alone
  would not notice.
- `indifferent` fruits below stage 5 (*not for me*) *and* the same fruits at stage 5
  (*avoid*) — `S4-avocado-danger`, `S4-citrus_peel-danger`. Indifference is not
  permission; geosmin overrides it.
- Climacteric fruit with a future (`S1-fig-1-storage`) *and* non-climacteric fruit
  without one (`S1-strawberry-1-window`, `S1-grape-1-storage`), so "leave it on the
  counter" cannot be a safe universal answer.
- Enthusiasm that should transfer to an unseen fruit (`S7-lychee-*`) *and* enthusiasm
  that should **not** (`S7-pomelo_peel-*`).

## Known non-duplicates

Three prompts appear six times each — `what do you make of this apple`, `how do you feel
about this papaya`, `what about this cherry`. That is the `enthusiasm_curve` design: one
prompt held constant while the stage varies, which is what makes the group gradeable as a
monotonic property. A duplicate check must key on `(prompt, stage)`, not `prompt`.

## Audit status

| check | status |
|---|---|
| schema-valid, no duplicate ids | enforced by `run_eval.py` at load |
| every reference passes its own grader | `run_eval.py --adapter oracle` |
| null output fails everything | `run_eval.py --adapter null` |
| a cheerful constant fails safety | `run_eval.py --adapter constant` |
| no reference exceeds the 32-word voice limit | max observed 28 |
| gold not reachable by the model under test | this file is never concatenated into `data/train.jsonl`; `verify_holdout.py` check 5 |
| labels independently re-derived | pending: a second human re-labels a 20-case sample before the first paid run |

The last row is open on purpose. It is the one item here that a script cannot close.
