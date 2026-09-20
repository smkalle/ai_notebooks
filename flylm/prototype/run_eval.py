#!/usr/bin/env python3
"""Score an adapter against the frozen FlyLM held-out suite.

    python3 run_eval.py --adapter null        # must fail everything
    python3 run_eval.py --adapter constant    # must pass voice, fail safety
    python3 run_eval.py --adapter oracle      # must pass everything
    python3 run_eval.py --adapter hf:Qwen/Qwen3-0.6B-Base --reps 3

Run the first three before training anything. If the oracle does not pass and the null
does not fail, the suite is not evidence about a model - it is a bug.
"""
from __future__ import annotations

import argparse
import json
import sys

from flylm import adapters, harness

GOLDEN = "../evals/golden/held_out_golden.jsonl"


def fmt(v) -> str:
    return "  n/a " if v is None or v != v else f"{v:6.3f}"


def report(s: dict) -> int:
    print(f"\nadapter: {s['adapter']}   reps: {s['reps']}   rows: {s['n_rows']}"
          f"   errors: {s['n_errors']}")
    print(f"golden_hash: {s['golden_hash'][:16]}…")
    g = s["global_gates"]
    print(f"\nglobal gates   hard-gate pass {fmt(g['hard_gate_pass_rate'])}"
          f"   soft score {fmt(g['mean_soft_score'])}"
          f"   rows failing hard gates: {g['rows_failing_hard_gates']}"
          f"   most common: {g['most_common_failed_rule']}")
    srp = g.get("soft_rule_pass", {})
    if srp:
        print("soft rules     " + "   ".join(f"{k} {fmt(x)}" for k, x in srp.items())
              + "   (V4 measured, not scored)")
        lo, hi = g.get("first_person_band", (0, 1))
        fp = g.get("first_person_rate", 0)
        print(f"first person   {fmt(fp)}  band [{lo}, {hi}]"
              f"  {'ok' if lo <= fp <= hi else 'OUT OF BAND'}")
    c = s["classifier"]
    print(f"tier-1 classifier   abstained {fmt(c['abstention_rate'])}"
          f"   low confidence {fmt(c['low_confidence_rate'])}")

    print(f"\n{'slice':<18}{'n':>4}{'pass':>8}{'95% ci':>16}  gates")
    print("-" * 78)
    failed_hard = []
    for name, sl in s["slices"].items():
        ci = sl["pass_rate"]["ci95"]
        gates = []
        for gt in sl["gates"]:
            mark = {"pass": "ok", "FAIL": "FAIL", "not_measured": "--"}[gt["status"]]
            gates.append(f"{gt['metric']}={fmt(gt.get('value'))}[{mark}]")
            if gt["status"] == "FAIL" and gt["hard"]:
                failed_hard.append(f"{name}.{gt['metric']}")
        defer = f" (+{sl['n_deferred_to_judge']} deferred)" if sl["n_deferred_to_judge"] else ""
        print(f"{name:<18}{sl['n_cases']:>4}{fmt(sl['pass_rate']['value'])}"
              f"  [{ci[0]:.3f},{ci[1]:.3f}]{defer}")
        for g_ in gates:
            print(f"{'':<18}      {g_}")
    print("-" * 78)
    total_deferred = sum(sl["n_deferred_to_judge"] for sl in s["slices"].values())
    if total_deferred:
        print(f"note: {total_deferred} rows the tier-1 classifier could not read are DEFERRED, "
              f"not scored.\n      run with the judge (flylm.judge) to resolve them; until then "
              f"these slices are\n      measured on a subset and their intervals are wider than "
              f"they look.")
    if failed_hard:
        print(f"HARD GATES FAILED: {', '.join(failed_hard)}")
    else:
        print("all hard gates pass")
    return len(failed_hard)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="null | constant | oracle | hf:<model_id>")
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--slice", default=None, help="score one slice only")
    ap.add_argument("--json", action="store_true", help="print summary.json and nothing else")
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter dir for hf:")
    a = ap.parse_args()

    cases, golden_hash = harness.load_cases(a.golden)
    if a.slice:
        cases = [c for c in cases if c["slice"] == a.slice]
        if not cases:
            raise SystemExit(f"no cases in slice {a.slice!r}")
    kw = {"adapter_path": a.adapter_path} if a.adapter.startswith("hf:") and a.adapter_path else {}
    adapter = adapters.build(a.adapter, cases, **kw)
    out = a.out or f"runs/{a.adapter.replace(':', '_').replace('/', '_')}"
    s = harness.run(adapter, cases, reps=a.reps, out_dir=out, golden_hash=golden_hash)
    if a.json:
        print(json.dumps(s, indent=2))
        return 0
    n_failed = report(s)
    print(f"\nrows -> {out}/results.jsonl   summary -> {out}/summary.json")
    return 0            # a failing gate is a finding, not a crash


if __name__ == "__main__":
    sys.exit(main())
