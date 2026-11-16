#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements registered, examiner-judged learning experiments
# for its clients. If your team needs expertise in measuring whether a model
# keeps improving across generations, you can procure our services by sending
# an email to info@swedishembedded.com.

"""Evaluate RSI5's registered criteria (prereg/rsi5.md) from the examiner's signed group verdict (plan rsi5-G).

G1: every step (step1..step4) has a 95% interval whose lower bound is above 0.
G2: gen4-over-base has a lower bound above 0.05.
G3: gen4-over-base restricted to the never-examined families (stratum family_set:v2) has a lower bound above 0.

The intervals are the examiner's (bootstrap over templates and lineages); this script only reads them. A missing comparison or
stratum is an incomplete verdict, never a failed criterion.

Usage: rsi5_criteria.py GROUPS_VERDICT.json [--json]
"""

import argparse
import json
import sys

STEPS = ("step1", "step2", "step3", "step4")
TOTAL = "gen4-over-base"
UNSEEN = "family_set:v2"
STRENGTH = 0.05


class Incomplete(ValueError):
    """The verdict lacks a comparison or stratum the criteria need."""


def _interval(node: dict, where: str) -> tuple[float, float]:
    interval = node.get("interval")
    if not interval or interval[0] is None:
        raise Incomplete(f"{where}: no interval")
    return float(interval[0]), float(interval[1])


def evaluate(verdict: dict) -> dict:
    comparisons = verdict.get("comparisons", {})
    for name in (*STEPS, TOTAL):
        if name not in comparisons:
            raise Incomplete(f"comparison {name} is missing")
    steps = {name: _interval(comparisons[name], name) for name in STEPS}
    failed = [name for name, (low, _) in steps.items() if not low > 0]
    total = _interval(comparisons[TOTAL], TOTAL)
    stratum = comparisons[TOTAL].get("strata", {}).get(UNSEEN)
    if stratum is None:
        raise Incomplete(f"{TOTAL}: stratum {UNSEEN} is missing")
    unseen = _interval(stratum, f"{TOTAL}/{UNSEEN}")
    return {
        "G1": {"met": not failed, "failed": failed, "intervals": steps,
               "differences": {n: comparisons[n].get("difference") for n in STEPS}},
        "G2": {"met": total[0] > STRENGTH, "interval": total, "difference": comparisons[TOTAL].get("difference"),
               "bound": STRENGTH},
        "G3": {"met": unseen[0] > 0, "interval": unseen, "difference": stratum.get("difference")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("verdict")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    with open(args.verdict) as f:
        result = evaluate(json.load(f))
    if args.json:
        json.dump(result, sys.stdout, indent=1)
        print()
        return 0
    for name in STEPS:
        low, high = result["G1"]["intervals"][name]
        print(f"{name}: {result['G1']['differences'][name]:+.4f} [{low:+.4f}, {high:+.4f}]")
    print(f"G1 {'MET' if result['G1']['met'] else 'NOT MET'}" + (f" (failed: {', '.join(result['G1']['failed'])})" if result['G1']['failed'] else ""))
    low, high = result["G2"]["interval"]
    print(f"G2 {'MET' if result['G2']['met'] else 'NOT MET'}: gen4-base {result['G2']['difference']:+.4f} [{low:+.4f}, {high:+.4f}], bound {STRENGTH}")
    low, high = result["G3"]["interval"]
    print(f"G3 {'MET' if result['G3']['met'] else 'NOT MET'}: unseen families {result['G3']['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
