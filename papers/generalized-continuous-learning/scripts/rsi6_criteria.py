#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements registered, examiner-judged learning experiments
# for its clients. If your team needs expertise in measuring whether a model
# keeps acquiring new skills without losing old ones, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Evaluate RSI6's registered criteria (prereg/rsi6.md) from the examiner's signed group verdict (plan rsi6-H, language mean).

H1: step1..step4 (acquisition generation k vs k-1) each have a lower bound above 0.
H2: acquisition-over-volume (generation 4 of the acquisition lineages vs the volume-control lineages) has a lower bound above 0.
H3: for each earlier family j (lua51, python3, lua51-dialect-a), comparison retention<j> (generation 4 vs generation j) restricted
    to family j's own language stratum has a lower bound above -0.02 (at most a two-point drop).

The intervals are the examiner's; a missing comparison or stratum is an incomplete verdict, never a failed criterion.

Usage: rsi6_criteria.py GROUPS_VERDICT.json [--json]
"""

import argparse
import json
import sys

STEPS = ("step1", "step2", "step3", "step4")
VOLUME = "acquisition-over-volume"
RETAINED = {1: "lua51", 2: "python3", 3: "lua51-dialect-a"}
MARGIN = 0.02


class Incomplete(ValueError):
    """The verdict lacks a comparison or stratum the criteria need."""


def _interval(node, where):
    interval = (node or {}).get("interval")
    if not interval or interval[0] is None:
        raise Incomplete(f"{where}: no interval")
    return float(interval[0]), float(interval[1])


def _comparison(comparisons, name):
    if name not in comparisons:
        raise Incomplete(f"comparison {name} is missing")
    return comparisons[name]


def evaluate(verdict: dict) -> dict:
    comparisons = verdict.get("comparisons", {})
    steps = {n: _interval(_comparison(comparisons, n), n) for n in STEPS}
    volume = _interval(_comparison(comparisons, VOLUME), VOLUME)
    retention = {}
    for j, language in RETAINED.items():
        name = f"retention{j}"
        stratum = _comparison(comparisons, name).get("strata", {}).get(f"family_set:{language}")
        if stratum is None:
            raise Incomplete(f"{name}: stratum family_set:{language} is missing")
        retention[language] = _interval(stratum, f"{name}/{language}")
    return {
        "H1": {"met": all(lo > 0 for lo, _ in steps.values()), "failed": [n for n, (lo, _) in steps.items() if not lo > 0],
               "intervals": steps},
        "H2": {"met": volume[0] > 0, "interval": volume},
        "H3": {"met": all(lo > -MARGIN for lo, _ in retention.values()),
               "failed": [lang for lang, (lo, _) in retention.items() if not lo > -MARGIN], "intervals": retention, "margin": MARGIN},
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
    for name, (lo, hi) in result["H1"]["intervals"].items():
        print(f"{name}: [{lo:+.4f}, {hi:+.4f}]")
    print(f"H1 {'MET' if result['H1']['met'] else 'NOT MET'} {result['H1']['failed'] or ''}")
    lo, hi = result["H2"]["interval"]
    print(f"H2 {'MET' if result['H2']['met'] else 'NOT MET'}: acquisition over volume [{lo:+.4f}, {hi:+.4f}]")
    for language, (lo, hi) in result["H3"]["intervals"].items():
        print(f"retention {language}: [{lo:+.4f}, {hi:+.4f}]")
    print(f"H3 {'MET' if result['H3']['met'] else 'NOT MET'} {result['H3']['failed'] or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
