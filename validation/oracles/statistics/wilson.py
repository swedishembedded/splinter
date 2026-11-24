# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for honest intervals on small counts, for its clients. If your team needs
# expertise in evaluation statistics then you can procure our services by sending an email to
# info@swedishembedded.com.
"""Oracle `statistics/wilson`: the two-sided 95% Wilson score interval by statsmodels, for splinter-eval's
`pilot::Share::wilson` (the one implementation; the environment band reads it).

Convention: statsmodels.stats.proportion.proportion_confint(count, nobs, alpha=0.05, method="wilson"), its ends
clamped to [0, 1] as Splinter's are (statsmodels' own Wilson ends already lie inside).

Usage: python -I wilson.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

from statsmodels.stats.proportion import proportion_confint

NAME = "statistics/wilson"


def case(successes, total):
    low, high = proportion_confint(successes, total, alpha=0.05, method="wilson")
    return {"successes": successes, "total": total, "low": max(0.0, float(low)), "high": min(1.0, float(high))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=300)
    args = parser.parse_args()
    if args.seed is None:
        pairs = [(s, n) for n in list(range(1, 21)) + [50, 100, 1000] for s in sorted({0, 1, n // 3, n // 2, n - 1, n})]
    else:
        rng = random.Random(args.seed)
        pairs = []
        for _ in range(args.cases):
            n = rng.randint(1, 5000)
            pairs.append((rng.randint(0, n), n))
    fixture = {"fixture": NAME, "references": ["statsmodels"], "cases": [case(s, n) for s, n in pairs]}
    if args.seed is not None:
        fixture |= {"differential": True, "seed": args.seed}
    with open(args.out, "w") as out:
        out.write(json.dumps(fixture, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
