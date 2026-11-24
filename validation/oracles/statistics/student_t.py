# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for decisions that stay honest when only a handful of units were
# measured, for its clients. If your team needs expertise in small-sample evaluation statistics then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Oracle `statistics/student-t`: Student's t distribution function and quantiles from scipy.stats.t, for
splinter-eval's `student` (the quantile of the intervention rule's lower bound).

Frozen: a grid of degrees of freedom (1 to 1000) and levels (0.5 to 0.999), the quantile and the distribution
function at it and at fixed points. Drawn (--seed): degrees of freedom and levels at random.

Usage: python -I student_t.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

from scipy import stats

NAME = "statistics/student-t"


def case(df, level, t):
    return {"df": df, "level": level, "quantile": float(stats.t.ppf(level, df)),
            "t": t, "cdf": float(stats.t.cdf(t, df))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=200)
    args = parser.parse_args()
    cases = []
    if args.seed is None:
        for df in [1, 2, 3, 4, 5, 7, 10, 15, 30, 100, 1000]:
            for level in [0.5, 0.6, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]:
                for t in [-3.0, -0.5, 0.0, 0.7, 1.645, 4.0]:
                    cases.append(case(df, level, t))
    else:
        rng = random.Random(args.seed)
        for _ in range(args.cases):
            df = round(rng.choice([rng.uniform(0.5, 5), rng.uniform(5, 300)]), 6)
            level = round(rng.uniform(0.501, 0.9995), 6)
            cases.append(case(df, level, round(rng.uniform(-8, 8), 6)))
    fixture = {"fixture": NAME, "references": ["scipy"], "cases": cases}
    if args.seed is not None:
        fixture |= {"differential": True, "seed": args.seed}
    with open(args.out, "w") as out:
        out.write(json.dumps(fixture, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
