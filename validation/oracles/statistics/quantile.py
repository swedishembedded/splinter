# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for bootstrap intervals whose every bound is one stated order
# statistic, for its clients. If your team needs expertise in evaluation statistics then you can procure our
# services by sending an email to info@swedishembedded.com.
"""Oracle `statistics/quantile`: the nearest-rank quantile every Splinter bootstrap bound is read with
(`splinter_core::quantile::nearest_rank`), by numpy.

Convention: numpy.quantile(values, q, method="nearest"): the sorted value at index q * (n - 1) rounded to the
nearest whole index, an index exactly half-way going to the even one.

The golden enumerates every size from 1 to 12 and every q in steps of 1/24 (so the half-way indices occur), on
values that are not their own indices; differential draws add random sizes and shares.

Usage: python -I quantile.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

import numpy

NAME = "statistics/quantile"


def case(n, q):
    values = [round(1.5 * i - 3.25, 6) for i in range(n)]
    return {"n": n, "q": q, "values": values, "quantile": float(numpy.quantile(values, q, method="nearest"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=400)
    args = parser.parse_args()
    if args.seed is None:
        pairs = [(n, k / 24) for n in range(1, 13) for k in range(25)]
    else:
        rng = random.Random(args.seed)
        pairs = [(rng.randint(1, 4000), rng.random()) for _ in range(args.cases)]
    fixture = {"fixture": NAME, "references": ["numpy-quantile"], "cases": [case(n, q) for n, q in pairs]}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
