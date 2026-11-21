#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements measurements whose arithmetic is checked
# against exact references, for its clients. If your team needs expertise in
# clustered evaluation statistics and the learning signal of agent rollouts,
# you can procure our services by sending an email to
# info@swedishembedded.com.
"""Write conformance.json, the golden values crates/eval/tests/it/golden.rs
(and crates/pipelines' family-clustered interval specs) hold splinter to.

Every value is exact rational arithmetic from the definitions (fractions,
math.comb), never from the code under test:

* family-clustered percentile bootstrap: K families are drawn with
  replacement, and the statistic is the mean over every item of the families
  drawn (a ratio of sums: a large family weighs what its items weigh). The
  counts of each family in a draw are multinomial(K; 1/K, ..., 1/K), so the
  bootstrap distribution is enumerated exactly. Beside it, the exact
  quantiles of an item-level resample (the defect a clustered interval must
  not have) and whether they differ enough to be caught.
* learning signal: LS = V / mean cost, V = (n s2 + k v0) / (n + k), s2 the
  population variance of the rewards, v0 = a b / ((a + b)(a + b + 1)) with
  a = 1 + sum clamp(r), b = 1 + n - sum clamp(r), k = 2.

Run: python3 -I crates/eval/tests/fixtures/conformance/generate_conformance.py
(standard library only; rewrites conformance.json next to this script).
"""
import json
import math
import os
from fractions import Fraction

LEVEL = Fraction(95, 100)
PRIOR_ATTEMPTS = 2


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def cdf_of(pmf):
    out, running = [], Fraction(0)
    for key in sorted(pmf):
        running += pmf[key]
        out.append((key, running))
    return out


def quantile(cdf, q):
    return next(x for x, f in cdf if f >= q)


def clustered_case(name, families):
    k = len(families)
    sums = [sum(Fraction(v) for v in f) for f in families]
    sizes = [len(f) for f in families]
    pmf = {}
    for counts in compositions(k, k):
        coef = math.factorial(k)
        for c in counts:
            coef //= math.factorial(c)
        p = Fraction(coef, k**k)
        n = sum(c * s for c, s in zip(counts, sizes))
        if n == 0:
            continue
        ratio = sum(c * s for c, s in zip(counts, sums)) / n
        pmf[ratio] = pmf.get(ratio, Fraction(0)) + p
    cdf = cdf_of(pmf)
    tail = (1 - LEVEL) / 2
    low, high = quantile(cdf, tail), quantile(cdf, 1 - tail)

    items = [Fraction(v) for f in families for v in f]
    values = sorted(set(items))
    n = len(items)
    item_pmf = {}
    for counts in compositions(n, len(values)):
        coef = math.factorial(n)
        for c in counts:
            coef //= math.factorial(c)
        p = Fraction(coef, n**n)
        for c, v in zip(counts, values):
            p *= items.count(v) ** c
        mean = sum(c * v for c, v in zip(counts, values)) / n
        item_pmf[mean] = item_pmf.get(mean, Fraction(0)) + p
    item_cdf = cdf_of(item_pmf)
    item_low, item_high = quantile(item_cdf, tail), quantile(item_cdf, 1 - tail)
    # An item-level resample is caught when one of its bounds is not a
    # quantile of the clustered distribution within the tests' tolerance
    # (0.025 of CDF mass, five standard errors at 1,000 resamples).
    def off(bound, q):
        at = sum(p for x, p in pmf.items() if x <= bound)
        under = sum(p for x, p in pmf.items() if x < bound)
        return bound not in pmf or at + Fraction(25, 1000) < q or under - Fraction(25, 1000) > q

    return {
        "name": name,
        "families": [[float(v) for v in f] for f in families],
        "level": float(LEVEL),
        "mean": float(sum(sums) / sum(sizes)),
        "cdf": [[float(x), float(f)] for x, f in cdf],
        "low": float(low),
        "high": float(high),
        "item_level_low": float(item_low),
        "item_level_high": float(item_high),
        "item_level_detectable": off(item_low, tail) or off(item_high, 1 - tail),
    }


def signal_case(rewards, costs):
    n = len(rewards)
    r = [Fraction(x).limit_denominator(10**6) for x in rewards]
    mean = sum(r) / n if n else None
    s2 = sum((x - mean) ** 2 for x in r) / n if n else None
    successes = sum(min(max(x, Fraction(0)), Fraction(1)) for x in r)
    a, b = 1 + successes, 1 + n - successes
    v0 = a * b / ((a + b) * (a + b + 1))
    shrunk = (n * (s2 or 0) + PRIOR_ATTEMPTS * v0) / (n + PRIOR_ATTEMPTS)
    counted = [c for c in costs if c is not None]
    mean_cost = Fraction(sum(counted), len(counted)) if counted else None
    informative = n >= 2 and max(r) != min(r)
    return {
        "rewards": rewards,
        "costs": costs,
        "mean_reward": None if mean is None else float(mean),
        "variance": None if s2 is None else float(s2),
        "shrunk_variance": float(shrunk),
        "mean_cost": None if mean_cost is None else float(mean_cost),
        "learning_signal": None if mean_cost is None else float(shrunk / mean_cost),
        "informative": informative,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fixture = {
        "provenance": (
            "crates/eval/tests/fixtures/conformance/generate_conformance.py: exact "
            "rational arithmetic from the definitions (fractions, math.comb); "
            "independent of splinter's code"
        ),
        "clustered_bootstrap": [
            # Pass/fail attempts, every family all-pass or all-fail: tasks of
            # one family agree, so an item-level resample is far too narrow.
            clustered_case(
                "homogeneous-families",
                [[1] * 3, [0] * 1, [1] * 4, [0] * 2, [1] * 5, [0] * 2, [0] * 3, [1] * 4],
            ),
            # Paired differences (candidate less baseline) of unequal families.
            clustered_case(
                "paired-differences",
                [
                    [1, 1, 0],
                    [0],
                    [1, 1, 1, 0],
                    [-1, 0],
                    [1, 0, 0, 1, 1],
                    [0, 0],
                    [-1, -1, 0],
                    [1, 1, 1, 1],
                ],
            ),
        ],
        "learning_signal": [
            signal_case([], []),
            signal_case([1.0], [1000]),
            signal_case([1.0, 0.0, 0.0, 0.0], [1000, 3000, 2000, 2000]),
            signal_case([0.0, 0.0, 0.0, 0.0], [500, 500, 500, 500]),
            signal_case([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [100] * 8),
            signal_case([0.25, 0.5, 1.0, 0.0, 0.0], [None, 400, 600, None, 800]),
            signal_case([-0.5, 1.0, 0.2], [None, None, None]),
        ],
    }
    assert fixture["clustered_bootstrap"][0]["item_level_detectable"]
    with open(os.path.join(here, "conformance.json"), "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
