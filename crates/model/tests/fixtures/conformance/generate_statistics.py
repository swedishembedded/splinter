#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluation statistics whose numbers are
# checked against exact references, for its clients. If your team needs
# expertise in paired significance tests and bootstrap intervals for model
# releases, you can procure our services by sending an email to
# info@swedishembedded.com.
"""Write statistics.json, the golden values crates/model/tests/it/statistics_conformance.rs
holds splinter_model::stats to.

Every value is exact rational arithmetic from the definitions (Python's
fractions and math.comb), never from the code under test:

* sign test: n = wins + losses discordant pairs (ties dropped), one-sided
  p = P(Binomial(n, 1/2) >= wins); the two-sided p (twice the smaller tail,
  capped at 1) is written beside it so a test can show the two differ.
* percentile bootstrap of a mean: resampling n items with replacement makes
  the counts of each distinct value multinomial, so the bootstrap
  distribution of the mean is known exactly. Each case writes that
  distribution's CDF on its lattice (step 1/n), for the paired resample (one
  draw takes an item's two verdicts together) and for the unpaired one (the
  two arms resampled independently), which a correct paired interval must
  not match.

Run: python3 -I crates/model/tests/fixtures/conformance/generate_statistics.py
(standard library only; rewrites statistics.json next to this script).
"""
import json
import math
import os
from fractions import Fraction

LEVEL = 0.95


def sign_case(wins, losses, ties):
    n = wins + losses
    if n == 0:
        upper = lower = Fraction(1)
    else:
        total = 2**n
        upper = Fraction(sum(math.comb(n, i) for i in range(wins, n + 1)), total)
        lower = Fraction(sum(math.comb(n, i) for i in range(0, wins + 1)), total)
    two = min(Fraction(1), 2 * min(upper, lower))
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "one_sided_p": float(upper),
        "two_sided_p": float(two),
    }


def pass_at_k_case(n, c, k):
    """The unbiased pass@k of one task, 1 - C(n - c, k) / C(n, k), exact."""
    value = 1 - Fraction(math.comb(n - c, k), math.comb(n, k))
    return {"n": n, "passed": c, "k": k, "pass_at_k": float(value),
            # The biased "any of the first k" reading a defect would give: 1 if
            # any of the n passed, whatever k.
            "any_of_n": float(c > 0)}


def holm_case(p):
    """Holm's step-down adjustment of p, in exact rationals."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted, running = [None] * m, Fraction(0)
    for rank, i in enumerate(order):
        running = max(running, min(Fraction(1), (m - rank) * p[i]))
        adjusted[i] = running
    return {"p": [float(x) for x in p], "holm": [float(x) for x in adjusted]}


def multinomial_mean_pmf(n, counts, values):
    """PMF of the mean of n draws with replacement from items whose distinct
    `values` occur `counts` times, keyed by the numerator over n."""
    probs = [Fraction(c, n) for c in counts]
    pmf = {}

    def walk(at, left, coef, prob, total):
        if at == len(values) - 1:
            key = total + left * values[at]
            p = coef * prob * probs[at] ** left
            pmf[key] = pmf.get(key, Fraction(0)) + p
            return
        for k in range(left + 1):
            walk(
                at + 1,
                left - k,
                coef * math.comb(left, k),
                prob * probs[at] ** k,
                total + k * values[at],
            )

    walk(0, n, 1, Fraction(1), 0)
    return pmf


def binomial_pmf(n, c):
    p = Fraction(c, n)
    return {k: math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)}


def cdf_lattice(pmf, n):
    """[(mean, CDF at mean)] over the support, means as floats over n."""
    out, running = [], Fraction(0)
    for key in sorted(pmf):
        running += pmf[key]
        out.append([key / n, float(running)])
    return out


def quantile(cdf, q):
    return next(x for x, f in cdf if f >= q)


def paired_case(name, both_right, candidate_only, baseline_only, both_wrong):
    n = both_right + candidate_only + baseline_only + both_wrong
    paired = multinomial_mean_pmf(
        n, [candidate_only, baseline_only, both_right + both_wrong], [1, -1, 0]
    )
    cand = binomial_pmf(n, both_right + candidate_only)
    base = binomial_pmf(n, both_right + baseline_only)
    unpaired = {}
    for x, px in cand.items():
        for y, py in base.items():
            unpaired[x - y] = unpaired.get(x - y, Fraction(0)) + px * py
    tail = (1 - LEVEL) / 2
    paired_cdf = cdf_lattice(paired, n)
    unpaired_cdf = cdf_lattice(unpaired, n)
    case = {
        "name": name,
        "both_right": both_right,
        "candidate_only": candidate_only,
        "baseline_only": baseline_only,
        "both_wrong": both_wrong,
        "level": LEVEL,
        "mean": (candidate_only - baseline_only) / n,
        "step": 1 / n,
        "paired_cdf": paired_cdf,
        "paired_low": quantile(paired_cdf, tail),
        "paired_high": quantile(paired_cdf, 1 - tail),
        "unpaired_low": quantile(unpaired_cdf, tail),
        "unpaired_high": quantile(unpaired_cdf, 1 - tail),
    }
    # Whether an unpaired resample is caught on this case: its bounds lie
    # more than one lattice step (the test's tolerance) outside the paired
    # ones. Concordant arms make the paired interval much the narrower.
    case["unpaired_detectable"] = (
        case["unpaired_low"] < case["paired_low"] - 1.5 / n
        or case["unpaired_high"] > case["paired_high"] + 1.5 / n
    )
    return case


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fixture = {
        "provenance": (
            "crates/model/tests/fixtures/conformance/generate_statistics.py: exact "
            "rational arithmetic from the definitions (fractions, math.comb); "
            "independent of splinter's code"
        ),
        "sign_test": [
            sign_case(*c)
            for c in [
                (0, 0, 5),
                (5, 0, 2),
                (3, 1, 0),
                (1, 3, 0),
                (7, 3, 4),
                (13, 5, 0),
                (20, 20, 3),
                (60, 40, 0),
                (1, 9, 0),
                (250, 200, 10),
                (11, 3, 0),
            ]
        ],
        "pass_at_k": [
            pass_at_k_case(n, c, k)
            for n, c, k in [
                (1, 0, 1), (1, 1, 1), (4, 1, 1), (4, 1, 2), (4, 1, 4), (8, 3, 2), (8, 3, 5),
                (8, 3, 6), (16, 1, 8), (16, 4, 4), (16, 16, 16), (16, 0, 16), (5, 2, 5),
                (100, 7, 10), (200, 1, 199),
            ]
        ],
        "holm": [
            holm_case([Fraction(x) for x in ps])
            for ps in [
                ["1/100", "4/100", "3/100"],
                ["1/50", "1/50", "1/2", "3/10", "9/10"],
                ["3/5", "1/1000", "1/5", "7/100", "1/40"],
                ["1"],
            ]
        ],
        "paired_bootstrap": [
            paired_case("gain-over-40-items", 16, 9, 3, 12),
            paired_case("no-gain-over-40-items", 10, 6, 6, 18),
            paired_case("fourteen-tasks", 3, 5, 1, 5),
            # Fine enough that the 2.5% and 5% quantiles lie lattice steps
            # apart: a bound read at the wrong share is caught.
            paired_case("gain-over-200-items", 80, 45, 15, 60),
        ],
    }
    assert fixture["paired_bootstrap"][0]["unpaired_detectable"], "no case catches an unpaired resample"
    fine = fixture["paired_bootstrap"][-1]
    cdf = [(x, f) for x, f in fine["paired_cdf"]]
    at = lambda q: next(x for x, f in cdf if f >= q)
    # The bounds the tests accept for the 2.5% share (its quantile within
    # their CDF tolerance of 0.006) and for the 5% share do not overlap, nor
    # do those for 95% and 97.5%.
    assert at(0.05 - 0.006) > at(0.025 + 0.006), "the fine case cannot tell 2.5% from 5%"
    assert at(0.975 - 0.006) > at(0.95 + 0.006), "the fine case cannot tell 95% from 97.5%"
    with open(os.path.join(here, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
