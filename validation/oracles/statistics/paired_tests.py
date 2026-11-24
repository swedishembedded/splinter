# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for checking a learning system's release statistics against
# established statistical libraries, for its clients. If your team needs expertise in evaluation statistics then
# you can procure our services by sending an email to info@swedishembedded.com.
"""Oracle `statistics/paired-tests`: the paired sign test, Holm's adjustment and unbiased pass@k, computed by
scipy, statsmodels and the HumanEval estimator, for splinter-model's `stats` (one implementation of each).

Conventions (stated, and checked here before anything is written):
* sign test: n = wins + losses discordant pairs (ties dropped), X ~ Binomial(n, 1/2); one-sided p = P(X >= wins),
  scipy.stats.binomtest(..., alternative='greater'); two-sided p = scipy's 'minlike' two-sided p, which at p = 1/2
  equals Splinter's doubled tail min(1, 2 min(P(X >= wins), P(X <= wins))). n = 0 is Splinter's convention (p = 1)
  and has no scipy counterpart, so it is not drawn here (the exact-arithmetic fixture holds it).
* Holm: statsmodels.stats.multitest.multipletests(p, method='holm')[1], adjusted p-values in input order.
* pass@k: 1 - prod(1 - k / arange(n - c + 1, n + 1)), 1.0 when n - c < k (arXiv:2107.03374, eq. 1), k <= n.

Usage: python -I paired_tests.py --out FILE [--seed N --cases M]   (no --seed: the frozen draw)
"""

import argparse
import json
import random

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

NAME = "statistics/paired-tests"


def doubled_tail(wins, n):
    upper = stats.binom.sf(wins - 1, n, 0.5)
    lower = stats.binom.cdf(wins, n, 0.5)
    return min(1.0, 2.0 * min(upper, lower))


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def draw(rng, cases):
    sign, holm, passk = [], [], []
    # Boundaries first: every split of small n, then random larger ones.
    small = [(w, n - w) for n in range(1, 13) for w in range(n + 1)]
    for wins, losses in small + [(rng.randint(0, 300), rng.randint(0, 300)) for _ in range(cases)]:
        n = wins + losses
        if n == 0:
            continue
        one = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue
        two = stats.binomtest(wins, n, 0.5, alternative="two-sided").pvalue
        assert abs(two - doubled_tail(wins, n)) <= 1e-12 * max(1.0, two), (wins, n, two)
        sign.append({"wins": wins, "losses": losses, "one_sided_p": float(one), "two_sided_p": float(two)})
    for _ in range(cases):
        m = rng.randint(1, 12)
        pool = [rng.choice([0.0, 1.0, 0.05, 0.01]) if rng.random() < 0.2 else rng.random() ** 3 for _ in range(m)]
        if m > 2 and rng.random() < 0.3:
            pool[1] = pool[0]  # ties
        holm.append({"p": pool, "adjusted": [float(x) for x in multipletests(pool, method="holm")[1]]})
    for _ in range(cases):
        n = rng.randint(1, 60)
        c = rng.randint(0, n)
        k = rng.randint(1, n)
        passk.append({"n": n, "c": c, "k": k, "pass_at_k": pass_at_k(n, c, k)})
    return [{"kind": "sign", **x} for x in sign] + [{"kind": "holm", **x} for x in holm] + \
        [{"kind": "pass_at_k", **x} for x in passk]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=150)
    args = parser.parse_args()
    seed = 20261124 if args.seed is None else args.seed
    fixture = {"fixture": NAME, "references": ["scipy", "statsmodels", "humaneval-pass-at-k"],
               "cases": draw(random.Random(seed), args.cases)}
    if args.seed is not None:
        fixture |= {"differential": True, "seed": seed}
    with open(args.out, "w") as out:
        out.write(json.dumps(fixture, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
