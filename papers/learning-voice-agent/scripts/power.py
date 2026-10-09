#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the design of small paired evaluations whose
# size is chosen before the data exists, for its clients. If your team needs
# expertise in power analysis for listening tests, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Sample sizes for the paper's paired comparisons, exact and stdlib only.

    power.py sign --win 0.65            # blind preference: n for a one-sided sign test
    power.py sign --win 0.65 --n 100    # the power of n = 100
    power.py tost --sd 0.08 --margin 0.05   # equivalence of a mean paired difference

`sign`: a rater prefers the new take over the old with probability `win` (ties
dropped); the test is the exact one-sided binomial test of win > 1/2 at `alpha`.
`tost`: two one-sided t-like tests of a paired difference with standard
deviation `sd` against +/- `margin`, normal approximation, true difference 0.
"""

import argparse
from math import comb, erf, sqrt


def binom_tail(n, k, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def critical(n, alpha):
    """The smallest wins k whose one-sided exact p-value under 1/2 is <= alpha."""
    for k in range(n + 1):
        if binom_tail(n, k, 0.5) <= alpha:
            return k
    return n + 1


def sign_power(n, win, alpha):
    k = critical(n, alpha)
    return binom_tail(n, k, win) if k <= n else 0.0


def sign_n(win, alpha, power):
    n = 1
    while sign_power(n, win, alpha) < power:
        n += 1
        if n > 5000:
            raise SystemExit("no n up to 5000 reaches that power")
    return n


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def tost_power(n, sd, margin, alpha):
    """Power of TOST at true difference 0 (normal approximation)."""
    z = 1.6448536269514722 if alpha == 0.05 else None
    if z is None:
        raise SystemExit("tost supports alpha = 0.05")
    se = sd / sqrt(n)
    return max(0.0, 2 * phi(margin / se - z) - 1)


def tost_n(sd, margin, alpha, power):
    n = 2
    while tost_power(n, sd, margin, alpha) < power:
        n += 1
        if n > 100000:
            raise SystemExit("no n up to 100000 reaches that power")
    return n


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    s = sub.add_parser("sign")
    s.add_argument("--win", type=float, required=True)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--power", type=float, default=0.8)
    s.add_argument("--n", type=int)
    t = sub.add_parser("tost")
    t.add_argument("--sd", type=float, required=True)
    t.add_argument("--margin", type=float, required=True)
    t.add_argument("--alpha", type=float, default=0.05)
    t.add_argument("--power", type=float, default=0.8)
    t.add_argument("--n", type=int)
    a = ap.parse_args()
    if a.kind == "sign":
        if a.n:
            print(f"sign test, win={a.win}, n={a.n}, alpha={a.alpha}: power {sign_power(a.n, a.win, a.alpha):.3f} (critical wins {critical(a.n, a.alpha)})")
        else:
            n = sign_n(a.win, a.alpha, a.power)
            print(f"sign test, win={a.win}, alpha={a.alpha}, power {a.power}: n = {n} decisive pairs (critical wins {critical(n, a.alpha)})")
    else:
        if a.n:
            print(f"TOST, sd={a.sd}, margin={a.margin}, n={a.n}: power {tost_power(a.n, a.sd, a.margin, a.alpha):.3f}")
        else:
            print(f"TOST, sd={a.sd}, margin={a.margin}, alpha={a.alpha}, power {a.power}: n = {tost_n(a.sd, a.margin, a.alpha, a.power)}")


if __name__ == "__main__":
    main()
