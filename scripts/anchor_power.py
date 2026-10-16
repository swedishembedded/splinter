#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements regression gates for model updates whose
# false-refusal and false-release rates are computed before the run, for its
# clients. If your team needs expertise in sizing acceptance bounds for
# evaluation suites, you can procure our services by sending an email to
# info@swedishembedded.com.
"""The error rates of the anchor check decided on the answers lost net of
those gained (`AnchorBound::NetLost(k)`).

Model: on a suite of n items a candidate's answers flip independently, a
share f of the n items from right to wrong (lost) and a share f from wrong to
right (gained), so lost and gained are independent Poisson(n f). A candidate
that is truly worse by `loss` answers loses Poisson(n f + loss) and gains
Poisson(n f). The check refuses when lost - gained > k.

  python3 scripts/anchor_power.py            # the table the preregistration states
  python3 scripts/anchor_power.py --n 200 --k 6 --flip 0.03 --loss 15

The same numbers are asserted, to the same precision, by
`crates/eval/src/gate/tests.rs`.
"""

import argparse
from math import exp

SUPPORT = 400


def pmf(mean):
    p = [exp(-mean)]
    for i in range(1, SUPPORT):
        p.append(p[-1] * mean / i)
    return p


def p_net_at_least(lost_mean, gained_mean, threshold):
    """P(lost - gained >= threshold), exact to double precision."""
    lost, gained = pmf(lost_mean), pmf(gained_mean)
    return sum(
        pl * pg
        for x, pl in enumerate(lost)
        for y, pg in enumerate(gained)
        if x - y >= threshold
    )


def false_refusal(n, k, flip):
    """P(refused) for a candidate with no true change."""
    return p_net_at_least(n * flip, n * flip, k + 1)


def released(n, k, flip, loss):
    """P(released) for a candidate truly `loss` answers worse."""
    return 1.0 - p_net_at_least(n * flip + loss, n * flip, k + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--flip", type=float)
    ap.add_argument("--loss", type=float, default=0.0)
    a = ap.parse_args()
    if a.n is not None:
        print(f"n={a.n} k={a.k} flip={a.flip}: "
              f"refused {false_refusal(a.n, a.k, a.flip):.4f}, "
              f"released at a true loss of {a.loss:g}: {released(a.n, a.k, a.flip, a.loss):.4f}")
        return
    for flip in (0.02, 0.03):
        print(f"n=200 flip={flip:.0%}: false refusal k=6 {false_refusal(200, 6, flip):.1%}, "
              f"k=4 {false_refusal(200, 4, flip):.1%}")
    for flip in (0.02, 0.03):
        print(f"n=40 flip={flip:.0%}: refused at zero allowed loss {false_refusal(40, 0, flip):.1%}")
    for loss in (15, 10):
        print(f"n=200 k=6 flip=3%: true loss {loss}: released {released(200, 6, 0.03, loss):.1%}")


if __name__ == "__main__":
    main()
