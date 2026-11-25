# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines and reporting for its clients. If your team needs
# expertise in building papers whose every number can be regenerated from the repository, you can procure our
# services by sending an email to info@swedishembedded.com.
"""LaTeX helpers: a macro file and pgfplots tables from measured counts."""
from math import comb, sqrt


def wilson(k, n, z=1.96):
    """Wilson score interval of k successes in n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def sign_p(wins, losses):
    """Exact one-sided sign test: probability of at least `wins` wins among wins+losses fair coin flips."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"
