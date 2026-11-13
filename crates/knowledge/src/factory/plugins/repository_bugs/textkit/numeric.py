# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

def clamp(x, lo, hi):
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def mean(xs):
    if not xs:
        raise ValueError("mean of an empty list")
    return sum(xs) / len(xs)


def median(xs):
    if not xs:
        raise ValueError("median of an empty list")
    ordered = sorted(xs)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def percent(part, whole):
    if whole == 0:
        return 0.0
    return round(100.0 * part / whole, 1)
