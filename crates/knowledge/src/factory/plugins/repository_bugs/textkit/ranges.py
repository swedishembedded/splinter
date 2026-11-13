# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_length(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))
