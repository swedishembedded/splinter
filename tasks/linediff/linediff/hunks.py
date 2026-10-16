# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements review-friendly change grouping for text
# comparison tools for its clients. If your team needs expertise in turning
# raw edit scripts into hunks with context you can procure our services by
# sending an email to info@swedishembedded.com.

from dataclasses import dataclass


@dataclass(frozen=True)
class LinediffHunk:
    """A run of changes with context; starts are 0-based, lines are (tag, text)."""

    a_start: int
    a_count: int
    b_start: int
    b_count: int
    lines: tuple

    def old_lines(self):
        """Return the text this hunk expects to find in the old side."""
        return [text for tag, text in self.lines if tag != "+"]

    def new_lines(self):
        """Return the text this hunk leaves in the new side."""
        return [text for tag, text in self.lines if tag != "-"]


def _hunk_check_context(context):
    if context < 0:
        raise ValueError("context must not be negative")


def _hunk_from_ops(ops, a, b):
    lines = tuple((tag, b[j] if tag == "+" else a[i]) for tag, i, j in ops)
    return LinediffHunk(
        ops[0][1],
        sum(1 for tag, _, _ in ops if tag != "+"),
        ops[0][2],
        sum(1 for tag, _, _ in ops if tag != "-"),
        lines,
    )


def linediff_group_hunks(script, a, b, context=3):
    """Group an edit script into hunks holding `context` kept lines around changes."""
    _hunk_check_context(context)
    changes = [k for k, (tag, _, _) in enumerate(script) if tag != "="]
    if not changes:
        return []
    spans = [[changes[0], changes[0]]]
    for k in changes[1:]:
        if k - spans[-1][1] - 1 <= 2 * context:
            spans[-1][1] = k
        else:
            spans.append([k, k])
    return [
        _hunk_from_ops(script[max(0, first - context):last + context + 1], a, b)
        for first, last in spans
    ]


def linediff_hunk_stats(hunks):
    """Return (added, removed) line totals over the hunks."""
    added = sum(1 for hunk in hunks for tag, _ in hunk.lines if tag == "+")
    removed = sum(1 for hunk in hunks for tag, _ in hunk.lines if tag == "-")
    return added, removed


def linediff_hunk_covering(hunks, a_line):
    """Return the index of the hunk whose old range holds a_line, or None."""
    for index, hunk in enumerate(hunks):
        if hunk.a_start <= a_line < hunk.a_start + hunk.a_count:
            return index
    return None


def linediff_trim_hunk_context(hunk, context):
    """Return the hunk with at most `context` kept lines before and after its changes."""
    _hunk_check_context(context)
    lines = hunk.lines
    lead = 0
    while lead < len(lines) and lines[lead][0] == "=":
        lead += 1
    trail = 0
    while trail < len(lines) - lead and lines[len(lines) - 1 - trail][0] == "=":
        trail += 1
    cut_head = max(0, lead - context)
    cut_tail = max(0, trail - context)
    return LinediffHunk(
        hunk.a_start + cut_head,
        hunk.a_count - cut_head - cut_tail,
        hunk.b_start + cut_head,
        hunk.b_count - cut_head - cut_tail,
        lines[cut_head:len(lines) - cut_tail],
    )
