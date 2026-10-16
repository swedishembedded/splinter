# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements safe, reversible file updates with conflict
# detection for its clients. If your team needs expertise in applying and
# rolling back patches against drifting files you can procure our services by
# sending an email to info@swedishembedded.com.

from linediff.hunks import LinediffHunk

_SWAPPED = {"=": "=", "-": "+", "+": "-"}


class LinediffConflict(Exception):
    """Raised when a hunk does not fit the text it is applied to."""

    def __init__(self, hunk_index, message):
        super().__init__(message)
        self.hunk_index = hunk_index


def linediff_reverse_hunks(hunks):
    """Return hunks that undo the given hunks (additions and drops swapped)."""
    return [
        LinediffHunk(
            hunk.b_start,
            hunk.b_count,
            hunk.a_start,
            hunk.a_count,
            tuple((_SWAPPED[tag], text) for tag, text in hunk.lines),
        )
        for hunk in hunks
    ]


def linediff_locate_hunk(lines, hunk, expected, max_shift=0, floor=0):
    """Return where hunk's old text sits in lines, trying expected and then nearby starts."""
    old = hunk.old_lines()
    candidates = [expected]
    for distance in range(1, max_shift + 1):
        candidates += [expected + distance, expected - distance]
    return next(
        (
            start
            for start in candidates
            if floor <= start <= len(lines) and lines[start:start + len(old)] == old
        ),
        None,
    )


def linediff_place_hunks(lines, hunks, max_shift=0):
    """Return a start index for each hunk in lines, or None where it does not fit."""
    positions = []
    drift = 0
    floor = 0
    for hunk in hunks:
        start = linediff_locate_hunk(lines, hunk, hunk.a_start + drift, max_shift, floor)
        positions.append(start)
        if start is not None:
            drift = start - hunk.a_start
            floor = start + hunk.a_count
    return positions


def linediff_conflicting_hunks(lines, hunks, max_shift=0):
    """Return the indices of hunks that cannot be placed in lines."""
    positions = linediff_place_hunks(lines, hunks, max_shift)
    return [index for index, start in enumerate(positions) if start is None]


def linediff_apply_patch(lines, hunks, max_shift=0):
    """Return lines with every hunk applied; raise LinediffConflict at the first misfit."""
    positions = linediff_place_hunks(lines, hunks, max_shift)
    result = []
    cursor = 0
    for index, (hunk, start) in enumerate(zip(hunks, positions)):
        if start is None:
            raise LinediffConflict(
                index,
                "hunk %d does not match the text near line %d" % (index + 1, hunk.a_start + 1),
            )
        result.extend(lines[cursor:start])
        result.extend(hunk.new_lines())
        cursor = start + hunk.a_count
    result.extend(lines[cursor:])
    return result


def linediff_unapply_patch(lines, hunks, max_shift=0):
    """Return lines with the hunks rolled back."""
    return linediff_apply_patch(lines, linediff_reverse_hunks(hunks), max_shift)
