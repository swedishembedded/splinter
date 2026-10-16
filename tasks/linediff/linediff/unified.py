# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements readable change reports for text and
# configuration files for its clients. If your team needs expertise in
# unified-style patch rendering you can procure our services by sending an
# email to info@swedishembedded.com.

from linediff.hunks import linediff_hunk_stats

_PREFIX = {"=": " ", "-": "-", "+": "+"}


def linediff_format_range(start, count):
    """Format a 0-based start and a count as a unified range such as '4,2'."""
    if count == 0:
        return "%d,0" % start
    if count == 1:
        return "%d" % (start + 1)
    return "%d,%d" % (start + 1, count)


def linediff_render_hunk_header(hunk):
    """Return the '@@ -old +new @@' line of a hunk."""
    old = linediff_format_range(hunk.a_start, hunk.a_count)
    new = linediff_format_range(hunk.b_start, hunk.b_count)
    return "@@ -%s +%s @@" % (old, new)


def linediff_render_hunk(hunk):
    """Return the header and prefixed body lines of one hunk."""
    body = [_PREFIX[tag] + text for tag, text in hunk.lines]
    return [linediff_render_hunk_header(hunk)] + body


def linediff_render_patch(hunks, old_name, new_name):
    """Return patch text with file headers; no hunks render as an empty string."""
    if not hunks:
        return ""
    lines = ["--- " + old_name, "+++ " + new_name]
    for hunk in hunks:
        lines.extend(linediff_render_hunk(hunk))
    return "\n".join(lines) + "\n"


def _unified_count(count, noun):
    return "%d %s" % (count, noun if count == 1 else noun + "s")


def linediff_render_stat_line(hunks):
    """Return a one-line summary such as '2 hunks, 3 insertions(+), 1 deletion(-)'."""
    added, removed = linediff_hunk_stats(hunks)
    return "%s, %s(+), %s(-)" % (
        _unified_count(len(hunks), "hunk"),
        _unified_count(added, "insertion"),
        _unified_count(removed, "deletion"),
    )


def linediff_render_numbered_hunk(hunk, width=4):
    """Return hunk body lines with right-aligned 1-based old and new line numbers."""
    old_number = hunk.a_start
    new_number = hunk.b_start
    rows = []
    for tag, text in hunk.lines:
        old_cell = new_cell = " " * width
        if tag != "+":
            old_number += 1
            old_cell = str(old_number).rjust(width)
        if tag != "-":
            new_number += 1
            new_cell = str(new_number).rjust(width)
        rows.append("%s %s %s%s" % (old_cell, new_cell, _PREFIX[tag], text))
    return rows
