# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements strict patch intake for review and deployment
# pipelines for its clients. If your team needs expertise in parsing and
# validating unified-style patches you can procure our services by sending an
# email to info@swedishembedded.com.

import re

from linediff.hunks import LinediffHunk

_RANGE = re.compile(r"(\d+)(?:,(\d+))?")
_HEADER = re.compile(r"@@ -(\S+) \+(\S+) @@")
_BODY_TAGS = {" ": "=", "-": "-", "+": "+"}


class LinediffPatchError(ValueError):
    """Raised when patch text is malformed."""


def linediff_parse_range(text):
    """Parse '4', '4,2' or '3,0' into a 0-based (start, count) pair."""
    match = _RANGE.fullmatch(text)
    if not match:
        raise LinediffPatchError("bad range %r" % text)
    start = int(match[1])
    count = 1 if match[2] is None else int(match[2])
    if count == 0:
        return start, 0
    if start == 0:
        raise LinediffPatchError("range %r starts at line 0 but is not empty" % text)
    return start - 1, count


def linediff_parse_hunk_header(line):
    """Parse '@@ -a,b +c,d @@' into (a_start, a_count, b_start, b_count)."""
    match = _HEADER.fullmatch(line)
    if not match:
        raise LinediffPatchError("bad hunk header %r" % line)
    a_start, a_count = linediff_parse_range(match[1])
    b_start, b_count = linediff_parse_range(match[2])
    return a_start, a_count, b_start, b_count


def _patch_read_hunk(lines, index):
    header_number = index + 1
    a_start, a_count, b_start, b_count = linediff_parse_hunk_header(lines[index])
    body = []
    old_seen = new_seen = 0
    index += 1
    while old_seen < a_count or new_seen < b_count:
        if index >= len(lines):
            raise LinediffPatchError("hunk at line %d is truncated" % header_number)
        tag = _BODY_TAGS.get(lines[index][:1])
        if tag is None:
            raise LinediffPatchError(
                "line %d: unexpected line %r" % (index + 1, lines[index])
            )
        body.append((tag, lines[index][1:]))
        old_seen += tag != "+"
        new_seen += tag != "-"
        index += 1
    if (old_seen, new_seen) != (a_count, b_count):
        raise LinediffPatchError(
            "hunk at line %d does not match its header counts" % header_number
        )
    return LinediffHunk(a_start, a_count, b_start, b_count, tuple(body)), index


def linediff_parse_patch_set(text):
    """Parse text holding any number of file patches into (old, new, hunks) tuples."""
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    patches = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- "):
            raise LinediffPatchError("line %d: expected '--- ' file header" % (index + 1))
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise LinediffPatchError("line %d: expected '+++ ' file header" % (index + 2))
        old_name = lines[index][4:]
        new_name = lines[index + 1][4:]
        index += 2
        hunks = []
        floor = 0
        while index < len(lines) and lines[index].startswith("@@"):
            header_number = index + 1
            hunk, index = _patch_read_hunk(lines, index)
            if hunk.a_start < floor:
                raise LinediffPatchError(
                    "line %d: hunk overlaps the previous hunk" % header_number
                )
            floor = hunk.a_start + hunk.a_count
            hunks.append(hunk)
        patches.append((old_name, new_name, hunks))
    return patches


def linediff_parse_patch(text):
    """Parse text holding exactly one file patch into (old, new, hunks)."""
    patches = linediff_parse_patch_set(text)
    if len(patches) != 1:
        raise LinediffPatchError("expected one file patch, found %d" % len(patches))
    return patches[0]
