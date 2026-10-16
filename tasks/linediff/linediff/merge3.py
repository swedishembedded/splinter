# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements three-way reconciliation of diverging text
# and configuration files for its clients. If your team needs expertise in
# merging parallel edits and surfacing true conflicts you can procure our
# services by sending an email to info@swedishembedded.com.

from linediff.editscript import linediff_change_blocks, linediff_edit_script


def linediff_changed_spans(base, other):
    """Return (start, end, replacement) for each base range that other rewrote."""
    blocks = linediff_change_blocks(linediff_edit_script(base, other))
    return [(a0, a1, other[b0:b1]) for a0, a1, b0, b1 in blocks]


def _merge_clusters(ours_spans, theirs_spans):
    spans = sorted(ours_spans + theirs_spans, key=lambda span: (span[0], span[1]))
    clusters = []
    for start, end, _ in spans:
        if clusters and start <= clusters[-1][1]:
            clusters[-1][1] = max(clusters[-1][1], end)
        else:
            clusters.append([start, end])
    return clusters


def _side_version(base, start, end, spans):
    out = []
    cursor = start
    for span_start, span_end, replacement in spans:
        if start <= span_start and span_end <= end:
            out.extend(base[cursor:span_start])
            out.extend(replacement)
            cursor = span_end
    out.extend(base[cursor:end])
    return out


def linediff_merge_chunks(base, ours, theirs):
    """Merge into ("clean", lines) and ("conflict", ours_lines, theirs_lines) chunks."""
    ours_spans = linediff_changed_spans(base, ours)
    theirs_spans = linediff_changed_spans(base, theirs)
    chunks = []
    cursor = 0
    for start, end in _merge_clusters(ours_spans, theirs_spans):
        if cursor < start:
            chunks.append(("clean", base[cursor:start]))
        original = base[start:end]
        mine = _side_version(base, start, end, ours_spans)
        yours = _side_version(base, start, end, theirs_spans)
        if mine == yours or yours == original:
            chunks.append(("clean", mine))
        elif mine == original:
            chunks.append(("clean", yours))
        else:
            chunks.append(("conflict", mine, yours))
        cursor = end
    if cursor < len(base):
        chunks.append(("clean", base[cursor:]))
    return chunks


def linediff_has_conflicts(chunks):
    """Return True when any chunk is a conflict."""
    return any(chunk[0] == "conflict" for chunk in chunks)


def linediff_render_merge(chunks, ours_label="ours", theirs_label="theirs"):
    """Flatten chunks into lines, wrapping each conflict in marker lines."""
    lines = []
    for chunk in chunks:
        if chunk[0] == "clean":
            lines.extend(chunk[1])
        else:
            lines.append("<<<<<<< " + ours_label)
            lines.extend(chunk[1])
            lines.append("=======")
            lines.extend(chunk[2])
            lines.append(">>>>>>> " + theirs_label)
    return lines


def linediff_merge_three(base, ours, theirs, ours_label="ours", theirs_label="theirs"):
    """Return (lines, conflict_count) for a three-way merge with marker lines."""
    chunks = linediff_merge_chunks(base, ours, theirs)
    conflicts = sum(1 for chunk in chunks if chunk[0] == "conflict")
    return linediff_render_merge(chunks, ours_label, theirs_label), conflicts
