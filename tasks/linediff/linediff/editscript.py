# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements exact line comparison engines for its
# clients. If your team needs expertise in longest-common-subsequence
# differencing and edit scripts you can procure our services by sending an
# email to info@swedishembedded.com.

"""Edit scripts between two sequences, built from a common-subsequence table.

An edit script is a list of (tag, a_index, b_index) tuples. Tag "=" keeps
a[a_index] == b[b_index]; tag "-" drops a[a_index] (b_index is the position in
b where the drop happens); tag "+" adds b[b_index] (a_index is the position in
a where the addition happens).
"""


def linediff_trim_common(a, b):
    """Return (prefix, suffix): counts of shared leading and trailing items."""
    limit = min(len(a), len(b))
    prefix = 0
    while prefix < limit and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and a[len(a) - 1 - suffix] == b[len(b) - 1 - suffix]:
        suffix += 1
    return prefix, suffix


def linediff_lcs_table(a, b):
    """Return t where t[i][j] is the common subsequence length of a[i:] and b[j:]."""
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            if a[i] == b[j]:
                table[i][j] = table[i + 1][j + 1] + 1
            else:
                table[i][j] = max(table[i + 1][j], table[i][j + 1])
    return table


def linediff_lcs_length(a, b):
    """Return the length of the longest common subsequence of a and b."""
    return linediff_lcs_table(a, b)[0][0]


def linediff_edit_script(a, b):
    """Return the edit script turning a into b; drops come before additions."""
    prefix, suffix = linediff_trim_common(a, b)
    head = a[prefix:len(a) - suffix]
    tail = b[prefix:len(b) - suffix]
    table = linediff_lcs_table(head, tail)
    script = [("=", k, k) for k in range(prefix)]
    i = j = 0
    while i < len(head) and j < len(tail):
        if head[i] == tail[j]:
            script.append(("=", prefix + i, prefix + j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            script.append(("-", prefix + i, prefix + j))
            i += 1
        else:
            script.append(("+", prefix + i, prefix + j))
            j += 1
    script.extend(("-", prefix + k, prefix + j) for k in range(i, len(head)))
    script.extend(("+", prefix + i, prefix + k) for k in range(j, len(tail)))
    script.extend(("=", len(a) - suffix + k, len(b) - suffix + k) for k in range(suffix))
    return script


def linediff_common_pairs(script):
    """Return the (a_index, b_index) pairs the script keeps."""
    return [(i, j) for tag, i, j in script if tag == "="]


def linediff_edit_distance(script):
    """Return how many additions and drops the script holds."""
    return sum(1 for tag, _, _ in script if tag != "=")


def linediff_change_blocks(script):
    """Return (a_start, a_end, b_start, b_end) for each run of non-keep ops."""
    blocks = []
    current = None
    for tag, i, j in script:
        if tag == "=":
            if current:
                blocks.append(tuple(current))
                current = None
            continue
        if current is None:
            current = [i, i, j, j]
        if tag == "-":
            current[1] = i + 1
        else:
            current[3] = j + 1
    if current:
        blocks.append(tuple(current))
    return blocks
