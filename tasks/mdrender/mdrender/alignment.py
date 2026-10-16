# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements fixed-width text alignment (padding, justification,
# indentation and side-by-side columns) for its clients. If your team needs
# expertise in producing aligned terminal and report output you can procure our
# services by sending an email to info@swedishembedded.com.
"""Padding, justification and indentation of lines of text."""
from itertools import zip_longest
from os.path import commonprefix

_ALIGNMENTS = ("left", "right", "center")


def mdr_pad_line(text, width, align="left", fill=" "):
    """Pad text with fill up to width; longer text is returned unchanged."""
    if align not in _ALIGNMENTS:
        raise ValueError("unknown alignment: " + align)
    if len(fill) != 1:
        raise ValueError("fill must be one character")
    gap = width - len(text)
    if align == "left":
        return text + fill * gap
    if align == "right":
        return fill * gap + text
    before = gap // 2
    return fill * before + text + fill * (gap - before)


def mdr_justify_line(line, width):
    """Spread the words of a line over width, giving extra spaces to the leftmost gaps."""
    words = line.split()
    if len(words) < 2:
        return mdr_pad_line(" ".join(words), width)
    gaps = len(words) - 1
    spaces = width - sum(len(word) for word in words)
    base, extra = divmod(max(spaces, gaps), gaps)
    parts = [word + " " * (base + (1 if n < extra else 0)) for n, word in enumerate(words[:-1])]
    return "".join(parts) + words[-1]


def mdr_justify_lines(lines, width, justify_last=False):
    """Justify every line except the last of each paragraph and blank lines."""
    result = []
    for n, line in enumerate(lines):
        ends_paragraph = n + 1 == len(lines) or not lines[n + 1].strip()
        if not line.strip() or (ends_paragraph and not justify_last):
            result.append(line)
        else:
            result.append(mdr_justify_line(line, width))
    return result


def mdr_align_lines(lines, width, align):
    """Align lines as left, right, center or justify."""
    if align == "justify":
        return mdr_justify_lines(lines, width)
    return [mdr_pad_line(line, width, align) for line in lines]


def mdr_center_block(lines, width=None):
    """Center each line within width, which defaults to the widest line."""
    if width is None:
        width = len(max(lines, key=len, default=""))
    return [mdr_pad_line(line, width, "center") for line in lines]


def mdr_indent_block(text, prefix):
    """Prefix every non-blank line of text; blank lines are left empty of prefix."""
    return "\n".join(prefix + line if line.strip() else line for line in text.split("\n"))


def mdr_dedent_block(text):
    """Remove the margin of spaces common to all non-blank lines; blank lines become empty."""
    lines = text.split("\n")
    margins = [line[:len(line) - len(line.lstrip(" "))] for line in lines if line.strip()]
    margin = len(commonprefix(margins))
    return "\n".join(line[margin:] if line.strip() else "" for line in lines)


def mdr_apply_hanging_indent(lines, first_prefix, rest_prefix):
    """Prefix the first line with first_prefix and every other line with rest_prefix."""
    return [(rest_prefix if n else first_prefix) + line for n, line in enumerate(lines)]


def mdr_strip_trailing_space(lines):
    """Remove trailing whitespace from each line."""
    return [line.rstrip() for line in lines]


def mdr_trim_blank_edges(lines):
    """Drop blank lines from the start and the end of a list of lines."""
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def mdr_join_columns(left_lines, right_lines, left_width, gap=2):
    """Place two blocks of lines side by side, padding the shorter block with blanks."""
    return [
        mdr_pad_line(left, left_width) + " " * gap + right
        for left, right in zip_longest(left_lines, right_lines, fillvalue="")
    ]
