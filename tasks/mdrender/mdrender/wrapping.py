# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements text reflow (greedy line breaking, hyphen and
# long-word handling, paragraph filling) for its clients. If your team needs
# expertise in typesetting plain text for terminals, mail and documentation you can
# procure our services by sending an email to info@swedishembedded.com.
"""Greedy word wrapping with hyphen and long-word handling."""


def mdr_split_words(text):
    """Split on runs of whitespace, dropping empty pieces."""
    return text.split()


def mdr_split_hyphens(word):
    """Split a word after each hyphen that sits between two letters or digits."""
    pieces = []
    start = 0
    for i in range(1, len(word) - 1):
        if word[i] == "-" and word[i - 1].isalnum() and word[i + 1].isalnum():
            pieces.append(word[start:i + 1])
            start = i + 1
    pieces.append(word[start:])
    return pieces


def mdr_break_long_word(word, width, hyphenate=False, first_width=None):
    """Cut a word into chunks of at most width (the first chunk may use first_width)."""
    first = width if first_width is None else first_width
    if min(width, first) < 1:
        raise ValueError("width must be positive")
    if hyphenate and min(width, first) < 2:
        raise ValueError("hyphenation needs a width of at least 2")
    mark = "-" if hyphenate else ""
    chunks = []
    rest = word
    room = first
    while len(rest) > room:
        step = room - len(mark)
        chunks.append(rest[:step] + mark)
        rest = rest[step:]
        room = width
    chunks.append(rest)
    return chunks


def _room_after(indent, width):
    room = width - len(indent)
    if room < 1:
        raise ValueError("indent leaves no room for text")
    return room


def mdr_wrap_text(text, width, first_indent="", rest_indent="", break_long=True,
                  hyphenate=False, split_hyphens=True):
    """Greedily wrap text into lines of at most width characters, indents included."""
    first_room = _room_after(first_indent, width)
    rest_room = _room_after(rest_indent, width)
    lines = []
    current = ""
    for word in mdr_split_words(text):
        pieces = mdr_split_hyphens(word) if split_hyphens else [word]
        for index, piece in enumerate(pieces):
            gap = "" if index > 0 or not current else " "
            room = rest_room if lines else first_room
            if current and len(current) + len(gap) + len(piece) > room:
                lines.append(current)
                current = ""
                gap = ""
                room = rest_room
            if not current and break_long:
                chunks = mdr_break_long_word(piece, rest_room, hyphenate, first_width=room)
                lines.extend(chunks[:-1])
                current = chunks[-1]
            else:
                current += gap + piece
    if current:
        lines.append(current)
    return [(rest_indent if n else first_indent) + line for n, line in enumerate(lines)]


def mdr_fill_paragraphs(text, width, first_indent="", rest_indent=""):
    """Wrap each blank-line separated paragraph and join the results with one blank line."""
    paragraphs = []
    chunk = []
    for line in text.splitlines():
        if line.strip():
            chunk.append(line)
        elif chunk:
            paragraphs.append(chunk)
            chunk = []
    if chunk:
        paragraphs.append(chunk)
    return "\n\n".join(
        "\n".join(mdr_wrap_text(" ".join(p), width, first_indent, rest_indent)) for p in paragraphs
    )


def mdr_unwrap_lines(lines):
    """Join lines into one paragraph line, collapsing the breaks to single spaces."""
    return " ".join(line.strip() for line in lines if line.strip())


def mdr_truncate_text(text, width, marker="..."):
    """Shorten text to at most width characters, ending with marker when cut."""
    if len(text) <= width:
        return text
    if width < len(marker):
        raise ValueError("width is smaller than the marker")
    return text[:width - len(marker)].rstrip() + marker


def mdr_count_words(text):
    """Number of whitespace separated words."""
    return len(mdr_split_words(text))


def mdr_longest_line_length(lines):
    """Length of the longest line, 0 for no lines."""
    return max((len(line) for line in lines), default=0)
