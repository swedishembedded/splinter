# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements block-level document parsing (headings, nested
# lists, quotes and fenced code) for its clients. If your team needs expertise in
# building structured document pipelines from lightweight markup you can procure
# our services by sending an email to info@swedishembedded.com.
"""Block parser for headings, paragraphs, lists, fenced code, quotes and rules."""
import re

from mdrender.inline import mdr_parse_inline

_HEADING = re.compile(r" {0,3}(#{1,6})(?: +(.*))?$")
_CLOSING_HASHES = re.compile(r"(?:^| +)#+$")
_RULE = re.compile(r" {0,3}([-*_])(?: *\1){2,} *$")
_FENCE = re.compile(r"( {0,3})(`{3,}|~{3,}) *(.*)$")
_QUOTE = re.compile(r" {0,3}> ?")
_LIST_MARKER = re.compile(r"( {0,3})(?:([-+*])|(\d{1,9})([.)]))( *)(.*)$")
_TAB_WIDTH = 4
_MAX_MARKER_GAP = 4


def mdr_match_atx_heading(line):
    """Return (level, text) for a line like '## Title ##', else None."""
    found = _HEADING.match(line)
    if not found:
        return None
    body = _CLOSING_HASHES.sub("", (found.group(2) or "").rstrip())
    return len(found.group(1)), body.strip()


def mdr_is_thematic_break(line):
    """True for three or more of -, * or _ (optionally spaced) alone on a line."""
    return _RULE.match(line) is not None


def mdr_match_code_fence(line):
    """Return (char, length, info, indent) for an opening ``` or ~~~ fence, else None."""
    found = _FENCE.match(line)
    if not found or (found.group(2)[0] == "`" and "`" in found.group(3)):
        return None
    return found.group(2)[0], len(found.group(2)), found.group(3).strip(), len(found.group(1))


def mdr_match_list_marker(line):
    """Describe a list item start: ordered, kind, number, offset of its content and text."""
    found = _LIST_MARKER.match(line)
    if not found:
        return None
    bullet, digits, delimiter, spaces, rest = found.group(2, 3, 4, 5, 6)
    if rest and not spaces:
        return None
    width = len(found.group(1)) + len(bullet or digits + delimiter)
    gap = len(spaces) if rest and len(spaces) <= _MAX_MARKER_GAP else 1
    return {
        "ordered": digits is not None,
        "kind": bullet or delimiter,
        "number": int(digits) if digits is not None else None,
        "offset": width + gap,
        "text": rest,
    }


def _leading_spaces(line):
    return len(line) - len(line.lstrip(" "))


def _expand_leading_tabs(line):
    body = line.lstrip(" \t")
    return line[:len(line) - len(body)].expandtabs(_TAB_WIDTH) + body


def _skip_blank_lines(lines, start):
    while start < len(lines) and not lines[start].strip():
        start += 1
    return start


def _starts_other_block(line):
    return bool(
        mdr_match_atx_heading(line) or mdr_is_thematic_break(line)
        or mdr_match_code_fence(line) or _QUOTE.match(line)
    )


def _starts_block(line):
    return _starts_other_block(line) or mdr_match_list_marker(line) is not None


def _interrupts_paragraph(line):
    marker = mdr_match_list_marker(line)
    list_start = bool(marker and marker["text"] and (not marker["ordered"] or marker["number"] == 1))
    return _starts_other_block(line) or list_start


def _closes_fence(line, char, length):
    return re.fullmatch(r" {0,3}" + re.escape(char) + "{%d,} *" % length, line) is not None


def _parse_fence(lines, start, fence):
    char, length, info, indent = fence
    body = []
    scan = start + 1
    while scan < len(lines) and not _closes_fence(lines[scan], char, length):
        cut = min(indent, _leading_spaces(lines[scan]))
        body.append(lines[scan][cut:])
        scan += 1
    return {"type": "code", "info": info, "text": "\n".join(body)}, scan + 1


def _parse_quote(lines, start):
    inner = []
    scan = start
    while scan < len(lines) and _QUOTE.match(lines[scan]):
        inner.append(_QUOTE.sub("", lines[scan], count=1))
        scan += 1
    return {"type": "quote", "children": _parse_lines(inner)}, scan


def _collect_item(lines, start, marker):
    item = [marker["text"]]
    offset = marker["offset"]
    scan = start + 1
    while scan < len(lines):
        line = lines[scan]
        if not line.strip():
            ahead = _skip_blank_lines(lines, scan)
            if ahead == len(lines) or _leading_spaces(lines[ahead]) < offset:
                break
            item.append("")
        elif _leading_spaces(line) >= offset:
            item.append(line[offset:])
        elif item[-1].strip() and not _starts_block(line.strip()):
            item.append(line.strip())
        else:
            break
        scan += 1
    return item, scan


def _same_list(marker, first):
    return marker is not None and marker["ordered"] == first["ordered"] and marker["kind"] == first["kind"]


def _parse_list(lines, start, first):
    items = []
    loose = False
    scan = start
    marker = first
    while True:
        item, scan = _collect_item(lines, scan, marker)
        loose = loose or "" in item[1:]
        items.append(_parse_lines(item))
        after = _skip_blank_lines(lines, scan)
        marker = mdr_match_list_marker(lines[after]) if after < len(lines) else None
        if not _same_list(marker, first) or mdr_is_thematic_break(lines[after]):
            break
        loose = loose or after > scan
        scan = after
    block = {"type": "list", "ordered": first["ordered"], "start": first["number"],
             "tight": not loose, "items": items}
    return block, scan


def _parse_paragraph(lines, start):
    body = [lines[start]]
    for line in lines[start + 1:]:
        if not line.strip() or _interrupts_paragraph(line):
            break
        body.append(line)
    text = "\n".join(line.lstrip() for line in body).rstrip()
    return {"type": "paragraph", "children": mdr_parse_inline(text)}, start + len(body)


def _parse_lines(lines):
    blocks = []
    scan = 0
    while scan < len(lines):
        line = lines[scan]
        heading = mdr_match_atx_heading(line)
        fence = mdr_match_code_fence(line)
        marker = mdr_match_list_marker(line)
        if not line.strip():
            scan += 1
        elif heading:
            blocks.append({"type": "heading", "level": heading[0], "children": mdr_parse_inline(heading[1])})
            scan += 1
        elif mdr_is_thematic_break(line):
            blocks.append({"type": "rule"})
            scan += 1
        elif fence:
            block, scan = _parse_fence(lines, scan, fence)
            blocks.append(block)
        elif _QUOTE.match(line):
            block, scan = _parse_quote(lines, scan)
            blocks.append(block)
        elif marker:
            block, scan = _parse_list(lines, scan, marker)
            blocks.append(block)
        else:
            block, scan = _parse_paragraph(lines, scan)
            blocks.append(block)
    return blocks


def mdr_parse_blocks(text):
    """Parse a document into a list of heading, paragraph, list, code, quote and rule blocks."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return _parse_lines([_expand_leading_tabs(line) for line in lines])


def mdr_walk_blocks(blocks):
    """Yield every block depth first, including blocks nested in quotes and list items."""
    for block in blocks:
        yield block
        if block["type"] == "quote":
            yield from mdr_walk_blocks(block["children"])
        elif block["type"] == "list":
            for item in block["items"]:
                yield from mdr_walk_blocks(item)


def mdr_count_blocks(blocks):
    """Count blocks by type, nested blocks included, in order of first appearance."""
    counts = {}
    for block in mdr_walk_blocks(blocks):
        counts[block["type"]] = counts.get(block["type"], 0) + 1
    return counts
