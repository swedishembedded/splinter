# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements inline markup parsing (emphasis, code spans, links
# and escapes) for its clients. If your team needs expertise in turning lightweight
# markup into structured documents you can procure our services by sending an email
# to info@swedishembedded.com.
"""Parser for the inline subset: emphasis, strong, code spans, links and escapes."""
import re
import string

_ESCAPABLE = frozenset(string.punctuation)
_BACKTICKS = re.compile(r"`+")
_TITLE = re.compile(r'"([^"]*)"')
_AUTOLINK = re.compile(r"<((?:https?://|mailto:)[^\s<>]+)>")
_MAX_DELIMITER_RUN = 3


def _run_length(text, start, char):
    end = start
    while end < len(text) and text[end] == char:
        end += 1
    return end - start


def _code_span(text, start):
    ticks = _run_length(text, start, "`")
    closing = _BACKTICKS.search(text, start + ticks)
    while closing is not None and len(closing.group()) != ticks:
        closing = _BACKTICKS.search(text, closing.end())
    if closing is None:
        return None
    content = text[start + ticks:closing.start()].replace("\n", " ")
    if content.startswith(" ") and content.endswith(" ") and content.strip():
        content = content[1:-1]
    return {"type": "code", "text": content}, closing.end()


def _skip_verbatim(text, scan):
    here = text[scan]
    if here == "\\":
        return scan + 2
    if here == "`":
        span = _code_span(text, scan)
        return span[1] if span else scan + _run_length(text, scan, "`")
    return None


def _find_closer(text, start, char, length):
    scan = start
    while scan < len(text):
        skipped = _skip_verbatim(text, scan)
        if skipped is not None:
            scan = skipped
        elif text[scan] == char:
            run = _run_length(text, scan, char)
            after = text[scan + run:scan + run + 1]
            word_ends = char == "*" or not after.isalnum()
            if run == length and not text[scan - 1].isspace() and word_ends:
                return scan
            scan += run
        else:
            scan += 1
    return None


def _emphasis(text, start):
    char = text[start]
    run = _run_length(text, start, char)
    if run > _MAX_DELIMITER_RUN:
        return None
    following = text[start + run:start + run + 1]
    if not following or following.isspace():
        return None
    if char == "_" and start > 0 and text[start - 1].isalnum():
        return None
    closer = _find_closer(text, start + run, char, run)
    if closer is None:
        return None
    node = {"type": "strong" if run >= 2 else "emph", "children": _scan(text[start + run:closer])}
    if run == _MAX_DELIMITER_RUN:
        node = {"type": "emph", "children": [node]}
    return node, closer + run


def _closing_bracket(text, start):
    depth = 0
    scan = start
    while scan < len(text):
        skipped = _skip_verbatim(text, scan)
        if skipped is not None:
            scan = skipped
            continue
        if text[scan] == "[":
            depth += 1
        elif text[scan] == "]":
            depth -= 1
            if depth == 0:
                return scan
        scan += 1
    return None


def _link(text, start):
    close = _closing_bracket(text, start)
    if close is None or text[close + 1:close + 2] != "(":
        return None
    scan = close + 2
    depth = 0
    while scan < len(text) and not text[scan].isspace():
        if text[scan] == "(":
            depth += 1
        elif text[scan] == ")":
            if depth == 0:
                break
            depth -= 1
        scan += 1
    href = text[close + 2:scan]
    scan += _run_length(text, scan, " ")
    title = None
    if text[scan:scan + 1] == '"':
        quoted = _TITLE.match(text, scan)
        if quoted is None:
            return None
        title = quoted.group(1)
        scan = quoted.end() + _run_length(text, quoted.end(), " ")
    if text[scan:scan + 1] != ")":
        return None
    children = _scan(text[start + 1:close])
    return {"type": "link", "href": href, "title": title, "children": children}, scan + 1


def _scan(text):
    nodes = []
    pending = ""
    scan = 0
    while scan < len(text):
        here = text[scan]
        found = None
        if here == "\\" and text[scan + 1:scan + 2] in _ESCAPABLE:
            pending += text[scan + 1]
            scan += 2
            continue
        if here == "\\" and text[scan + 1:scan + 2] == "\n":
            found = ({"type": "linebreak"}, scan + 2)
        elif here == "\n":
            kind = "linebreak" if pending.endswith("  ") else "softbreak"
            pending = pending.rstrip(" ")
            found = ({"type": kind}, scan + 1 + _run_length(text, scan + 1, " "))
        elif here == "`":
            found = _code_span(text, scan)
        elif here in "*_":
            found = _emphasis(text, scan)
        elif here == "[":
            found = _link(text, scan)
        elif here == "<":
            match = _AUTOLINK.match(text, scan)
            if match:
                url = match.group(1)
                found = ({"type": "link", "href": url, "title": None,
                          "children": [{"type": "text", "text": url}]}, match.end())
        if found is None:
            width = _run_length(text, scan, here) if here in "`*_" else 1
            pending += text[scan:scan + width]
            scan += width
            continue
        if pending:
            nodes.append({"type": "text", "text": pending})
            pending = ""
        nodes.append(found[0])
        scan = found[1]
    if pending:
        nodes.append({"type": "text", "text": pending})
    return nodes


def mdr_parse_inline(text):
    """Parse inline markup into a list of text, emph, strong, code, link and break nodes."""
    return _scan(text)


def mdr_collect_links(nodes):
    """Return the href of every link in the nodes, depth first in document order."""
    hrefs = []
    for node in nodes:
        if node["type"] == "link":
            hrefs.append(node["href"])
        hrefs.extend(mdr_collect_links(node.get("children", [])))
    return hrefs


def mdr_collect_code_spans(nodes):
    """Return the text of every code span in the nodes, depth first in document order."""
    spans = []
    for node in nodes:
        if node["type"] == "code":
            spans.append(node["text"])
        spans.extend(mdr_collect_code_spans(node.get("children", [])))
    return spans
