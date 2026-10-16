# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements plain-text rendering of structured documents (wrapped
# paragraphs, hanging list markers, quotes and underlined headings) for its clients.
# If your team needs expertise in producing readable terminal and e-mail output from
# markup you can procure our services by sending an email to info@swedishembedded.com.
"""Render parsed blocks and inline nodes as wrapped plain text."""
from mdrender.wrapping import mdr_wrap_text

_BULLETS = "*-+"
_CODE_INDENT = "    "
_QUOTE_PREFIX = "> "


def mdr_render_inline_text(nodes):
    """Render inline nodes to a string; hard breaks become newlines and soft breaks spaces."""
    parts = []
    for node in nodes:
        kind = node["type"]
        if kind == "text":
            parts.append(node["text"])
        elif kind == "emph":
            parts.append("_" + mdr_render_inline_text(node["children"]) + "_")
        elif kind == "strong":
            parts.append("*" + mdr_render_inline_text(node["children"]) + "*")
        elif kind == "code":
            parts.append("`" + node["text"] + "`")
        elif kind == "link":
            label = mdr_render_inline_text(node["children"])
            parts.append(label if label == node["href"] else "%s (%s)" % (label, node["href"]))
        elif kind == "softbreak":
            parts.append(" ")
        elif kind == "linebreak":
            parts.append("\n")
        else:
            raise ValueError("unknown node type: " + kind)
    return "".join(parts)


def _wrap_inline(nodes, width, first_indent="", rest_indent=""):
    lines = []
    for segment in mdr_render_inline_text(nodes).split("\n"):
        lines.extend(mdr_wrap_text(segment, width, first_indent if not lines else rest_indent, rest_indent))
    return lines


def _heading_lines(block, width):
    level = block["level"]
    if level > 2:
        return _wrap_inline(block["children"], width, "#" * level + " ", " " * (level + 1))
    lines = _wrap_inline(block["children"], width)
    if not lines:
        return []
    return lines + [("=" if level == 1 else "-") * max(len(line) for line in lines)]


def _prefixed(lines, first, rest):
    result = []
    for n, line in enumerate(lines):
        prefix = first if n == 0 else rest
        result.append(prefix + line if line else prefix.rstrip())
    return result


def _join_blocks(blocks, width, depth, separator):
    lines = []
    for block in blocks:
        rendered = _block_lines(block, width, depth)
        if rendered:
            lines.extend((separator if lines else []) + rendered)
    return lines


def _list_lines(block, width, depth):
    start = block["start"]
    if block["ordered"]:
        labels = ["%d." % (start + n) for n in range(len(block["items"]))]
    else:
        labels = [_BULLETS[depth % len(_BULLETS)]] * len(block["items"])
    widest = max(len(label) for label in labels)
    separator = [] if block["tight"] else [""]
    lines = []
    for label, item in zip(labels, block["items"]):
        marker = label.ljust(widest) + " "
        body = _join_blocks(item, width - len(marker), depth + 1, separator)
        lines.extend((separator if lines else []) + (_prefixed(body, marker, " " * len(marker)) or [label]))
    return lines


def _block_lines(block, width, depth):
    kind = block["type"]
    if kind == "paragraph":
        return _wrap_inline(block["children"], width)
    if kind == "heading":
        return _heading_lines(block, width)
    if kind == "rule":
        return ["-" * width]
    if kind == "code":
        return _prefixed(block["text"].split("\n"), _CODE_INDENT, _CODE_INDENT)
    if kind == "quote":
        inner = _join_blocks(block["children"], width - len(_QUOTE_PREFIX), depth, [""])
        return _prefixed(inner, _QUOTE_PREFIX, _QUOTE_PREFIX)
    if kind == "list":
        return _list_lines(block, width, depth)
    raise ValueError("unknown block type: " + kind)


def mdr_render_text(blocks, width=72):
    """Render blocks as plain text of at most width columns, separated by blank lines."""
    return "\n".join(_join_blocks(blocks, width, 0, [""]))
