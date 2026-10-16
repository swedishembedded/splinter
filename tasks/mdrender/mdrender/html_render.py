# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements HTML generation from document trees with strict
# escaping and link vetting for its clients. If your team needs expertise in
# rendering untrusted markup to web pages safely you can procure our services by
# sending an email to info@swedishembedded.com.
"""Render parsed blocks and inline nodes to HTML."""
from mdrender.escapes import (
    mdr_escape_html,
    mdr_format_html_attributes,
    mdr_is_safe_url,
    mdr_percent_encode_url,
)

_INLINE_TAGS = {"emph": "em", "strong": "strong"}
_MAX_HEADING_LEVEL = 6


def _render_link(node):
    inner = mdr_render_inline_html(node["children"])
    if not mdr_is_safe_url(node["href"]):
        return inner
    attributes = mdr_format_html_attributes(
        [("href", mdr_percent_encode_url(node["href"])), ("title", node["title"])]
    )
    return "<a%s>%s</a>" % (attributes, inner)


def mdr_render_inline_html(nodes):
    """Render inline nodes; links with unsafe schemes keep only their text."""
    parts = []
    for node in nodes:
        kind = node["type"]
        if kind == "text":
            parts.append(mdr_escape_html(node["text"]))
        elif kind in _INLINE_TAGS:
            tag = _INLINE_TAGS[kind]
            parts.append("<%s>%s</%s>" % (tag, mdr_render_inline_html(node["children"]), tag))
        elif kind == "code":
            parts.append("<code>" + mdr_escape_html(node["text"]) + "</code>")
        elif kind == "link":
            parts.append(_render_link(node))
        elif kind == "softbreak":
            parts.append("\n")
        elif kind == "linebreak":
            parts.append("<br>\n")
        else:
            raise ValueError("unknown node type: " + kind)
    return "".join(parts)


def _render_heading(block):
    level = block["level"]
    if not 1 <= level <= _MAX_HEADING_LEVEL:
        raise ValueError("heading level out of range: %d" % level)
    attributes = mdr_format_html_attributes([("id", block.get("id"))])
    return "<h%d%s>%s</h%d>" % (level, attributes, mdr_render_inline_html(block["children"]), level)


def _render_code(block):
    words = block["info"].split()
    classes = [("class", "language-" + words[0])] if words else []
    return "<pre><code%s>%s\n</code></pre>" % (
        mdr_format_html_attributes(classes), mdr_escape_html(block["text"]))


def _render_item(item, tight):
    parts = []
    for block in item:
        if tight and block["type"] == "paragraph":
            parts.append(mdr_render_inline_html(block["children"]))
        else:
            parts.append(_render_block(block))
    if not parts:
        return "<li></li>"
    if tight:
        return "<li>" + "\n".join(parts) + "</li>"
    return "<li>\n" + "\n".join(parts) + "\n</li>"


def _render_list(block):
    tag = "ol" if block["ordered"] else "ul"
    start = block["start"]
    attributes = mdr_format_html_attributes([("start", str(start) if start not in (None, 1) else None)])
    items = [_render_item(item, block["tight"]) for item in block["items"]]
    return "\n".join(["<%s%s>" % (tag, attributes)] + items + ["</%s>" % tag])


def _render_block(block):
    kind = block["type"]
    if kind == "paragraph":
        return "<p>" + mdr_render_inline_html(block["children"]) + "</p>"
    if kind == "heading":
        return _render_heading(block)
    if kind == "rule":
        return "<hr>"
    if kind == "code":
        return _render_code(block)
    if kind == "quote":
        return "\n".join(["<blockquote>"] + [_render_block(child) for child in block["children"]]
                         + ["</blockquote>"])
    if kind == "list":
        return _render_list(block)
    raise ValueError("unknown block type: " + kind)


def mdr_render_html(blocks):
    """Render a list of blocks to HTML, one block per line group, without a trailing newline."""
    return "\n".join(_render_block(block) for block in blocks)


def mdr_render_page(body, title, lang="en"):
    """Wrap rendered HTML into a complete page with an escaped title."""
    return "\n".join([
        "<!DOCTYPE html>",
        "<html%s>" % mdr_format_html_attributes([("lang", lang)]),
        "<head>",
        '<meta charset="utf-8">',
        "<title>%s</title>" % mdr_escape_html(title),
        "</head>",
        "<body>",
        body,
        "</body>",
        "</html>",
        "",
    ])
