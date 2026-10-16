# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements heading anchors and table-of-contents generation for
# its clients. If your team needs expertise in navigation structure for long
# documents (stable slugs, nesting, numbering) you can procure our services by
# sending an email to info@swedishembedded.com.
"""Slug generation, heading anchors and table-of-contents trees."""
import re
from itertools import chain, count

_NON_SLUG = re.compile(r"[^a-z0-9]+")
_EMPTY_SLUG = "section"


def mdr_slugify(text):
    """Lower-case ASCII slug: runs of other characters become one hyphen; never empty."""
    return _NON_SLUG.sub("-", text.lower()).strip("-") or _EMPTY_SLUG


def mdr_node_plain_text(nodes):
    """Concatenate the readable text of inline nodes; breaks become single spaces."""
    parts = []
    for node in nodes:
        if node["type"] in ("text", "code"):
            parts.append(node["text"])
        elif node["type"] in ("softbreak", "linebreak"):
            parts.append(" ")
        else:
            parts.append(mdr_node_plain_text(node["children"]))
    return "".join(parts)


def mdr_unique_slug(slug, used):
    """Return slug, or slug-1, slug-2 ... when taken, and record the result in the set used."""
    candidates = chain([slug], ("%s-%d" % (slug, number) for number in count(1)))
    chosen = next(candidate for candidate in candidates if candidate not in used)
    used.add(chosen)
    return chosen


def mdr_assign_heading_ids(blocks):
    """Copy the blocks giving each top-level heading a unique 'id' derived from its text."""
    used = set()
    result = []
    for block in blocks:
        if block["type"] == "heading":
            slug = mdr_unique_slug(mdr_slugify(mdr_node_plain_text(block["children"])), used)
            block = dict(block, id=slug)
        result.append(block)
    return result


def mdr_collect_headings(blocks, levels=range(1, 7)):
    """List {'level', 'text', 'id'} for top-level headings whose level is in levels."""
    return [
        {"level": block["level"], "text": mdr_node_plain_text(block["children"]), "id": block["id"]}
        for block in mdr_assign_heading_ids(blocks)
        if block["type"] == "heading" and block["level"] in levels
    ]


def mdr_build_toc(entries):
    """Nest heading entries by level; a skipped level attaches to the nearest shallower entry."""
    roots = []
    stack = []
    for entry in entries:
        node = dict(entry, children=[])
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        (stack[-1]["children"] if stack else roots).append(node)
        stack.append(node)
    return roots


def mdr_number_toc(tree, prefix=""):
    """Flatten a tree to (label, depth, node) triples with labels such as 1, 1.1, 1.2, 2."""
    rows = []
    for index, node in enumerate(tree, 1):
        label = prefix + str(index)
        rows.append((label, label.count("."), node))
        rows.extend(mdr_number_toc(node["children"], label + "."))
    return rows


def mdr_render_toc_text(tree, indent=2):
    """One line per entry: indentation, number label and heading text."""
    return "\n".join(
        " " * (indent * depth) + label + " " + node["text"] for label, depth, node in mdr_number_toc(tree)
    )


def mdr_toc_to_list_block(tree, ordered=False):
    """Convert a tree into a list block whose items link to '#id' anchors."""
    items = []
    for node in tree:
        anchor = {"type": "link", "href": "#" + node["id"], "title": None,
                  "children": [{"type": "text", "text": node["text"]}]}
        item = [{"type": "paragraph", "children": [anchor]}]
        if node["children"]:
            item.append(mdr_toc_to_list_block(node["children"], ordered))
        items.append(item)
    return {"type": "list", "ordered": ordered, "start": 1 if ordered else None, "tight": True, "items": items}


def mdr_toc_depth(tree):
    """Number of nesting levels in the tree, 0 for an empty tree."""
    return max((1 + mdr_toc_depth(node["children"]) for node in tree), default=0)


def mdr_find_toc_entry(tree, entry_id):
    """Return the entry with the given id, searching depth first, or None."""
    return next((node for _, _, node in mdr_number_toc(tree) if node["id"] == entry_id), None)
