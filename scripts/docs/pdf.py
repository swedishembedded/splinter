#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements documentation pipelines that publish one manual
# in every format from a single source of order, for its clients. If your team needs
# expertise in documentation engineering then you can procure our services by sending
# an email to info@swedishembedded.com.

"""Render a documentation publication as one PDF, in the manifest's order.

usage: pdf.py PUBLICATION_DIR OUTPUT.pdf

PUBLICATION_DIR is what the publisher wrote (manifest.json and the pages). The
order is the publication manifest's navigation, which comes from the source
manifest, so the PDF has no ordering of its own. Each navigation group becomes a
part, each page a chapter. Links between pages become plain text, since a PDF
has no page to open; absolute links stay links.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

FENCE = re.compile(r"^\s*(```|~~~)")
PAGE_LINK = re.compile(r"\[([^\]]+)\]\((?!https?:|mailto:|#)[^)\s]*\.md(?:#[^)]*)?\)")
HEADING = re.compile(r"^(#{1,5}) ")
HTML_COMMENT = re.compile(r"<!--.*?-->\n?", re.S)


def leaves(nodes):
    """The (group title, route) of every page below `nodes`, in order."""
    for node in nodes:
        if "route" in node:
            yield node["route"]
        else:
            yield from leaves(node["children"])


def chapter(text):
    """A page as a chapter: headings one level deeper, page links as text."""
    out, fenced = [], False
    for line in HTML_COMMENT.sub("", text).splitlines():
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            line = HEADING.sub(lambda m: m.group(1) + "# ", line)
            line = PAGE_LINK.sub(r"\1", line)
        out.append(line)
    return "\n".join(out)


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    publication, output = Path(argv[1]), Path(argv[2])
    manifest = json.loads((publication / "manifest.json").read_text(encoding="utf-8"))
    files = {page["route"]: page["file"] for page in manifest["pages"]}
    parts = []
    home = files.get("")
    if home:
        parts.append(chapter((publication / home).read_text(encoding="utf-8")))
    for group in manifest["navigation"]:
        parts.append("# " + group["title"])
        for route in leaves(group.get("children", [])):
            if route != "":
                parts.append(chapter((publication / files[route]).read_text(encoding="utf-8")))
    project = manifest["project"]
    header = f'---\ntitle: "{project["title"]}"\nsubtitle: "{project["description"]}"\n---\n\n'
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "manual.md"
        source.write_text(header + "\n\n".join(parts) + "\n", encoding="utf-8")
        preamble = Path(work) / "preamble.tex"
        preamble.write_text(
            "\\usepackage{fvextra}\n"
            "\\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\\\\{\\}}\n"
            "\\usepackage{longtable,booktabs,array}\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "pandoc", str(source), "--from", "gfm", "--pdf-engine=xelatex", "--toc",
                "--top-level-division=chapter", "-V", "documentclass=report", "-V", "geometry:margin=2.2cm",
                "-V", "fontsize=10pt", "-H", str(preamble), "-o", str(output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main(sys.argv)
