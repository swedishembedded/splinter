# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for source intake whose every section maps back to the bytes it came from,
# for its clients. If your team needs expertise in turning documents into addressable training material then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Oracle `knowledge/sections`: documents built section by section, whose sections' byte ranges are known because
they were written, never parsed (plan 3A).

A plain-text document is paragraphs (one to three lines of words in several scripts) between runs of blank lines (empty,
or spaces, tabs and no-break spaces). A Markdown document is an optional preamble, then sections that each open with an
ATX heading (zero to three spaces, one to six `#`, a space or the end of the line) and hold paragraphs, blank lines and
fenced code blocks whose lines may look like headings; lines that look like headings and are not (four spaces of
indent, seven `#`, `#` with no space after) are body text. Lines end in LF or CRLF, the last one sometimes in none;
documents may open and close with blank lines; one document is empty and one is large.

A section's range runs from the first byte of its first non-blank line to the last byte of its last non-blank line,
line terminator excluded (crates/knowledge/src/sections.rs states this rule); it is recorded as the builder appends
those lines. The check: the sectioner returns exactly these ranges, and so neither fabricates nor drops any text.

Usage: python -I sections.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

NAME = "knowledge/sections"
REFERENCE = "constructed-ground-truth"
WORDS = ["baud", "UART", "Gr\u00f6\u00dfe", "na\u00efve", "caf\u00e9", "\u65e5\u672c\u8a9e", "\u043a\u0438\u0440\u0438\u043b\u043b\u0438\u0446\u0430", "\u6570\u636e", "emoji\U0001f600", "e\u0301", "x", "C#"]
BLANKS = ["", " ", "\t", "  \t ", "\u00a0", " \u00a0 "]
FAKE_HEADINGS = ["    # four spaces of indent", "####### seven", "#nospace", "\t# a tab first"]


class Builder:
    def __init__(self, rng):
        self.rng = rng
        self.out = bytearray()
        self.sections = []
        self.open = None

    def line(self, text, blank=False):
        start = len(self.out)
        self.out += text.encode("utf-8")
        end = len(self.out)
        self.out += self.rng.choice([b"\n", b"\n", b"\r\n"])
        if not blank:
            self.open = [self.open[0] if self.open else start, end]

    def blank(self):
        self.line(self.rng.choice(BLANKS), blank=True)

    def close(self):
        if self.open:
            self.sections.append(self.open)
        self.open = None

    def text(self):
        data = bytes(self.out)
        # The last line sometimes has no terminator; a range never includes one either way.
        if self.rng.random() < 0.3 and data.endswith(b"\r\n"):
            data = data[:-2]
        elif self.rng.random() < 0.3 and data.endswith(b"\n"):
            data = data[:-1]
        return data.decode("utf-8")

    def words(self, n):
        return " ".join(self.rng.choice(WORDS) for _ in range(n))


def plain(rng, paragraphs):
    b = Builder(rng)
    for _ in range(rng.randint(0, 2)):
        b.blank()
    for p in range(paragraphs):
        for _ in range(rng.randint(1, 3)):
            b.line(b.words(rng.randint(1, 6)))
        b.close()
        for _ in range(rng.randint(1, 3) if p + 1 < paragraphs else rng.randint(0, 2)):
            b.blank()
    return b.text(), b.sections


def markdown(rng, count):
    b = Builder(rng)
    for _ in range(rng.randint(0, 2)):
        b.blank()
    if rng.random() < 0.5:
        b.line("A preamble " + b.words(3))
    for _ in range(count):
        b.close()
        heading = " " * rng.randint(0, 3) + "#" * rng.randint(1, 6)
        b.line(heading + rng.choice(["", " " + b.words(2), "\t" + b.words(1)]))
        for _ in range(rng.randint(0, 3)):
            kind = rng.choice(["para", "blank", "fake", "fence"])
            if kind == "para":
                b.line(b.words(rng.randint(1, 5)))
            elif kind == "blank":
                b.blank()
            elif kind == "fake":
                b.line(rng.choice(FAKE_HEADINGS))
            else:
                mark = rng.choice(["```", "````", "~~~"])
                b.line(mark + rng.choice(["", "rust"]))
                b.line("# inside a fence, not a heading")
                b.line(b.words(2))
                b.line(mark)
        for _ in range(rng.randint(0, 2)):
            b.blank()
    b.close()
    return b.text(), b.sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=300)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261127 if args.seed is None else args.seed)
    cases = [{"media_type": "text/plain", "text": "", "sections": []},
             {"media_type": "text/markdown", "text": "", "sections": []}]
    for n in range(args.cases):
        if n % 2:
            text, sections = markdown(rng, rng.randint(0, 5))
            media = "text/markdown"
        else:
            text, sections = plain(rng, rng.randint(0, 5))
            media = "text/plain"
        cases.append({"media_type": media, "text": text, "sections": sections})
    big, big_sections = plain(rng, 2000)
    cases.append({"media_type": "text/plain", "text": big, "sections": big_sections})
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
