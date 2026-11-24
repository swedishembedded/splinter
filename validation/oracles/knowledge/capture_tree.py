# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for capturing repositories whose every captured byte and every skipped
# file is accounted for, for its clients. If your team needs expertise in training-data provenance then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Oracle `knowledge/capture-tree`: directory trees built entry by entry, with what a repository capture must make of
them known because each entry was chosen to be captured or skipped (plan 3A).

A tree holds text files (Markdown and plain, in several scripts), files that are not text (a NUL byte, bytes that
are not UTF-8), files over the size limit, symbolic links, a file whose name is not UTF-8, the directories a capture
never enters (`.git`, `target`, `node_modules`) below the root, and a nested repository; names are chosen so that the
order of the walk and the order of the names disagree (`a/b` against `a-b`, `A` against `a`).

The expected capture, by the rules crates/knowledge/src/capture/repository.rs states: every regular UTF-8 text file
of at most the limit is a part, named by its `/`-separated path, Markdown by extension; an oversized file is skipped
as too large with its size, a file that is not text as not text, a link as not regular, a name that is not UTF-8 as
such; ignored directories are neither entered nor listed; parts and skipped entries are each in the byte order of
their names (a source orders its parts by name, crates/core/src/source.rs), not in the order of the walk.

Usage: python -I capture_tree.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

NAME = "knowledge/capture-tree"
REFERENCE = "constructed-ground-truth"
LIMIT = 64
TEXTS = ["baud 115200\n", "# Title\n\nGr\u00f6\u00dfe caf\u00e9\n", "\u65e5\u672c\u8a9e\n", "", "x" * LIMIT]
DIRS = ["", "a/", "a-b/", "A/", "docs/", "sub/", "a/deep/"]


def tree(rng):
    entries, parts, skipped = [], [], []
    taken = set()

    def fresh(name):
        if name in taken or any(t.startswith(name + "/") or name.startswith(t + "/") for t in taken):
            return False
        taken.add(name)
        return True

    for _ in range(rng.randint(3, 10)):
        d = rng.choice(DIRS)
        kind = rng.choice(["text", "text", "markdown", "nul", "latin1", "big", "link"])
        stem = rng.choice(["a", "b", "a-b", "A", "notes", "z"])
        ext = {"markdown": ".md", "text": rng.choice([".txt", ".rs", ""])}.get(kind, ".bin")
        name = f"{d}{stem}{ext}"
        if not fresh(name):
            continue
        if kind in ("text", "markdown"):
            text = rng.choice(TEXTS)
            entries.append({"path": name, "kind": "file", "hex": text.encode("utf-8").hex()})
            parts.append({"name": name, "media_type": "text/markdown" if ext == ".md" else "text/plain",
                          "text": text})
        elif kind == "nul":
            entries.append({"path": name, "kind": "file", "hex": b"ab\x00cd".hex()})
            skipped.append({"path": name, "reason": "not_text"})
        elif kind == "latin1":
            entries.append({"path": name, "kind": "file", "hex": "caf\u00e9".encode("latin-1").hex()})
            skipped.append({"path": name, "reason": "not_text"})
        elif kind == "big":
            size = LIMIT + rng.randint(1, 40)
            entries.append({"path": name, "kind": "file", "hex": (b"y" * size).hex()})
            skipped.append({"path": name, "reason": {"too_large": {"bytes": size}}})
        else:
            entries.append({"path": name, "kind": "link", "target": "elsewhere"})
            skipped.append({"path": name, "reason": "not_regular"})
    if rng.random() < 0.5:
        for ignored in ["target/out.txt", "a/node_modules/m.txt", "sub/.git/config", "a/deep/target/x.txt"]:
            if fresh(ignored):
                entries.append({"path": ignored, "kind": "file", "hex": b"ignored\n".hex()})
    if rng.random() < 0.3:
        # A name that is not UTF-8 (0xFF): recorded by its lossy spelling.
        entries.append({"path": "odd-\\xff.txt", "kind": "file", "name_hex": "6f64642dff2e747874", "hex": b"t\n".hex()})
        skipped.append({"path": "odd-\ufffd.txt", "reason": "non_utf8_name"})
    parts.sort(key=lambda p: p["name"].encode("utf-8"))
    skipped.sort(key=lambda s: s["path"].encode("utf-8"))
    return {"limit": LIMIT, "entries": entries, "parts": parts, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=200)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261128 if args.seed is None else args.seed)
    cases = [tree(rng) for _ in range(args.cases)]
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
