#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Text hygiene gate: the plain-text rules every tracked file obeys.

One script instead of a third-party hook bundle, so installing and running
the hooks never needs a network clone. Rules, per text file:

  * no typographic punctuation - em/en dashes, curly quotes, ellipsis and
    non-breaking spaces become their ASCII equivalents (a non-ASCII letter
    such as the one in the copyright line is fine; only punctuation that has
    an ASCII spelling is refused);
  * no trailing whitespace, exactly one final newline, LF line endings;
  * no merge-conflict markers and no private keys;
  * .toml, .json, .yaml/.yml files parse.

Usage:
    scripts/hooks/text-hygiene.py              # check every tracked file
    scripts/hooks/text-hygiene.py <file> ...   # check these
    scripts/hooks/text-hygiene.py --fix <file> ...

With --fix, the mechanical rules (punctuation, whitespace, line endings) are
repaired in place; the file is still reported, and the exit status is still
1, so a commit stops and the repair is re-staged deliberately rather than
committed unseen. Conflict markers, private keys and parse errors are never
auto-fixed.
"""
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

# Written as escapes so this file passes its own rule.
PUNCTUATION = {
    "\u2014": "-",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
}
PUNCTUATION_RE = re.compile("[" + "".join(PUNCTUATION) + "]")

# Verbatim source documents: their punctuation is input the pipeline must
# handle, so it is data, not prose. Exempt from the punctuation rule only;
# every other rule still applies. Each entry is a reviewed decision.
VERBATIM_DOCUMENTS = {
    "crates/splinter/examples/stm32_datasheet.md",
}
CONFLICT_RE = re.compile(r"^(<<<<<<< |=======$|>>>>>>> )", re.MULTILINE)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True)
    return [p for p in out.stdout.decode().split("\0") if p]


def repaired(text: str, verbatim: bool) -> str:
    if not verbatim:
        text = PUNCTUATION_RE.sub(lambda m: PUNCTUATION[m.group(0)], text)
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    lines = [ln.rstrip() for ln in lines]
    return "\n".join(lines).rstrip("\n") + "\n" if text.strip() else text


def structural_errors(path: str, text: str):
    errors = []
    if CONFLICT_RE.search(text):
        errors.append("unresolved merge-conflict marker")
    if PRIVATE_KEY_RE.search(text):
        errors.append("contains a private key")
    try:
        if path.endswith(".toml"):
            tomllib.loads(text)
        elif path.endswith(".json"):
            json.loads(text)
        elif path.endswith((".yaml", ".yml")):
            yaml.safe_load(text)
    except Exception as exc:  # every parser raises its own type
        errors.append(f"does not parse: {exc}")
    return errors


def main(argv):
    fix = "--fix" in argv
    paths = [a for a in argv if a != "--fix"] or tracked_files()
    failed = False
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            print(f"{path}: not valid UTF-8")
            failed = True
            continue

        problems = structural_errors(path, text)
        clean = repaired(text, path in VERBATIM_DOCUMENTS)
        if clean != text:
            if fix:
                p.write_text(clean, encoding="utf-8")
                problems.append("repaired punctuation/whitespace/line endings - re-stage it")
            else:
                problems.append(
                    "typographic punctuation, trailing whitespace, CRLF or a missing/extra final newline"
                )
        for problem in problems:
            print(f"{path}: {problem}")
        failed |= bool(problems)

    if failed and not fix:
        print("\ntext-hygiene: run 'scripts/hooks/text-hygiene.py --fix <file>...' for the mechanical fixes.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
