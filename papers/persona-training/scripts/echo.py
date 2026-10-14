#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements leakage audits of model-written training
# corpora for its clients. If your team needs expertise in checking whether
# a generated corpus gives away what it was meant to keep implicit, you can
# procure our services by sending an email to info@swedishembedded.com.
"""Counts the kept essays that echo the wording of the author's private notes in the narrator's voice.

usage: echo.py SEED SHOW_JSON ARTICLES_DIR [SEED SHOW_JSON ARTICLES_DIR ...]

SHOW_JSON is the output of `synthetic-author show --seed SEED`, whose
`input.convictions` are the four private notes the writer is given, in the
second person; ARTICLES_DIR is the corpus directory's articles/ (one kept
essay per NNNN.json). The essay gate turns away nine consecutive words of a
note as written; this counts the essays that repeat RUN consecutive words of
one of the author's own notes once "you", "your" and "yourself" are put in
the first person, under the gates' normalisation (lower case, runs of letters,
digits and apostrophes). Prints CSV: seed, essays, echoing.
"""
import csv
import glob
import json
import re
import sys

RUN = 6
FIRST_PERSON = {"you": "i", "your": "my", "yourself": "myself"}


def words(text):
    return [w.lower() for w in re.split(r"[^\w']+", text) if w]


def runs(tokens):
    return {tuple(tokens[i:i + RUN]) for i in range(len(tokens) - RUN + 1)}


def main():
    args = sys.argv[1:]
    if not args or len(args) % 3:
        sys.exit(__doc__)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["seed", "essays", "echoing"])
    for seed, show, articles in zip(args[0::3], args[1::3], args[2::3]):
        with open(show, encoding="utf-8") as f:
            notes = json.load(f)["input"]["convictions"]
        note_runs = set().union(*(runs([FIRST_PERSON.get(w, w) for w in words(n)]) for n in notes))
        essays = sorted(glob.glob(f"{articles}/[0-9]*.json"))
        if not essays:
            sys.exit(f"echo.py: no essays in {articles}")
        echoing = 0
        for path in essays:
            with open(path, encoding="utf-8") as f:
                echoing += bool(runs(words(json.load(f)["body"])) & note_runs)
        writer.writerow([seed, len(essays), echoing])


main()
