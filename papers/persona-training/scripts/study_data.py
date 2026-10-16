#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned language
# models for its clients. If your team needs expertise in turning raw
# examination outputs into auditable paired statistics, you can procure our
# services by sending an email to info@swedishembedded.com.
"""Turns the replication study's report into the tables the paper's build reads.

usage: study_data.py REPORT_DIR OUT_DIR [--training-seed-index N]

REPORT_DIR is the `report/` directory `synthetic-author study report` writes.
OUT_DIR receives arms.csv, cells.csv and paired.csv in the columns
scripts/collect.py writes (seed is the author's seed), so that build_data.py
can read them once the paper is laid out for the study's authors.

An adapter is trained once per training seed and the paper's tables have one
row per arm, so the adapter's row is the one of the N-th training seed of each
author, in the order the report lists them (default 0, the first). The
per-seed spread and the sign test across authors are in the report's own
tables (spread.csv, across_authors.csv) and are not reduced here.

Not made here, because the report does not hold them: authors.csv (the beliefs
come from `synthetic-author persona`), audit.csv, corpora.csv, training.csv,
cost.csv, spend.csv, echo.csv, confusion.csv and cross.csv, which have their
own producers. build_data.py lays its tables out for three authors
(AUTHORS) and stays on data/ until the paper is rewritten for the study.
"""
import csv
import os
import sys

ARMS = ["base", "named", "principles", "retrieval", "tuned"]
PAIRS = [("tuned", "base"), ("tuned", "named"), ("tuned", "principles"),
         ("tuned", "retrieval"), ("retrieval", "base"), ("principles", "base")]


def read(report, name):
    with open(os.path.join(report, name), newline="") as f:
        return list(csv.DictReader(f))


def chosen_seeds(rows, index):
    """The training seed of each author's index-th adapter, in listing order."""
    seeds = {}
    for r in rows:
        if r["arm"] == "tuned" and r["training_seed"]:
            seeds.setdefault(r["author"], [])
            if r["training_seed"] not in seeds[r["author"]]:
                seeds[r["author"]].append(r["training_seed"])
    chosen = {}
    for author, listed in seeds.items():
        if index >= len(listed):
            sys.exit(f"author {author} has {len(listed)} measured training seeds, not {index + 1}")
        chosen[author] = listed[index]
    return chosen


def wanted(row, chosen):
    """Whether a row of an arm is the one the paper's table shows."""
    if not row["training_seed"]:
        return True
    return chosen.get(row["author"]) == row["training_seed"]


def write(out, name, header, rows):
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(argv):
    index = 0
    if "--training-seed-index" in argv:
        at = argv.index("--training-seed-index")
        index = int(argv[at + 1])
        del argv[at:at + 2]
    if len(argv) != 2:
        sys.exit(__doc__)
    report, out = argv
    os.makedirs(out, exist_ok=True)
    accuracy = read(report, "accuracy.csv")
    chosen = chosen_seeds(accuracy, index)
    arms = [[r["author"], r["arm"], r["right"], r["n"]]
            for r in accuracy if r["arm"] in ARMS and wanted(r, chosen)]
    cells = [[r["author"], r["arm"], r["cell"], r["right"], r["n"]]
             for r in read(report, "cells.csv") if r["arm"] in ARMS and wanted(r, chosen)]
    paired = [[r["author"], r["candidate"], r["reference"], r["n"],
               r["only_candidate"], r["only_reference"]]
              for r in read(report, "paired.csv")
              if (r["candidate"], r["reference"]) in PAIRS and wanted(
                  {"author": r["author"], "training_seed": r["training_seed"]}, chosen)]
    write(out, "arms.csv", ["seed", "arm", "right", "n"], arms)
    write(out, "cells.csv", ["seed", "arm", "cell", "right", "n"], cells)
    write(out, "paired.csv", ["seed", "arm_a", "arm_b", "n", "only_a", "only_b"], paired)


if __name__ == "__main__":
    main(sys.argv[1:])
