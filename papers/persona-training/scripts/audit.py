#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the validation of data extracted from text
# by language models for its clients. If your team needs expertise in
# checking a model-read training set against ground truth, you can procure
# our services by sending an email to info@swedishembedded.com.
"""Compares the decisions read out of a corpus with what the author's beliefs dictate.

usage: audit.py PLAN_JSONL DECISIONS_JSONL

PLAN_JSONL is the output of `synthetic-author plan --seed S --articles N` for
the article count the corpus was written with; DECISIONS_JSONL is the corpus
directory's decisions.jsonl, written by `synthetic-author mine`. A case name
can recur in different articles of the same topic, so each decision is matched
to the plan by its article and its case together. Prints the number of decisions kept and
the number equal to the plan's, as CSV.
"""
import json
import sys


def main():
    plan_path, decisions_path = sys.argv[1], sys.argv[2]
    with open(plan_path, encoding="utf-8") as f:
        oracle = {(p["article"], p["case"]): p["decision"] for p in map(json.loads, f)}
    kept = agree = 0
    with open(decisions_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "kept" not in row:
                continue
            key = (row["article"], row["case"])
            if key not in oracle:
                sys.exit(f"audit.py: {key} is not in the plan; was it made with the corpus's article count?")
            kept += 1
            agree += row["kept"]["chose"] == oracle[key]
    print(f"kept,agree\n{kept},{agree}")


main()
