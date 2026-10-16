#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements measurement of model-written training data
# for its clients. If your team needs expertise in finding out why a
# generator's proposals are refused, you can procure our services by sending
# an email to info@swedishembedded.com.

"""Classifies the `no_subject` rejections of a task report against the admission rule.

    classify_rejections.py TASKS_JSON

The rule: the subject the generator names must be in the instruction (case,
whitespace and code markup aside) and named by the source or a cited section.
Each rejection is read against it and counted in one class:

  markup    the subject's words are in the instruction once code quotes and
            quotation marks are removed (the rule now ignores them)
  apart     every word of the subject is in the instruction, not adjacent
  absent    the subject is not in the instruction
  unsourced the instruction names it but neither the source nor a cited section does
"""

import json
import re
import sys
from collections import Counter


def plain(text):
    return re.sub(r"[`'\"*]", "", text.replace("()", "")).lower()


def classify(detail):
    named = re.match(r'the instruction does not name its subject "(.*?)" \(proposed: "(.*)"\)$', detail)
    if named:
        subject, instruction = named.groups()
        if plain(subject) in plain(instruction):
            return "markup"
        if set(re.findall(r"\w+", plain(subject))) <= set(re.findall(r"\w+", plain(instruction))):
            return "apart"
        return "absent"
    if "is named neither by" in detail:
        return "unsourced"
    return "other"


def main():
    report = json.load(open(sys.argv[1]))
    counts = Counter(classify(r["detail"]) for r in report["rejections"] if r["reason"] == "no_subject")
    print(json.dumps(dict(sorted(counts.items()))))


if __name__ == "__main__":
    main()
