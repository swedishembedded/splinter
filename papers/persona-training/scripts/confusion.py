#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the failure analysis of fine-tuned language
# models for its clients. If your team needs expertise in finding out which
# decisions a tuned model gets wrong and in what direction, you can procure
# our services by sending an email to info@swedishembedded.com.
"""Counts, for one arm, which option each examination item called for and which one the call took.

usage: confusion.py ITEMS_DIR EXAMS_DIR ARM SEED[,SEED...]

ITEMS_DIR holds items-SEED.jsonl, the output of `synthetic-author items --seed
SEED` (each item as the model is asked it, with the label the beliefs dictate);
EXAMS_DIR holds author-SEED/exam/ARM.jsonl (one outcome per line; the first
line for an item id stands, as in collect.py). An option is named by what it
is, whatever label the item's tool gives it. Prints CSV: seed, expected,
chosen, items; a reply with no call, or a call naming no listed label, is
counted as chosen "none".
"""
import csv
import json
import re
import sys

# The phrase each option is listed with, as the examination renders it.
PHRASES = {
    "settle the matter yourself, at once": "act_alone",
    "bring it before the assembly and act on its word": "consult_assembly",
    "accept the offer of outside help": "accept_offer",
    "wait, and do nothing yet": "wait",
}
# The argument through which each tool names the option taken.
CHOICE_ARGUMENT = {
    "decide": "choice", "choose_course": "course", "rule_on_matter": "ruling",
    "settle_matter": "way", "log_resolution": "selected",
}
OPTION_LINE = re.compile(r"^- ([^:]+): (.+)$", re.MULTILINE)


def first_outcomes(path):
    outcomes = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            outcomes.setdefault(row["id"], row)
    return outcomes


def main():
    items_dir, exams_dir, arm = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = [int(s) for s in sys.argv[4].split(",")]
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["seed", "expected", "chosen", "items"])
    for seed in seeds:
        outcomes = first_outcomes(f"{exams_dir}/author-{seed}/exam/{arm}.jsonl")
        counts = {}
        with open(f"{items_dir}/items-{seed}.jsonl", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        if len(items) != len(outcomes) or {i["id"] for i in items} != set(outcomes):
            sys.exit(f"confusion.py: {seed}: the items and the {arm} outcomes do not cover the same ids")
        for item in items:
            labels = {label: PHRASES[phrase] for label, phrase in OPTION_LINE.findall(item["prompt"])}
            expected = labels[item["expected"]]
            call = outcomes[item["id"]]["call"]
            chosen = "none"
            if call and call["name"] == item["tool"]:
                chosen = labels.get(call["arguments"].get(CHOICE_ARGUMENT[item["tool"]]), "none")
            if (chosen == expected) != outcomes[item["id"]]["correct"]:
                sys.exit(f"confusion.py: {seed} {item['id']}: the recorded grade disagrees with the call")
            counts[(expected, chosen)] = counts.get((expected, chosen), 0) + 1
        for (expected, chosen), n in sorted(counts.items()):
            writer.writerow([seed, expected, chosen, n])


main()
