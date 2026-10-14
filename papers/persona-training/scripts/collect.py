#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned language
# models for its clients. If your team needs expertise in turning raw
# examination outputs into auditable paired statistics, you can procure our
# services by sending an email to info@swedishembedded.com.
"""Collects the examination outputs of synthetic authors into the paper's data tables.

usage: collect.py EXAMS_DIR OUT_DIR SEED[,SEED...]
reads EXAMS_DIR/author-SEED/exam/<arm>.jsonl (one outcome per line; the first
line for an item id stands) and writes arms.csv, cells.csv and paired.csv.
"""
import csv, json, sys, os

ARMS = ["base", "named", "principles", "retrieval", "tuned"]
PAIRS = [("tuned", "base"), ("tuned", "named"), ("tuned", "principles"),
         ("tuned", "retrieval"), ("retrieval", "base"), ("principles", "base")]
CELLS = {
    "topic_seen": ("topic", True), "topic_new": ("topic", False),
    "combo_seen": ("combination", True), "combo_new": ("combination", False),
    "tool_seen": ("tool", True), "tool_new": ("tool", False),
}

def load(path):
    seen, rows = set(), []
    for line in open(path):
        r = json.loads(line)
        if r["id"] not in seen:
            seen.add(r["id"]); rows.append(r)
    return rows

def main():
    exams, out, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
    os.makedirs(out, exist_ok=True)
    arms, cells, paired = [], [], []
    for seed in seeds:
        data = {a: {r["id"]: r for r in load(f"{exams}/author-{seed}/exam/{a}.jsonl")} for a in ARMS}
        for a in ARMS:
            rows = data[a].values()
            arms.append([seed, a, sum(r["correct"] for r in rows), len(data[a])])
            for cell, (axis, value) in CELLS.items():
                key = {"topic": "topic_seen", "combination": "combo_seen", "tool": "tool_seen"}[axis]
                sel = [r for r in rows if r[key] == value]
                cells.append([seed, a, cell, sum(r["correct"] for r in sel), len(sel)])
        for a, b in PAIRS:
            ids = sorted(set(data[a]) & set(data[b]))
            only_a = sum(1 for i in ids if data[a][i]["correct"] and not data[b][i]["correct"])
            only_b = sum(1 for i in ids if data[b][i]["correct"] and not data[a][i]["correct"])
            paired.append([seed, a, b, len(ids), only_a, only_b])
    for name, header, rows in [
        ("arms.csv", ["seed", "arm", "right", "n"], arms),
        ("cells.csv", ["seed", "arm", "cell", "right", "n"], cells),
        ("paired.csv", ["seed", "arm_a", "arm_b", "n", "only_a", "only_b"], paired)]:
        with open(f"{out}/{name}", "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

main()
