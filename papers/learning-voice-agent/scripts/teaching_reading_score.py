# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Score readers of teaching against the gold labels. Usage: teaching_reading_score.py UTTERANCES READINGS [split]

A reading is right for a teaching when its kind is the gold kind and its content
holds the gold key; right for a non-teaching when it is null. Three readers:
the plain one, the model, and the plain one with the model asked only when it finds nothing.
"""
import json
import sys


def kind_and_text(t):
    if t is None:
        return "none", ""
    (name, body), = t.items()
    key = {"Fact": "fact", "Rule": "rule", "Procedure": "procedure"}[name]
    return key, json.dumps(body).lower()


def right(gold, reading):
    kind, text = kind_and_text(reading)
    if gold["kind"] == "none":
        return kind == "none"
    return kind == gold["kind"] and any(k in text for k in gold["key"].lower().split("|"))


def score(gold_rows, readings, pick):
    out = {"positives": 0, "found": 0, "negatives": 0, "false_positives": 0, "wrong_kind_or_content": 0}
    for g in gold_rows:
        r = pick(readings[g["id"]])
        if g["kind"] == "none":
            out["negatives"] += 1
            out["false_positives"] += r is not None
        else:
            out["positives"] += 1
            out["found"] += right(g, r)
            out["wrong_kind_or_content"] += (r is not None) and not right(g, r)
    return out


def main():
    gold = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    readings = {r["id"]: r for r in json.load(open(sys.argv[2]))}
    split = sys.argv[3] if len(sys.argv) > 3 else None
    rows = [g for g in gold if split in (None, g["split"])]
    readers = {
        "plain": lambda r: r["plain"],
        "model": lambda r: r["model"],
        "plain then model": lambda r: r["plain"] or r["model"],
    }
    for name, pick in readers.items():
        s = score(rows, readings, pick)
        print(f"{name:17s} found {s['found']:3d}/{s['positives']:3d}  false positives {s['false_positives']:2d}/{s['negatives']:2d}  wrong reading of a teaching {s['wrong_kind_or_content']}")


if __name__ == "__main__":
    main()
