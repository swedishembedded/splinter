# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Summarise a teach-eval report: pass rates by kind, untaught against taught. Usage: teaching_summary.py REPORT"""
import collections
import json
import sys


def summarise(report):
    kinds = collections.defaultdict(lambda: {"scenarios": 0, "asks": 0, "untaught": 0, "taught": 0, "understood": 0, "teachings": 0, "kept": 0})
    for r in report["results"]:
        k = kinds[r["kind"]]
        k["scenarios"] += 1
        k["understood"] += r["understood"]
        k["teachings"] += r["teachings"]
        k["kept"] += r["kept"]
        for o in r["outcomes"]:
            k["asks"] += 1
            k["untaught"] += o["untaught_passes"]
            k["taught"] += o["taught_passes"]
    return dict(kinds)


if __name__ == "__main__":
    report = json.load(open(sys.argv[1]))
    for kind, k in sorted(summarise(report).items()):
        print(f"{kind:11s} scenarios {k['scenarios']:3d}  asks {k['asks']:3d}  untaught {k['untaught']:3d}  taught {k['taught']:3d}  understood {k['understood']}/{k['teachings']}  kept {k['kept']}")
