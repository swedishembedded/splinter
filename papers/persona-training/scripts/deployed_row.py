#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned language
# models for its clients. If your team needs expertise in sizing and
# analysing a small paired evaluation, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Prints the data/deployed.csv row for one run that examined only the deployed arm.

    deployed_row.py REPORT --arm NAME --description TEXT --reference REPORT:ARM

REPORT is the output of `splinter exam --json --deployed-only` (or a log that
contains it) for the candidate under the persona prompt. REFERENCE names the
report and arm it is compared with, the prompted base of a four-arm run of the
same frozen exam. Both answered greedily under the same judge version, so the
arms are paired by task on the tasks both answered. Counts are recomputed from
the per-task records and must agree with each report's own summary; the
interval is the family-clustered percentile bootstrap (10,000 resamples,
seed 1) of the difference in the share of tasks judged right. The
length-matched subset is the tasks whose two answers differ in length by at most
a quarter of the longer, as in the exam report. Standard
library only.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from math import comb

HEADER = (
    "arm,description,candidate,run,families,tasks,families_trained_on,"
    "cand_right,prompted_right,cand_invented,prompted_invented,cand_chars,prompted_chars,"
    "cand_only,prompted_only,family_wins,family_losses,family_ties,"
    "diff_points,ci_low,ci_high,p_tasks,p_families,"
    "lm_tasks,lm_cand_only,lm_prompted_only,lm_p,"
    "inv_cand_only,inv_prompted_only,p_invented,"
    "judge_version,judge_controls,precision_pass,precision_fail,hard_wrong,hard_passed,"
    "reference_run,reference_judge_version"
).split(",")

CANDIDATE_ARM = "candidate-persona"
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 1
LENGTH_MATCH = 0.25  # answers differing by at most this share of the longer one


def load_report(path):
    text = open(path, encoding="utf-8").read()
    start = text.find("\n{\n") + 1 if "\n{\n" in text else text.index("{")
    doc = json.JSONDecoder().raw_decode(text[start:])[0]
    return doc.get("ran", doc), doc.get("run", "")


def upper_tail(wins, n):
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(wins, n + 1)) / 2**n


def greedy(record, arm):
    return record["arms"][arm][0]


def verified(what, logged, recomputed):
    if logged != recomputed:
        sys.exit(f"{what}: report says {logged}, its records give {recomputed}")
    return logged


def check_summary(ran, arm, records):
    """The arm's summary in the report must equal what its records give."""
    summary = next(a for a in ran["arms"] if a["arm"] == arm)
    right = sum(greedy(r, arm)["judged"] for r in records)
    invented = sum(greedy(r, arm)["grounded"] is False for r in records)
    verified(f"{arm} right", summary["right"], right)
    verified(f"{arm} invented", summary["invented"], invented)


def bootstrap(by_family):
    """Percentile interval of the difference, resampling families with replacement."""
    families = list(by_family)
    rng = random.Random(BOOTSTRAP_SEED)
    diffs = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        n = reference = candidate = 0
        for _ in families:
            for ref, cand in by_family[rng.choice(families)]:
                n += 1
                reference += ref
                candidate += cand
        diffs.append((candidate - reference) / n)
    diffs.sort()
    return diffs[int(0.025 * BOOTSTRAP_RESAMPLES)], diffs[int(0.975 * BOOTSTRAP_RESAMPLES)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report")
    ap.add_argument("--arm", required=True, help="short run name")
    ap.add_argument("--description", required=True)
    ap.add_argument("--reference", required=True, help="REPORT:ARM of the prompted base")
    args = ap.parse_args()

    ran, run_id = load_report(args.report)
    ref_path, ref_arm = args.reference.rsplit(":", 1)
    ref_ran, ref_run_id = load_report(ref_path)
    records = ran["records"]
    verified("tasks", ran["tasks"], len(records))
    check_summary(ran, CANDIDATE_ARM, records)
    check_summary(ref_ran, ref_arm, ref_ran["records"])
    version = ran["judge"]["calibration"]["producer"]["version"]
    ref_version = ref_ran["judge"]["calibration"]["producer"]["version"]
    if version != ref_version:
        sys.exit(f"judge version {version} differs from the reference's {ref_version}")

    reference = {r["task"]: r for r in ref_ran["records"]}
    paired = [r for r in records if r["task"] in reference]
    if len(paired) != len(records):
        sys.exit(f"{len(records) - len(paired)} tasks are absent from the reference")
    by_family, family_diff = defaultdict(list), defaultdict(int)
    only = {"cand": 0, "prompted": 0}
    invented_only = {"cand": 0, "prompted": 0}
    cand_right = prompted_right = cand_invented = prompted_invented = 0
    cand_chars = prompted_chars = 0
    matched = {"tasks": 0, "cand": 0, "prompted": 0}
    for rec in paired:
        ref = reference[rec["task"]]
        if ref["family"] != rec["family"]:
            sys.exit(f"task {rec['task']} is in different families in the two reports")
        c, p = greedy(rec, CANDIDATE_ARM), greedy(ref, ref_arm)
        by_family[rec["family"]].append((int(p["judged"]), int(c["judged"])))
        family_diff[rec["family"]] += int(c["judged"]) - int(p["judged"])
        only["cand"] += c["judged"] and not p["judged"]
        only["prompted"] += p["judged"] and not c["judged"]
        cand_right += c["judged"]
        prompted_right += p["judged"]
        c_inv, p_inv = c["grounded"] is False, p["grounded"] is False
        cand_invented += c_inv
        prompted_invented += p_inv
        invented_only["cand"] += c_inv and not p_inv
        invented_only["prompted"] += p_inv and not c_inv
        if abs(c["chars"] - p["chars"]) <= LENGTH_MATCH * max(c["chars"], p["chars"]):
            matched["tasks"] += 1
            matched["cand"] += c["judged"] and not p["judged"]
            matched["prompted"] += p["judged"] and not c["judged"]
        cand_chars += c["chars"]
        prompted_chars += p["chars"]
    n = len(paired)
    fam_wins = sum(v > 0 for v in family_diff.values())
    fam_losses = sum(v < 0 for v in family_diff.values())
    low, high = bootstrap(by_family)
    verified("families", ran["families"], len(by_family))
    out = {
        "arm": args.arm, "description": args.description, "candidate": ran["candidate"], "run": run_id,
        "families": len(by_family), "tasks": n, "families_trained_on": ran["families_trained_on"],
        "cand_right": cand_right, "prompted_right": prompted_right,
        "cand_invented": cand_invented, "prompted_invented": prompted_invented,
        "cand_chars": f"{cand_chars / n:.1f}", "prompted_chars": f"{prompted_chars / n:.1f}",
        "cand_only": only["cand"], "prompted_only": only["prompted"],
        "family_wins": fam_wins, "family_losses": fam_losses,
        "family_ties": len(by_family) - fam_wins - fam_losses,
        "diff_points": f"{100 * (cand_right - prompted_right) / n:.2f}",
        "ci_low": f"{100 * low:.2f}", "ci_high": f"{100 * high:.2f}",
        "p_tasks": f"{upper_tail(only['cand'], only['cand'] + only['prompted']):.8f}",
        "p_families": f"{upper_tail(fam_wins, fam_wins + fam_losses):.8f}",
        "lm_tasks": matched["tasks"], "lm_cand_only": matched["cand"], "lm_prompted_only": matched["prompted"],
        "lm_p": f"{upper_tail(matched['cand'], matched['cand'] + matched['prompted']):.8f}",
        "inv_cand_only": invented_only["cand"], "inv_prompted_only": invented_only["prompted"],
        "p_invented": f"{upper_tail(invented_only['prompted'], sum(invented_only.values())):.8f}",
        "judge_version": version, "judge_controls": ran["judge"]["calibration"]["n"],
        "precision_pass": f"{ran['judge']['calibration']['precision_pass']:.4f}",
        "precision_fail": f"{ran['judge']['calibration']['precision_fail']:.4f}",
        "hard_wrong": ran["hard_controls"]["wrong_answers"], "hard_passed": ran["hard_controls"]["passed"],
        "reference_run": ref_run_id, "reference_judge_version": ref_version,
    }
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow([out[k] for k in HEADER])


if __name__ == "__main__":
    main()
