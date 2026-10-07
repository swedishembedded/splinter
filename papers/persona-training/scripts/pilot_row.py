#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned language
# models for its clients. If your team needs expertise in sizing and
# analysing a small paired evaluation, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Prints the data/pilot.csv row for one completed run on the frozen exam.

    pilot_row.py REPORT --arm NAME --description TEXT --power P --power-fpr F

REPORT is the output of `splinter exam --json` or a log that contains it.
Every field is read from the report's `ran` object; the counts that can be
recomputed from its per-task records (right, invented, discordant tasks,
family wins and losses) are recomputed and must agree with the report's
summary, or the script refuses the report. POWER and FPR are the power and
false-positive rate that `splinter exam-set power --from-report REPORT
--families 39 --tasks-per-family 6 --json` prints under `planned`.
Standard library only.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict

HEADER = (
    "arm,description,state,families,tasks,"
    "base_right,prompted_right,cand_right,persona_right,"
    "base_invented,prompted_invented,cand_invented,persona_invented,"
    "base_chars,prompted_chars,cand_chars,persona_chars,"
    "persona_only,prompted_only,family_wins,family_losses,"
    "diff_points,ci_low,ci_high,p_tasks,p_families,"
    "lm_tasks,lm_persona_only,lm_prompted_only,lm_p,"
    "discordance,icc,mdd_points,hard_wrong,hard_passed,power_39,power_39_fpr,"
    "p_cand_vs_prompted,p_persona_vs_base,p_persona_vs_cand,p_cand_vs_base,p_prompted_vs_base,"
    "holm_cand_vs_prompted,holm_persona_vs_base,holm_persona_vs_cand,holm_cand_vs_base,holm_prompted_vs_base,"
    "run"
).split(",")

ARM_KEYS = {"base": "base", "prompted": "prompted", "candidate": "cand", "candidate-persona": "persona"}
SECONDARIES = {
    ("candidate", "prompted"): "cand_vs_prompted",
    ("candidate-persona", "base"): "persona_vs_base",
    ("candidate-persona", "candidate"): "persona_vs_cand",
    ("candidate", "base"): "cand_vs_base",
    ("prompted", "base"): "prompted_vs_base",
}


def load_report(path):
    text = open(path, encoding="utf-8").read()
    start = text.find("\n{\n") + 1 if "\n{\n" in text else text.index("{")
    doc = json.JSONDecoder().raw_decode(text[start:])[0]
    return doc.get("ran", doc), doc.get("run", "")


def verified(what, logged, recomputed):
    if logged != recomputed:
        sys.exit(f"{what}: report says {logged}, its records give {recomputed}")
    return logged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report")
    ap.add_argument("--arm", required=True, help="short run name")
    ap.add_argument("--description", required=True)
    ap.add_argument("--power", required=True, type=float)
    ap.add_argument("--power-fpr", required=True, type=float)
    args = ap.parse_args()

    ran, run_id = load_report(args.report)
    records = ran["records"]
    first = lambda r, arm: r["arms"][arm][0]
    families = defaultdict(int)
    only = {"persona": 0, "prompted": 0}
    out = {"arm": args.arm, "description": args.description, "state": "completed", "run": run_id}
    out["families"] = verified("families", ran["families"], len({r["family"] for r in records}))
    out["tasks"] = verified("tasks", ran["tasks"], len(records))
    for summary in ran["arms"]:
        key = ARM_KEYS[summary["arm"]]
        right = sum(first(r, summary["arm"])["judged"] for r in records)
        invented = sum(first(r, summary["arm"])["grounded"] is False for r in records)
        out[f"{key}_right"] = verified(f"{key} right", summary["right"], right)
        out[f"{key}_invented"] = verified(f"{key} invented", summary["invented"], invented)
        out[f"{key}_chars"] = f"{summary['mean_chars']:.1f}"
    for r in records:
        a, b = first(r, "candidate-persona")["judged"], first(r, "prompted")["judged"]
        families[r["family"]] += int(a) - int(b)
        only["persona"] += a and not b
        only["prompted"] += b and not a
    primary = next(c for c in ran["comparisons"] if (c["first"], c["second"]) == ("candidate-persona", "prompted"))
    out["persona_only"] = verified("persona-only tasks", primary["first_only"], only["persona"])
    out["prompted_only"] = verified("prompted-only tasks", primary["second_only"], only["prompted"])
    out["family_wins"] = sum(v > 0 for v in families.values())
    out["family_losses"] = sum(v < 0 for v in families.values())
    out["diff_points"] = f"{100 * primary['difference']['mean']:.2f}"
    out["ci_low"] = f"{100 * primary['difference']['low']:.2f}"
    out["ci_high"] = f"{100 * primary['difference']['high']:.2f}"
    out["p_tasks"] = f"{primary['p_tasks']:.8f}"
    out["p_families"] = f"{primary['p_families']:.8f}"
    lm = primary["length_matched"]
    out.update(lm_tasks=lm["tasks"], lm_persona_only=lm["first_only"], lm_prompted_only=lm["second_only"], lm_p=f"{lm['p_value']:.8f}")
    out["discordance"] = f"{primary['power']['discordance']:.6f}"
    out["icc"] = f"{primary['power']['icc']:.6f}"
    out["mdd_points"] = f"{100 * primary['power']['minimum_detectable_difference']:.2f}"
    out["hard_wrong"] = ran["hard_controls"]["wrong_answers"]
    out["hard_passed"] = ran["hard_controls"]["passed"]
    out["power_39"] = f"{args.power:.4f}"
    out["power_39_fpr"] = f"{args.power_fpr:.4f}"
    for c in ran["comparisons"]:
        key = SECONDARIES.get((c["first"], c["second"]))
        if key:
            out[f"p_{key}"] = f"{c['p_families']:.8f}"
            out[f"holm_{key}"] = f"{c['holm_p']:.8f}"
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow([out[k] for k in HEADER])


if __name__ == "__main__":
    main()
