#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned language
# models for its clients. If your team needs expertise in sizing and
# analysing a small paired evaluation, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Appends the rows of one full frozen-exam run to data/full.csv and data/full_comparisons.csv.

    full_row.py REPORT --arm NAME --description TEXT --pilot-report REPORT

REPORT is the output of `splinter exam --json` without `--pilot-families` (or
a log that contains it): every task of the exam that the candidate was not
trained on, four arms. PILOT-REPORT is the report of a pilot run of the same
exam on its first families; the tasks outside it are summarised separately,
because the pilot's tasks were seen when the candidates were chosen.

data/full.csv gets one row per run: the exam, the judge, the four arms and the
primary comparison's tasks outside the pilot, and how many of the pilot's
answers the full run repeated word for word. data/full_comparisons.csv gets
one row per comparison the report made (the primary first, then the five
secondaries). Every count that can be recomputed from the per-task records
(arm totals, discordant tasks, family wins and losses, length-matched tasks,
exact sign tests) is recomputed and must agree with the report's summary, or
the script refuses the report; the interval and the Holm adjustment are the
report's, and the paper's build recomputes the Holm adjustment again.
Standard library only.
"""

import argparse
import csv
from collections import defaultdict
from math import comb
from pathlib import Path

from pilot_row import ARM_KEYS, load_report, verified

DATA = Path(__file__).resolve().parent.parent / "data"
LENGTH_MATCH = 0.25  # answers differing by at most this share of the longer one

RUN_HEADER = (
    "arm,description,candidate,run,exam,families,tasks,tasks_in_exam,families_trained_on,"
    "judge_version,judge_controls,precision_pass,precision_fail,hard_wrong,hard_passed,"
    "base_right,prompted_right,cand_right,persona_right,"
    "base_invented,prompted_invented,cand_invented,persona_invented,"
    "base_chars,prompted_chars,cand_chars,persona_chars,"
    "discordance,icc,mdd_points,"
    "rest_tasks,rest_families,rest_persona_right,rest_prompted_right,"
    "rest_persona_only,rest_prompted_only,rest_family_wins,rest_family_losses,"
    "rest_p_tasks,rest_p_families,pilot_answers_repeated,pilot_answers_total"
).split(",")
COMPARISON_HEADER = (
    "arm,first,second,first_only,second_only,family_wins,family_losses,"
    "diff_points,ci_low,ci_high,p_tasks,p_families,"
    "lm_tasks,lm_first_only,lm_second_only,lm_p,holm_p"
).split(",")
PRIMARY = ("candidate-persona", "prompted")


def upper_tail(wins, n):
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(wins, n + 1)) / 2**n


def greedy(record, arm):
    return record["arms"][arm][0]


class Tally:
    """Paired counts of one comparison over a set of task records."""

    def __init__(self, records, first, second):
        self.tasks = len(records)
        self.first_right = self.second_right = 0
        self.first_only = self.second_only = 0
        self.matched = [0, 0, 0]  # tasks, first only, second only
        by_family = defaultdict(int)
        for r in records:
            a, b = greedy(r, first), greedy(r, second)
            self.first_right += a["judged"]
            self.second_right += b["judged"]
            self.first_only += a["judged"] and not b["judged"]
            self.second_only += b["judged"] and not a["judged"]
            by_family[r["family"]] += int(a["judged"]) - int(b["judged"])
            if abs(a["chars"] - b["chars"]) <= LENGTH_MATCH * max(a["chars"], b["chars"]):
                self.matched[0] += 1
                self.matched[1] += a["judged"] and not b["judged"]
                self.matched[2] += b["judged"] and not a["judged"]
        self.families = len(by_family)
        self.family_wins = sum(v > 0 for v in by_family.values())
        self.family_losses = sum(v < 0 for v in by_family.values())
        self.p_tasks = upper_tail(self.first_only, self.first_only + self.second_only)
        self.p_families = upper_tail(self.family_wins, self.family_wins + self.family_losses)
        self.lm_p = upper_tail(self.matched[1], self.matched[1] + self.matched[2])


def comparison_row(arm, c, tally):
    name = f"{c['first']} vs {c['second']}"
    for field, mine in (
        ("tasks", tally.tasks),
        ("first_only", tally.first_only),
        ("second_only", tally.second_only),
    ):
        verified(f"{name}: {field}", c[field], mine)
    lm = c["length_matched"]
    verified(f"{name}: length-matched tasks", lm["tasks"], tally.matched[0])
    verified(f"{name}: length-matched first only", lm["first_only"], tally.matched[1])
    verified(f"{name}: length-matched second only", lm["second_only"], tally.matched[2])
    for field, logged, mine in (
        ("p_tasks", c["p_tasks"], tally.p_tasks),
        ("p_families", c["p_families"], tally.p_families),
        ("length-matched p", lm["p_value"], tally.lm_p),
    ):
        if abs(logged - mine) > 1e-9:
            raise SystemExit(f"{name}: {field} is {logged} in the report, {mine} from its records")
    d = c["difference"]
    return {
        "arm": arm, "first": c["first"], "second": c["second"],
        "first_only": tally.first_only, "second_only": tally.second_only,
        "family_wins": tally.family_wins, "family_losses": tally.family_losses,
        "diff_points": f"{100 * d['mean']:.2f}", "ci_low": f"{100 * d['low']:.2f}", "ci_high": f"{100 * d['high']:.2f}",
        "p_tasks": f"{tally.p_tasks:.8f}", "p_families": f"{tally.p_families:.8f}",
        "lm_tasks": tally.matched[0], "lm_first_only": tally.matched[1], "lm_second_only": tally.matched[2],
        "lm_p": f"{tally.lm_p:.8f}",
        "holm_p": "" if c["holm_p"] is None else f"{c['holm_p']:.8f}",
    }


def append(name, header, rows):
    path = DATA / name
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in rows:
            writer.writerow([row[k] for k in header])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report")
    ap.add_argument("--arm", required=True, help="short run name")
    ap.add_argument("--description", required=True)
    ap.add_argument("--pilot-report", required=True)
    args = ap.parse_args()

    ran, run_id = load_report(args.report)
    pilot, _ = load_report(args.pilot_report)
    if pilot["exam"] != ran["exam"]:
        raise SystemExit("the pilot report is of a different exam")
    records = ran["records"]
    verified("tasks", ran["tasks"], len(records))
    verified("families", ran["families"], len({r["family"] for r in records}))
    pilot_tasks = {r["task"] for r in pilot["records"]}
    rest = [r for r in records if r["task"] not in pilot_tasks]
    if len(records) - len(rest) != len(pilot_tasks):
        raise SystemExit("the pilot's tasks are not all among the full run's")

    calibration = ran["judge"]["calibration"]
    out = {
        "arm": args.arm, "description": args.description, "candidate": ran["candidate"], "run": run_id,
        "exam": ran["exam"], "families": ran["families"], "tasks": ran["tasks"],
        "tasks_in_exam": ran["tasks_in_exam"], "families_trained_on": ran["families_trained_on"],
        "judge_version": calibration["producer"]["version"], "judge_controls": calibration["n"],
        "precision_pass": f"{calibration['precision_pass']:.4f}", "precision_fail": f"{calibration['precision_fail']:.4f}",
        "hard_wrong": ran["hard_controls"]["wrong_answers"], "hard_passed": ran["hard_controls"]["passed"],
    }
    for summary in ran["arms"]:
        key = ARM_KEYS[summary["arm"]]
        right = sum(greedy(r, summary["arm"])["judged"] for r in records)
        invented = sum(greedy(r, summary["arm"])["grounded"] is False for r in records)
        out[f"{key}_right"] = verified(f"{key} right", summary["right"], right)
        out[f"{key}_invented"] = verified(f"{key} invented", summary["invented"], invented)
        out[f"{key}_chars"] = f"{summary['mean_chars']:.1f}"

    comparisons = [
        comparison_row(args.arm, c, Tally(records, c["first"], c["second"])) for c in ran["comparisons"]
    ]
    if (comparisons[0]["first"], comparisons[0]["second"]) != PRIMARY or len(comparisons) != 6:
        raise SystemExit("the report lacks the primary comparison or a secondary one")
    power = ran["comparisons"][0]["power"]
    out.update(
        discordance=f"{power['discordance']:.6f}", icc=f"{power['icc']:.6f}",
        mdd_points=f"{100 * power['minimum_detectable_difference']:.2f}",
    )
    t = Tally(rest, *PRIMARY)
    out.update(
        rest_tasks=t.tasks, rest_families=t.families, rest_persona_right=t.first_right,
        rest_prompted_right=t.second_right, rest_persona_only=t.first_only, rest_prompted_only=t.second_only,
        rest_family_wins=t.family_wins, rest_family_losses=t.family_losses,
        rest_p_tasks=f"{t.p_tasks:.8f}", rest_p_families=f"{t.p_families:.8f}",
    )
    # Greedy decoding should repeat a pilot's answers word for word.
    full_by_task = {r["task"]: r for r in records}
    repeated = [
        greedy(full_by_task[r["task"]], arm)["text"] == greedy(r, arm)["text"]
        for r in pilot["records"]
        for arm in r["arms"]
    ]
    out.update(pilot_answers_repeated=sum(repeated), pilot_answers_total=len(repeated))
    append("full.csv", RUN_HEADER, [out])
    append("full_comparisons.csv", COMPARISON_HEADER, comparisons)


if __name__ == "__main__":
    main()
