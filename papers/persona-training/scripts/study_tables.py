#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned language
# models for its clients. If your team needs expertise in reporting a
# replicated, pre-registered evaluation without hand-typed numbers, you can
# procure our services by sending an email to info@swedishembedded.com.

"""Builds the replication study's tables and macros from its report.

usage: study_tables.py REPORT_DIR OUT_DIR

REPORT_DIR holds the CSV files `synthetic-author study report` writes
(accuracy, across_authors, gains, spread, corpora, given_up, unmeasured). The
statistics are the report's own (the sign tests across authors with Holm's
correction, the item-clustered intervals); this script lays them out and checks
that the files agree with one another: the same authors everywhere, one
primary comparison, every comparison's author counts adding up, and each
author's tuned accuracy equal to the mean over its training seeds. OUT_DIR
receives study_macros.tex and the study_*.tex tables.

Authors are lettered A, B, ... in the order corpora.csv lists them, which is
the order the frozen protocol drew them.
"""

import csv
import string
import sys
from pathlib import Path

from tex import GAP, fmt3, fmt_p, fmt_points, table, tex_name, thousands

# The arms the accuracy table shows, in reading order: the trivial baselines,
# the prompted arms, retrieval, the adapter and its ablations.
ARM_ORDER = [
    "first-listed", "majority-option", "base", "named", "principles",
    "principles-verdicts", "retrieval-2", "retrieval", "retrieval-12",
    "retrieval-records-2", "retrieval-records-6", "retrieval-records-12",
    "tuned-reader-only", "tuned-fixed-wording-only", "tuned",
]
TOLERANCE = 1e-9


def fail(message):
    sys.exit(f"study_tables.py: {message}")


def read(report, name):
    path = report / name
    if not path.is_file():
        fail(f"{path} is missing; pass the report/ directory `study report` wrote")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(value, what):
    try:
        return float(value)
    except ValueError:
        fail(f"{what}: {value!r} is not a number")


class Study:
    """The report's tables, checked against one another."""

    def __init__(self, report):
        self.corpora = read(report, "corpora.csv")
        self.accuracy = read(report, "accuracy.csv")
        self.across = read(report, "across_authors.csv")
        self.gains = read(report, "gains.csv")
        self.spread = read(report, "spread.csv")
        self.given_up = read(report, "given_up.csv")
        self.unmeasured = read(report, "unmeasured.csv")
        self.authors = [r["author"] for r in self.corpora]
        if not self.authors or len(set(self.authors)) != len(self.authors):
            fail(f"corpora.csv lists {len(self.authors)} authors, {len(set(self.authors))} distinct")
        self.names = {r["author"]: r["name"] for r in self.corpora}
        self.check()

    def letter(self, author):
        return string.ascii_uppercase[self.authors.index(author)]

    def check(self):
        known = set(self.authors)
        for name, rows in (("accuracy.csv", self.accuracy), ("gains.csv", self.gains),
                           ("spread.csv", self.spread)):
            stray = {r["author"] for r in rows} - known
            if stray:
                fail(f"{name} names authors corpora.csv does not list: {sorted(stray)}")
        primary = [r for r in self.across if r["primary"] == "true"]
        if len(primary) != 1:
            fail(f"across_authors.csv marks {len(primary)} comparisons primary, not one")
        for r in self.across:
            counted = sum(int(r[k]) for k in ("authors_improved", "authors_worse", "authors_tied"))
            if counted != int(r["authors_measured"]):
                fail(f"{r['candidate']} against {r['reference']}: {counted} authors counted, {r['authors_measured']} measured")
            if int(r["authors_measured"]) > len(self.authors):
                fail(f"{r['candidate']} against {r['reference']}: more authors measured than the study has")
        for author in self.authors:
            seeds = [number(r["rate"], "accuracy.csv rate") for r in self.accuracy
                     if r["author"] == author and r["arm"] == "tuned" and r["training_seed"]]
            row = self.spread_row(author, "tuned")
            if seeds and row and abs(sum(seeds) / len(seeds) - number(row["mean"], "spread.csv mean")) > TOLERANCE:
                fail(f"author {author}: spread.csv's tuned mean differs from the mean of its seeds in accuracy.csv")

    def spread_row(self, author, arm):
        rows = [r for r in self.spread if r["author"] == author and r["arm"] == arm]
        return rows[0] if rows else None

    def primary(self):
        return next(r for r in self.across if r["primary"] == "true")

    def rate(self, author, arm):
        """An arm's accuracy for one author; an adapter's is its mean over the
        training seeds it was trained at. None when the arm was not measured."""
        row = self.spread_row(author, arm)
        if row:
            return number(row["mean"], "spread.csv mean")
        rates = [number(r["rate"], "accuracy.csv rate") for r in self.accuracy
                 if r["author"] == author and r["arm"] == arm]
        return sum(rates) / len(rates) if rates else None

    def arms(self):
        measured = {r["arm"] for r in self.accuracy}
        ordered = [a for a in ARM_ORDER if a in measured]
        return ordered + sorted(measured - set(ordered))


def arms_table(study):
    body = []
    for arm in study.arms():
        rates = [r for r in (study.rate(a, arm) for a in study.authors) if r is not None]
        if not rates:
            continue
        body.append(f"{tex_name(arm)} & {len(rates)} & {fmt3(sum(rates) / len(rates))} & "
                    f"{fmt3(min(rates))} & {fmt3(max(rates))}")
        if arm in ("majority-option", "principles-verdicts", "retrieval-records-12"):
            body[-1] += GAP
    return table("@{}lrrrr@{}", "Arm & Authors & Mean accuracy & Lowest & Highest", body)


def authors_table(study):
    """Each author's accuracy under the arms of the primary comparison and the
    paired gain between them, with its item-clustered interval."""
    p = study.primary()
    cand, ref = p["candidate"], p["reference"]
    body = []
    for author in study.authors:
        gain = [r for r in study.gains if r["author"] == author
                and r["candidate"] == cand and r["reference"] == ref]
        rc, rr = study.rate(author, cand), study.rate(author, ref)
        if not gain or rc is None or rr is None:
            body.append(f"{study.letter(author)} & {study.names[author]} & \\multicolumn{{5}}{{c}}{{not measured}}")
            continue
        g = gain[0]
        body.append(
            f"{study.letter(author)} & {study.names[author]} & {fmt3(rr)} & {fmt3(rc)} & "
            f"{fmt_points(number(g['gain'], 'gain'))} [{fmt_points(number(g['low'], 'low'))}, {fmt_points(number(g['high'], 'high'))}] & "
            f"{g['items_won']}/{g['items_lost']} & {fmt_p(number(g['p_items'], 'p_items'))}")
    header = (f"& Author & {tex_name(ref)} & {tex_name(cand)} & Gain, points [95\\% CI] & "
              "Items won/lost & $p$")
    return table("@{}llrrlrr@{}", header, body)


def comparisons_table(study):
    """Every comparison the protocol named: the confirmatory family with
    Holm's correction first, then the exploratory ones, raw."""
    body = []
    rows = sorted(study.across, key=lambda r: (r["confirmatory"] != "true", r["primary"] != "true"))
    for i, r in enumerate(rows):
        holm = fmt_p(number(r["p_holm"], "p_holm")) if r["confirmatory"] == "true" and r["p_holm"] else "--"
        p = fmt_p(number(r["p"], "p")) if r["p"] else "--"
        mean = fmt_points(number(r["mean_gain"], "mean_gain")) if r["mean_gain"] else "--"
        spread = (f"[{fmt_points(number(r['min_gain'], 'min_gain'))}, {fmt_points(number(r['max_gain'], 'max_gain'))}]"
                  if r["min_gain"] else "")
        mark = r"\textbf{yes}" if r["significant"] == "true" else "no"
        if r["confirmatory"] != "true":
            mark = "--"
        line = (f"{tex_name(r['candidate'])} vs {tex_name(r['reference'])} & "
                f"{r['authors_improved']}/{r['authors_worse']}/{r['authors_tied']} of {r['authors_measured']} & "
                f"{mean} {spread} & {p} & {holm} & {mark}")
        if r["confirmatory"] == "true" and i + 1 < len(rows) and rows[i + 1]["confirmatory"] != "true":
            line += GAP
        body.append(line)
    header = r"Comparison & Better/worse/tied & Mean gain, points [range] & $p$ & Holm $p$ & Confirmed"
    return table("@{}lllrrl@{}", header, body)


def spread_table(study):
    body = []
    for author in study.authors:
        r = study.spread_row(author, "tuned")
        if not r:
            continue
        sd = fmt3(number(r["sd"], "sd")) if r["sd"] else "--"
        body.append(f"{study.letter(author)} & {r['seeds']} & {fmt3(number(r['mean'], 'mean'))} & "
                    f"{fmt3(number(r['min'], 'min'))} & {fmt3(number(r['max'], 'max'))} & {sd}")
    return table("@{}lrrrrr@{}", "Author & Seeds & Mean & Lowest & Highest & SD", body)


def corpora_table(study):
    body = []
    for r in study.corpora:
        body.append(f"{study.letter(r['author'])} & {r['name']} & {thousands(r['articles_planned'])} & "
                    f"{thousands(r['articles_kept'])} & {thousands(r['cases_read'])} & "
                    f"{thousands(r['decisions_kept'])} & {thousands(r['calls'])}")
    header = "& Author & Planned & Kept & Cases read & Decisions kept & Requests"
    return table("@{}llrrrrr@{}", header, body)


def macros(study):
    p = study.primary()
    confirmatory = [r for r in study.across if r["confirmatory"] == "true"]
    seeds = max((int(r["seeds"]) for r in study.spread if r["arm"] == "tuned"), default=0)
    items = {int(r["n"]) for r in study.accuracy if r["arm"] == "base"}
    if len(items) > 1:
        fail(f"the base arm was asked different numbers of items: {sorted(items)}")
    values = {
        "StudyAuthors": len(study.authors),
        "StudyTrainingSeeds": seeds,
        "StudyItemsPerAuthor": items.pop() if items else 0,
        "StudyArticlesKept": thousands(sum(int(r["articles_kept"]) for r in study.corpora)),
        "StudyDecisionsKept": thousands(sum(int(r["decisions_kept"]) for r in study.corpora)),
        "StudyPrimaryCandidate": tex_name(p["candidate"]),
        "StudyPrimaryReference": tex_name(p["reference"]),
        "StudyPrimaryImproved": p["authors_improved"],
        "StudyPrimaryMeasured": p["authors_measured"],
        "StudyPrimaryP": fmt_p(number(p["p"], "p")) if p["p"] else "--",
        "StudyPrimaryPHolm": fmt_p(number(p["p_holm"], "p_holm")) if p["p_holm"] else "--",
        "StudyPrimaryMeanGain": fmt_points(number(p["mean_gain"], "mean_gain")) if p["mean_gain"] else "--",
        "StudyConfirmatory": len(confirmatory),
        "StudyConfirmed": sum(r["significant"] == "true" for r in confirmatory),
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()]
    for author in study.authors:
        lines.append(f"\\newcommand{{\\StudyAuthorName{study.letter(author)}}}{{{study.names[author]}}}")
        lines.append(f"\\newcommand{{\\StudyAuthorSeed{study.letter(author)}}}{{{author}}}")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 3:
        fail("usage: study_tables.py REPORT_DIR OUT_DIR")
    study = Study(Path(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "study_macros.tex": macros(study),
        "study_arms.tex": arms_table(study),
        "study_authors.tex": authors_table(study),
        "study_comparisons.tex": comparisons_table(study),
        "study_spread.tex": spread_table(study),
        "study_corpora.tex": corpora_table(study),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
