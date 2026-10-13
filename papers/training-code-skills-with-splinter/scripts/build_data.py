#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned coding agents
# for its clients. If your team needs expertise in sizing and analysing a
# small paired evaluation of an agent before and after training, you can
# procure our services by sending an email to info@swedishembedded.com.

"""Builds everything the paper's tables and figures show from data/results.csv.

Nothing is typed by hand into the paper. The measured paired counts live in
data/results.csv; the exact one-sided sign test and the design table of how
many discordant exam families a decision needs are computed here with the
standard library only. An experiment without rows is printed as NOT RUN.
"""

import csv
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "generated"
ALPHA = 0.05

# The registered experiments, in the order the paper reports them.
EXPERIMENTS = [
    ("E1", "Plumbing: tools and masks reach the trainer (overfit test)"),
    ("E2", "One repository: adapter against base on frozen exam families"),
    ("E3", "Sequential repositories: gain and retention across rounds"),
    ("E4", "Repository evolution: new revision, continued adapter"),
    ("A1", "Ablation: training on all trajectories, not only verified ones"),
    ("A2", "Ablation: no replay of earlier repositories"),
]


def upper_tail(wins, n):
    """P(X >= wins) for X ~ Binomial(n, 1/2): the exact one-sided sign test."""
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(wins, n + 1)) / 2**n


def critical_wins(n):
    """Fewest wins out of n discordant clusters with one-sided p <= ALPHA, or None."""
    for w in range(n + 1):
        if upper_tail(w, n) <= ALPHA:
            return w
    return None


def power(n, q):
    """P(reject) when each discordant cluster is a win with probability q."""
    w = critical_wins(n)
    if w is None:
        return 0.0
    return sum(comb(n, i) * q**i * (1 - q) ** (n - i) for i in range(w, n + 1))


def read_results():
    path = ROOT / "data" / "results.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")


def design_table():
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Discordant & Wins needed & Smallest $p$ & \multicolumn{3}{c}{Power if $P(\text{win})=$} \\",
        r"clusters $n$ & for $p\le 0.05$ & (all wins) & 0.7 & 0.8 & 0.9 \\",
        r"\midrule",
    ]
    for n in (4, 5, 6, 8, 10, 15, 20, 30):
        w = critical_wins(n)
        need = str(w) if w is not None else "none"
        lines.append(
            f"{n} & {need} & {upper_tail(n, n):.2g} & "
            f"{power(n, 0.7):.2f} & {power(n, 0.8):.2f} & {power(n, 0.9):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("design_table.tex", "\n".join(lines) + "\n")
    curve = ["n,q70,q80,q90"]
    for n in range(2, 41):
        curve.append(f"{n},{power(n, 0.7):.4f},{power(n, 0.8):.4f},{power(n, 0.9):.4f}")
    write("power_curve.csv", "\n".join(curve) + "\n")


def results_table(rows):
    lines = [
        r"\begin{tabular}{llllrrrrl}",
        r"\toprule",
        r"Exp. & Repository & Round & Arm & Families & Wins & Losses & $p$ & Status \\",
        r"\midrule",
    ]
    measured = 0
    for key, title in EXPERIMENTS:
        mine = [r for r in rows if r["experiment"] == key]
        if not mine:
            lines.append(rf"{key} & \multicolumn{{8}}{{l}}{{{title}: \textbf{{NOT RUN}}}} \\")
            continue
        for r in mine:
            wins, losses = int(r["wins"]), int(r["losses"])
            p = upper_tail(wins, wins + losses)
            measured += 1
            lines.append(
                f"{key} & {r['repository']} & {r['round']} & {r['arm']} & {r['exam_families']} "
                f"& {wins} & {losses} & {p:.3f} & {r['status']} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("results_table.tex", "\n".join(lines) + "\n")
    return measured


def main():
    OUT.mkdir(exist_ok=True)
    rows = read_results()
    design_table()
    measured = results_table(rows)
    write(
        "macros.tex",
        f"\\newcommand{{\\measuredrows}}{{{measured}}}\n"
        f"\\newcommand{{\\alphalevel}}{{{ALPHA}}}\n",
    )


if __name__ == "__main__":
    main()
