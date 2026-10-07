#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned language
# models for its clients. If your team needs expertise in sizing and
# analysing a small paired evaluation, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Builds everything the paper's tables and figures show from data/*.csv.

Nothing here is typed by hand into the paper: the measured counts live in
data/, the exact statistics derived from them are computed here (standard
library only), and the output goes to generated/ as LaTeX tables, pgfplots
data and a macro file. To update a result, edit its row in data/ and run
`make`.
"""

import csv
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "generated"


def rows(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upper_tail(wins, n):
    """P(X >= wins) for X ~ Binomial(n, 1/2): the exact one-sided sign test."""
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(wins, n + 1)) / 2**n


def two_sided(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * upper_tail(max(a, b), n))


def sign_test_power(n, q, alpha=0.05):
    """Exact power of the two-sided sign test with n discordant units when
    each is won by the first arm with probability q."""
    reject = [k for k in range(n + 1) if two_sided(k, n - k) <= alpha]
    return sum(comb(n, k) * q**k * (1 - q) ** (n - k) for k in reject)


def num(x, digits=3):
    return f"{float(x):.{digits}f}"


def exam_tables(exams):
    for e in exams:
        for who in ("base", "prompted"):
            logged = float(e[f"p_vs_{who}"])
            computed = upper_tail(int(e[f"wins_vs_{who}"]), int(e[f"disc_vs_{who}"]))
            if abs(logged - computed) > 1e-4:
                sys.exit(f"{e['id']}: logged p {logged} != exact {computed} for {who}")

    lines = [
        r"\begin{tabular}{@{}llrrrrrrrrr@{}}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{judged right} & \multicolumn{3}{c}{invented} & \multicolumn{2}{c}{$p$ (family sign)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(l){10-11}",
        r"Run & Person & Tasks/fam. & base & prompt. & cand. & base & prompt. & cand. & vs base & vs prompt. \\",
        r"\midrule",
    ]
    for e in exams:
        lines.append(
            f"{e['id']} & {e['subject']} & {e['tasks']}/{e['families']} & "
            f"{e['base_right']} & {e['prompted_right']} & {e['candidate_right']} & "
            f"{e['base_invented']} & {e['prompted_invented']} & {e['candidate_invented']} & "
            f"{num(e['p_vs_base'])} & {num(e['p_vs_prompted'])} \\\\"
        )
    pooled = [e for e in exams if e["id"] in ("v6", "ablation", "v7", "adams")]
    tot = lambda k: sum(int(e[k]) for e in pooled)
    lines += [
        r"\midrule",
        f"pooled (v6, ablation, v7, adams) & & {tot('tasks')} & {tot('base_right')} & "
        f"{tot('prompted_right')} & {tot('candidate_right')} & {tot('base_invented')} & "
        f"{tot('prompted_invented')} & {tot('candidate_invented')} & -- & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines), pooled, tot


def bounds_table(exams):
    """What each exam's marginals allow a task-level test to conclude: with
    net extra wins n over the prompted base, the smallest one-sided exact
    sign p over every possible split of discordant tasks is 0.5**n."""
    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Run & Right (cand.\ / prompt.) & Net tasks & Smallest possible task-level $p$ \\",
        r"\midrule",
    ]
    for e in exams:
        net = int(e["candidate_right"]) - int(e["prompted_right"])
        floor = f"{0.5 ** net:.4f}" if net > 0 else "1 (no net gain)" if net == 0 else "1 (net loss)"
        lines.append(f"{e['id']} & {e['candidate_right']} / {e['prompted_right']} & {net:+d} & {floor} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def training_table(runs):
    lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Run & Train rec. & Steps run/budget & Step kept & Peak LR & Held-out rec. & Loss before $\to$ after & Tok.\ acc.\ before $\to$ after \\",
        r"\midrule",
    ]
    for r in runs:
        lr = r["peak_lr"] or "not recorded"
        lines.append(
            f"{r['run']} & {r['train_records']} & {r['steps_run']}/{r['steps_budget']} & {r['selected_step']} & {lr} & "
            f"{r['heldout_records']} & {float(r['loss_before']):.3f} $\\to$ {float(r['loss_after']):.3f} & "
            f"{float(r['acc_before']):.3f} $\\to$ {float(r['acc_after']):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def anchor_table(anchors):
    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Gate & Items & cand.\ only & base only & cand.\ acc. & base acc. & drop & $p$ (2-sided) \\",
        r"\midrule",
    ]
    for a in anchors:
        cw, bw = int(a["candidate_wins"]), int(a["baseline_wins"])
        drop = float(a["baseline_accuracy"]) - float(a["candidate_accuracy"])
        lines.append(
            f"{a['label']} & {a['items']} & {cw} & {bw} & {num(a['candidate_accuracy'])} & "
            f"{num(a['baseline_accuracy'])} & {drop:+.3f} & {two_sided(cw, bw):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def power_sim_table(sims):
    lines = [
        r"\begin{tabular}{@{}rrrrrrl@{}}",
        r"\toprule",
        r"Families & Tasks/fam. & Effect & Design eff. & Power & False pos. & Design \\",
        r"\midrule",
    ]
    for s in sims:
        lines.append(
            f"{s['families']} & {s['tasks_per_family']} & {float(s['effect']):.2f} & "
            f"{float(s['design_effect']):.2f} & {float(s['power']):.3f} & "
            f"{float(s['false_positive_rate']):.3f} & {s['note']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def pilot_table(pilots):
    lines = [
        r"\begin{tabular}{@{}llrrl@{}}",
        r"\toprule",
        r"Arm & Candidate & Families/tasks & Result & State \\",
        r"\midrule",
    ]
    for p in pilots:
        if p["state"] == "completed":
            result = (
                f"{p['candidate_right']} vs {p['prompted_right']} right; "
                f"{float(p['diff_points']):+.1f} pts "
                f"[{float(p['ci_low']):+.1f}, {float(p['ci_high']):+.1f}]"
            )
        else:
            result = "not measured"
        lines.append(
            f"{p['arm']} & {p['description']} & {p['families']}/{p['tasks']} & {result} & {p['state']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    exams = rows("exams.csv")
    table, pooled, tot = exam_tables(exams)
    (OUT / "exams_table.tex").write_text(table + "\n", encoding="utf-8")
    (OUT / "anchor_table.tex").write_text(anchor_table(rows("anchor.csv")) + "\n", encoding="utf-8")
    (OUT / "power_table.tex").write_text(power_sim_table(rows("power_simulation.csv")) + "\n", encoding="utf-8")
    (OUT / "bounds_table.tex").write_text(bounds_table(exams) + "\n", encoding="utf-8")
    (OUT / "training_table.tex").write_text(training_table(rows("training.csv")) + "\n", encoding="utf-8")
    pilots = rows("pilot.csv")
    (OUT / "pilot_table.tex").write_text(pilot_table(pilots) + "\n", encoding="utf-8")

    # pgfplots data: exact power of the family-level sign test.
    with open(OUT / "sign_power.csv", "w", encoding="utf-8") as f:
        f.write("n,q060,q070,q080,q090\n")
        for n in range(1, 61):
            f.write(f"{n}," + ",".join(f"{sign_test_power(n, q):.4f}" for q in (0.6, 0.7, 0.8, 0.9)) + "\n")

    # pgfplots data: the learning-rate schedules and monitoring losses.
    for run in ("v7", "e4"):
        with open(OUT / f"curve_{run}.csv", "w", encoding="utf-8") as f:
            f.write("step,lr,monitor\n")
            for r in rows("curves.csv"):
                if r["run"] == run:
                    f.write(f"{r['step']},{float(r['lr']):.6g},{r['monitor_loss']}\n")

    # pgfplots data: the share judged right per arm and exam.
    with open(OUT / "arms.csv", "w", encoding="utf-8") as f:
        f.write("x,label,base,prompted,candidate\n")
        for k, e in enumerate(exams):
            n = int(e["tasks"])
            f.write(
                f"{k},{e['id']},{int(e['base_right'])/n:.4f},"
                f"{int(e['prompted_right'])/n:.4f},{int(e['candidate_right'])/n:.4f}\n"
            )

    sw = lambda n, q: sign_test_power(n, q)
    macros = {
        "PowerSixNine": f"{sw(6, 0.9):.2f}",
        "PowerSevenNine": f"{sw(7, 0.9):.2f}",
        "PowerSixEight": f"{sw(6, 0.8):.2f}",
        "PowerTwentyEight": f"{sw(20, 0.8):.2f}",
        "PowerFortyEight": f"{sw(40, 0.8):.2f}",
        "PooledTasks": str(tot("tasks")),
        "PooledCand": str(tot("candidate_right")),
        "PooledPrompted": str(tot("prompted_right")),
        "PooledBase": str(tot("base_right")),
        "PooledCandInv": str(tot("candidate_invented")),
        "PooledPromptedInv": str(tot("prompted_invented")),
        "PooledBaseInv": str(tot("base_invented")),
        "PooledCandPct": f"{100*tot('candidate_right')/tot('tasks'):.0f}",
        "PooledPromptedPct": f"{100*tot('prompted_right')/tot('tasks'):.0f}",
        "PooledBasePct": f"{100*tot('base_right')/tot('tasks'):.0f}",
        "AnchorPValue": f"{two_sided(7, 10):.2f}",
        "AnchorSD": f"{(17 ** 0.5) / 110:.3f}",
    }
    (OUT / "macros.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
