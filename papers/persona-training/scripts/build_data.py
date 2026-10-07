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


def task_level_range(first_right, second_right, tasks):
    """The smallest and largest one-sided exact sign p, over every split of
    discordant tasks consistent with the marginals, for "first beats second"
    when only each arm's count of successes over the same tasks is known.
    Ignores family clustering; None when the first arm has no net lead."""
    net = first_right - second_right
    if net <= 0:
        return None
    ps = []
    for second_only in range(second_right + 1):
        first_only = net + second_only
        if first_only > first_right or first_only + second_only > tasks:
            continue
        ps.append(upper_tail(first_only, first_only + second_only))
    return min(ps), max(ps)


def fmt_p(p):
    return f"{p:.1e}".replace("e-0", "e-") if p < 1e-3 else f"{p:.3f}"


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
        r"\begin{tabular}{@{}lllrrrrrrrrr@{}}",
        r"\toprule",
        r" & & & & \multicolumn{3}{c}{judged right} & \multicolumn{3}{c}{invented} & \multicolumn{2}{c}{$p$ (family sign)} \\",
        r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-12}",
        r"Run & Set & Judge & Tasks/fam. & base & prompt. & cand. & base & prompt. & cand. & vs base & vs prompt. \\",
        r"\midrule",
    ]
    for e in exams:
        lines.append(
            f"{e['id']} & {e['taskset']} & v{e['judge_version']} & {e['tasks']}/{e['families']} & "
            f"{e['base_right']} & {e['prompted_right']} & {e['candidate_right']} & "
            f"{e['base_invented']} & {e['prompted_invented']} & {e['candidate_invented']} & "
            f"{num(e['p_vs_base'])} & {num(e['p_vs_prompted'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def bounds_table(exams):
    """What each exam's marginals allow a task-level test of the candidate
    against the prompted base to conclude: the range of the one-sided exact
    sign p over every split of discordant tasks the marginals permit."""
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Run & Right (cand.\ / prompt.) & Net tasks & Smallest possible $p$ & Largest possible $p$ \\",
        r"\midrule",
    ]
    for e in exams:
        c, p, t = int(e["candidate_right"]), int(e["prompted_right"]), int(e["tasks"])
        r = task_level_range(c, p, t)
        lo, hi = (fmt_p(r[0]), fmt_p(r[1])) if r else ("--", "--")
        lines.append(f"{e['id']} & {c} / {p} & {c - p:+d} & {lo} & {hi} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def prompt_table(exams):
    """The persona prompt's effect on the base, once per distinct task set:
    the base and the prompted base are the same two arms in every exam of a
    set, so each set is counted once."""
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{judged right} & \multicolumn{2}{c}{invented} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Set & Tasks/fam. & base $\to$ prompt. & $p$ range & base $\to$ prompt. & $p$ range \\",
        r"\midrule",
    ]
    seen = {}
    for e in exams:
        key = e["taskset"]
        arms = tuple(e[k] for k in ("base_right", "prompted_right", "base_invented", "prompted_invented"))
        if key in seen:
            if seen[key] != arms:
                sys.exit(f"task set {key}: base arms differ between exams")
            continue
        seen[key] = arms
        t = int(e["tasks"])
        b, pr, bi, pi = (int(x) for x in arms)
        right = task_level_range(pr, b, t)
        fewer = task_level_range(bi, pi, t)
        rng = lambda r: f"{fmt_p(r[0])}--{fmt_p(r[1])}" if r else "no net gain"
        lines.append(
            f"{key} ({e['subject']}) & {t}/{e['families']} & {b} $\\to$ {pr} & {rng(right)} & "
            f"{bi} $\\to$ {pi} & {rng(fewer)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def training_table(runs):
    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Run & Train rec. & Steps run/budget & Kept & Peak LR & Held-out rec. & Held-out loss & Token acc. \\",
        r"\midrule",
    ]
    for r in runs:
        lr = r["peak_lr"] + ("" if r["lr_source"] == "log" else r"$^{*}$")
        records = r["train_records"] + (f"+{r['replay_records']}" if int(r["replay_records"]) else "")
        lines.append(
            f"{r['run']} & {records} & {r['steps_run']}/{r['steps_budget']} & "
            f"{r['selected_step']} & {lr} & {r['heldout_records']} & "
            f"{float(r['loss_before']):.3f}$\\to${float(r['loss_after']):.3f} & "
            f"{float(r['acc_before']):.3f}$\\to${float(r['acc_after']):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def anchor_table(anchors):
    lines = [
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
        r"\toprule",
        r"Candidate & Items & Both right & Both wrong & Cand.\ only & Base only & Cand.\ acc. & Base acc. & Drop & $p$ \\",
        r"\midrule",
    ]
    for a in anchors:
        cw, bw = int(a["candidate_wins"]), int(a["baseline_wins"])
        items = int(a["items"])
        if int(a["both_right"]) + int(a["both_wrong"]) + cw + bw != items:
            sys.exit(f"{a['id']}: anchor cells do not add up to {items}")
        if abs((int(a["both_right"]) + cw) / items - float(a["candidate_accuracy"])) > 1e-3:
            sys.exit(f"{a['id']}: candidate accuracy does not match its cells")
        drop = float(a["baseline_accuracy"]) - float(a["candidate_accuracy"])
        lines.append(
            f"{a['label']} & {items} & {a['both_right']} & {a['both_wrong']} & {cw} & {bw} & "
            f"{num(a['candidate_accuracy'])} & {num(a['baseline_accuracy'])} & {drop:+.3f} & "
            f"{two_sided(cw, bw):.2f} \\\\"
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
            f"{float(s['design_effect']):.2f} & {float(s['power']):.2f} & "
            f"{float(s['false_positive_rate']):.3f} & {s['note']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def pilot_table(pilots):
    """The table of completed pilot runs on the frozen exam, or nothing when
    none has completed: a run that did not complete is not a result."""
    done = [p for p in pilots if p["state"] == "completed"]
    if not done:
        return ""
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{@{}llrll@{}}",
        r"\toprule",
        r"Arm & Candidate & Families/tasks & Right (cand.\ / prompt.) & Difference, 95\% interval \\",
        r"\midrule",
    ]
    for p in done:
        lines.append(
            f"{p['arm']} & {p['description']} & {p['families']}/{p['tasks']} & "
            f"{p['candidate_right']} / {p['prompted_right']} & "
            f"{float(p['diff_points']):+.1f} [{float(p['ci_low']):+.1f}, {float(p['ci_high']):+.1f}] points \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Completed pilot runs on the frozen exam: deployed candidate against the prompted base, "
        r"paired by task, with the family-clustered 95\% percentile bootstrap interval.}",
        r"\label{tab:pilot}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    exams = rows("exams.csv")
    (OUT / "exams_table.tex").write_text(exam_tables(exams) + "\n", encoding="utf-8")
    anchors = rows("anchor.csv")
    (OUT / "anchor_table.tex").write_text(anchor_table(anchors) + "\n", encoding="utf-8")
    (OUT / "power_table.tex").write_text(power_sim_table(rows("power_simulation.csv")) + "\n", encoding="utf-8")
    (OUT / "bounds_table.tex").write_text(bounds_table(exams) + "\n", encoding="utf-8")
    (OUT / "prompt_table.tex").write_text(prompt_table(exams) + "\n", encoding="utf-8")
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

    # The v7 anchor check: its discordant items, the spread of the paired
    # drop under no change, how often a drop beyond the bound happens by
    # chance, and the paired items a two-point drop needs at 80% power
    # (normal approximation n = (z_{alpha/2} + z_beta)^2 d / delta^2).
    v7 = next(a for a in anchors if a["id"] == "v7")
    cw, bw, items = int(v7["candidate_wins"]), int(v7["baseline_wins"]), int(v7["items"])
    disc, bound = cw + bw, float(v7["bound"])
    exceed = sum(comb(disc, k) for k in range(disc + 1) if (2 * k - disc) / items > bound) / 2**disc
    z = 1.959964 + 0.841621
    sw = lambda n, q: sign_test_power(n, q)
    macros = {
        "PowerSixNine": f"{sw(6, 0.9):.2f}",
        "PowerSevenNine": f"{sw(7, 0.9):.2f}",
        "PowerSixEight": f"{sw(6, 0.8):.2f}",
        "PowerTwentyEight": f"{sw(20, 0.8):.2f}",
        "PowerFortyEight": f"{sw(40, 0.8):.2f}",
        "AnchorDiscordant": str(disc),
        "AnchorPValue": f"{two_sided(cw, bw):.2f}",
        "AnchorSD": f"{disc ** 0.5 / items:.3f}",
        "AnchorExceed": f"{100 * exceed:.0f}",
        "AnchorItemsNeeded": f"{round(z * z * (disc / items) / 0.02**2, -2):,.0f}",
        "PilotsCompleted": str(sum(p["state"] == "completed" for p in pilots)),
    }
    (OUT / "macros.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
