#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines and reporting for its clients. If your team needs
# expertise in building papers whose every number can be regenerated from the repository, you can procure our
# services by sending an email to info@swedishembedded.com.
"""Builds the macros, tables and figure data of the paper from data/*.csv.

usage: build_data.py DATA_DIR OUT_DIR

Nothing is typed into the paper by hand: the measured counts are in DATA_DIR, the statistics derived from them
(Wilson intervals, the exact one-sided sign test) are computed here, and everything goes to OUT_DIR as LaTeX.
"""
import csv
import sys
from pathlib import Path

from tex import fmt_p, macro, sign_p, wilson

NAMES = {"qwen3-0.6b": "Zero", "qwen3-4b": "Four", "qwen3-8b": "Eight", "qwen3.8-27b": "Big", "qwen3-4b-strong": "FourStrong"}
SIZE = {"qwen3-0.6b": "0.6B", "qwen3-4b": "4B", "qwen3-8b": "8B", "qwen3.8-27b": "27B"}
WORDS = ["One", "Two", "Three", "Four", "Five"]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cap(word):
    return word[0].upper() + word[1:]


def main(data, out):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = []
    exams = rows(data / "exam.csv")
    for r in exams:
        for i, w in enumerate(WORDS):
            m.append(macro(f"exam{NAMES[r['model']]}{cap(r['arm'])}{w}", r[f"k{i + 1}"]))
        lo, hi = wilson(int(r["k5"]), 100)
        m.append(macro(f"exam{NAMES[r['model']]}{cap(r['arm'])}FiveLo", f"{100 * lo:.0f}"))
        m.append(macro(f"exam{NAMES[r['model']]}{cap(r['arm'])}FiveHi", f"{100 * hi:.0f}"))
    for r in rows(data / "paired.csv"):
        w, l = int(r["wins"]), int(r["losses"])
        key = f"{NAMES[r['model']]}{cap(r['arm'])}"
        m += [macro(f"wins{key}", w), macro(f"losses{key}", l), macro(f"p{key}", fmt_p(sign_p(w, l)))]
    comp = rows(data / "data.csv")
    m.append(macro("dataTrain", sum(int(r["train"]) for r in comp)))
    m.append(macro("dataValidation", sum(int(r["validation"]) for r in comp)))
    for name in ("train.csv", "behaviour.csv"):
        path = data / name
        if not path.exists():
            continue
        for r in rows(path):
            key = NAMES[r["model"]] + cap(r["arm"]) + (cap(r["set"]) if "set" in r else "")
            for c, v in r.items():
                if c not in ("model", "arm", "set"):
                    m.append(macro(f"{name.split('.')[0]}{key}{cap(c.replace('_', ''))}", v))
            if name == "behaviour.csv":
                m.append(macro(f"behaviour{key}LoopPct", f"{100 * int(r['loop_trials']) / int(r['trials']):.0f}"))
                m.append(macro(f"behaviour{key}RepeatPct", f"{100 * float(r['repeat_after_error']):.0f}"))
            if name == "train.csv":
                m.append(macro(f"train{key}Pct", f"{100 * int(r['passed']) / int(r['trials']):.0f}"))
    for r in rows(data / "optimiser.csv"):
        key = NAMES[r["model"]]
        m += [macro(f"optimiser{key}Kl", r["kl_median"]), macro(f"optimiser{key}Clip", f"{100 * float(r['clip_median']):.0f}"),
              macro(f"optimiser{key}Loop", r["loop_percent"]), macro(f"optimiser{key}Repeat", r["repeat_percent"])]
    for r in rows(data / "uncounted.csv"):
        m.append(macro(f"uncounted{NAMES[r['model']]}{cap(r['arm'])}", r["slots"]))
    for r in rows(data / "replication.csv"):
        m.append(macro("replication" + "".join(cap(w) for w in r["name"].split("_")), r["value"]))
    teacher = int(comp[0]["train"])
    m.append(macro("dataTeacherPct", f"{100 * teacher / sum(int(r['train']) for r in comp):.0f}"))
    rec = rows(data / "recovery.csv")
    tot = {k: sum(int(r[k]) for r in rec) for k in ("attempts", "passes", "continued", "solved")}
    for k, v in tot.items():
        m.append(macro(f"recovery{cap(k)}", f"{v:,}".replace(",", "\\,")))
    m.append(macro("recoverySolvedPct", f"{100 * tot['solved'] / tot['continued']:.0f}"))
    m.append(macro("recoveryPassPct", f"{100 * tot['passes'] / tot['attempts']:.0f}"))
    for r in rows(data / "integration.csv"):
        m.append(macro("integration" + "".join(cap(w) for w in r["name"].split("_")), f"{int(r['value']):,}".replace(",", "\\,")))
    (out / "macros.tex").write_text("".join(m))

    # the exam table: one row per model and arm
    lines = []
    for r in exams:
        ks = [int(r[f"k{i}"]) for i in range(1, 6)]
        lines.append(f"{SIZE[r['model']]} & {r['arm']} & " + " & ".join(map(str, ks)) + " \\\\\n")
    (out / "exam-table.tex").write_text("".join(lines))

    # the stage-by-stage exam table: pass@1, solved within five, loop-stopped share, repeated-call share
    beh = {(r["model"], r["arm"]): r for r in rows(data / "behaviour.csv") if r["set"] == "exam"}
    ex = {(r["model"], r["arm"]): r for r in exams}
    stages = [("base", "base model"), ("full", "round-one adapter"), ("rtworef", "round-one adapter$^{\\dagger}$"),
              ("rtwo", "round-two adapter$^{\\dagger}$"), ("pgref", "round-one adapter$^{\\ddagger}$"),
              ("pg", "round-one adapter + loop penalty$^{\\ddagger}$")]
    lines = []
    for model in ("qwen3-0.6b", "qwen3-4b", "qwen3-8b"):
        first = True
        for arm, label in stages:
            if (model, arm) not in ex:
                continue
            e, b = ex[(model, arm)], beh.get((model, arm))
            loops = f"{100 * int(b['loop_trials']) / int(b['trials']):.0f}" if b else "--"
            rep = f"{100 * float(b['repeat_after_error']):.0f}" if b else "--"
            lines.append(f"{SIZE[model] if first else ''} & {label} & {e['k1']} & {e['k5']} & {loops} & {rep} \\\\\n")
            first = False
        lines.append("\\addlinespace\n")
    (out / "stages-table.tex").write_text(
        "\\begin{tabular}{llrrrr}\n\\toprule\nModel & Measured arm & pass@1 & within 5 & loop-stopped (\\%) & repeat after error (\\%) \\\\\n\\midrule\n"
        + "".join(lines[:-1]) + "\\bottomrule\n\\end{tabular}\n")

    # composition of the training set by the model that made the records
    body = "".join(f"{r['source']} & {int(r['train']):,} & {int(r['validation']):,} \\\\\n".replace(",", "\\,") for r in comp)
    total = f"all & {sum(int(r['train']) for r in comp):,} & {sum(int(r['validation']) for r in comp):,} \\\\\n".replace(",", "\\,")
    (out / "data-table.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\nModel that made the record & train & validation \\\\\n\\midrule\n"
        + body + "\\midrule\n" + total + "\\bottomrule\n\\end{tabular}\n"
    )

    # held-out loss curves, one file per model and arm
    if (data / "curve.csv").exists():
        series = {}
        for r in rows(data / "curve.csv"):
            series.setdefault((r["model"], r["arm"]), []).append((r["step"], r["loss"]))
        for (model, arm), pts in series.items():
            with open(out / f"curve-{NAMES[model].lower()}-{arm}.dat", "w") as f:
                f.write("step loss\n" + "".join(f"{a} {b}\n" for a, b in pts))

    # solved-within-k curves, one file per model: k and each arm's count
    for model in NAMES:
        mine = [r for r in exams if r["model"] == model]
        if not mine:
            continue
        with open(out / f"withink-{NAMES[model].lower()}.dat", "w") as f:
            f.write("k " + " ".join(r["arm"] for r in mine) + "\n")
            for i in range(1, 6):
                f.write(f"{i} " + " ".join(r[f"k{i}"] for r in mine) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
