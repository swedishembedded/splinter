#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned language
# models for its clients. If your team needs expertise in sizing and
# analysing a paired evaluation with honest statistics, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Builds everything the paper's tables, figure and numbers show from data/*.csv.

usage: build_data.py [OUT_DIR]

Nothing is typed by hand into the paper: the measured counts live in data/
(arms, cells and paired come from scripts/collect.py), the statistics derived
from them are computed here with the standard library only (Wilson intervals,
exact one-sided binomial sign tests, Newcombe intervals), and the output goes
to OUT_DIR as LaTeX tables, pgfplots data and a macro file. The script refuses
to run when the tables are inconsistent with one another or when a derived
statistic differs from the value recorded when the measurements were taken.
"""

import csv
import sys
from math import comb, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_OUT = ROOT.parent.parent / "target" / "papers" / ".build" / "persona-training" / "gen"

SEEDS = [101, 202, 303]
LETTER = {101: "A", 202: "B", 303: "C"}
ARMS = ["base", "named", "principles", "retrieval", "tuned"]
ARM_LABEL = {
    "base": "base", "named": "named", "principles": "principles",
    "retrieval": "retrieval", "tuned": "tuned",
}
CELLS = ["topic_seen", "topic_new", "combo_seen", "combo_new", "tool_seen", "tool_new"]
CELL_LABEL = {
    "topic_seen": "topic seen", "topic_new": "topic new",
    "combo_seen": "combination seen", "combo_new": "combination new",
    "tool_seen": "tool seen", "tool_new": "tool new",
}
AXES = [("topic", "topics"), ("combo", "combinations"), ("tool", "tools")]
PAIRS = [
    ("tuned", "base"), ("tuned", "named"), ("tuned", "principles"),
    ("tuned", "retrieval"), ("retrieval", "base"), ("principles", "base"),
]
POOLED_PAIRS = PAIRS[:4]
EXAM_ITEMS = 160
CELL_ITEMS = 80
Z = 1.96
GAP = "@@gap"  # marks a table row that is followed by extra vertical space

# Values recorded when the examinations were run, at the precision they were
# recorded: the derived statistics must reproduce them.
RECORDED_WILSON_TUNED = {101: (0.801, 0.907), 202: (0.711, 0.838), 303: (0.718, 0.844)}
# ("lt", bound): p below bound.  ("eq", p, decimals, discordant): p rounded.
RECORDED_P = {
    (101, "tuned", "base"): ("lt", 1e-4),
    (101, "tuned", "named"): ("lt", 1e-4),
    (101, "tuned", "principles"): ("lt", 1e-4),
    (101, "tuned", "retrieval"): ("eq", 0.094, 3, 37),
    (101, "retrieval", "base"): ("lt", 1e-4),
    (101, "principles", "base"): ("eq", 0.0001, 4, None),
    (202, "tuned", "base"): ("lt", 1e-4),
    (202, "tuned", "named"): ("lt", 1e-4),
    (202, "tuned", "principles"): ("lt", 1e-4),
    (202, "tuned", "retrieval"): ("eq", 0.0009, 4, 66),
    (202, "retrieval", "base"): ("lt", 1e-4),
    (202, "principles", "base"): ("eq", 0.0205, 4, None),
    (303, "tuned", "base"): ("lt", 1e-4),
    (303, "tuned", "named"): ("lt", 1e-4),
    (303, "tuned", "principles"): ("lt", 1e-4),
    (303, "tuned", "retrieval"): ("lt", 1e-4),
    (303, "retrieval", "base"): ("lt", 1e-4),
    (303, "principles", "base"): ("eq", 0.674, 3, None),
}
RECORDED_DISCORDANT_TUNED_RETRIEVAL = {101: (23, 37), 202: (46, 66), 303: (57, 73)}


def fail(message):
    sys.exit(f"build_data.py: {message}")


def rows(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upper_tail(wins, n):
    """P(X >= wins) for X ~ Binomial(n, 1/2): the exact one-sided sign test."""
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(wins, n + 1)) / 2**n


def wilson(right, n, z=Z):
    """The Wilson score interval (low, high) of a proportion."""
    p = right / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return centre - half, centre + half


def newcombe_difference(right1, n1, right2, n2):
    """Newcombe's hybrid score interval for p1 - p2, built from the two Wilson
    intervals: (difference, low, high)."""
    p1, p2 = right1 / n1, right2 / n2
    l1, u1 = wilson(right1, n1)
    l2, u2 = wilson(right2, n2)
    d = p1 - p2
    return d, d - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fmt3(x):
    return f"{x:.3f}"


def fmt_p(p):
    """A p-value to three significant digits; very small ones in scientific form."""
    if p < 1e-3:
        mantissa, exponent = f"{p:.1e}".split("e")
        return rf"\ensuremath{{{mantissa}\times10^{{{int(exponent)}}}}}"
    return f"{p:.3g}"


def fmt_points(x):
    """Percentage points with an explicit sign and a typeset minus."""
    text = f"{100 * x:+.1f}"
    if text in ("+0.0", "-0.0"):
        return "0.0"
    return text.replace("-", "$-$")


def thousands(value):
    """An integer with a thin space between groups of three digits."""
    return f"{int(value):,}".replace(",", "\\,")


CORPUS_KEYS = ("planned_articles", "essays", "given_up", "words", "min_words", "max_words",
               "writer_requests", "reader_requests", "hosted_requests", "rejection_log_lines",
               "essays_read", "cases_read", "cases_kept", "fixed_briefings", "records")
GATE_KEYS = ("too_short", "belief_quoted", "brief_copied", "case_missing",
             "wrong_decision", "fact_misstated")
EPISODES_PER_ARTICLE = 3
TRAINING_TOOLS = 3


def check_corpora(corpora):
    """The counts of a corpus run reconcile with one another: every planned
    article was kept or given up, every request was a writing or a reading,
    every writing request either kept an essay or left one line in the
    rejection log, every reading request followed a draft that passed the
    form gates, and every kept decision made one record per training tool for
    each of its briefings."""
    for seed, r in corpora.items():
        n = {k: int(v) for k, v in r.items()}
        checks = [
            (n["essays"] + n["given_up"] == n["planned_articles"], "essays and given up != planned"),
            (n["writer_requests"] + n["reader_requests"] == n["hosted_requests"],
             "writing and reading requests != requests"),
            (n["essays"] + n["rejection_log_lines"] == n["writer_requests"],
             "essays and rejection-log lines != writing requests"),
            (sum(n[k] for k in GATE_KEYS) + n["writer_failed"] + n["reader_failed"]
             == n["rejection_log_lines"], "rejection reasons != rejection-log lines"),
            (n["essays"] + n["wrong_decision"] + n["fact_misstated"] + n["reader_failed"]
             == n["reader_requests"], "outcomes of readings != reading requests"),
            (n["essays_read"] <= n["essays"]
             and n["cases_read"] == EPISODES_PER_ARTICLE * n["essays_read"], "cases read"),
            (n["fixed_briefings"] <= n["cases_kept"] <= n["cases_read"], "decisions kept"),
            (n["records"] == TRAINING_TOOLS * (n["cases_kept"] + n["fixed_briefings"]),
             "records != one per tool and briefing"),
            (n["min_words"] * n["essays"] <= n["words"] <= n["max_words"] * n["essays"], "words"),
        ]
        for ok, what in checks:
            if not ok:
                fail(f"corpus {seed}: {what}")


# ---------------------------------------------------------------- loading

def load():
    arms = {}
    for r in rows("arms.csv"):
        arms[(int(r["seed"]), r["arm"])] = (int(r["right"]), int(r["n"]))
    cells = {}
    for r in rows("cells.csv"):
        cells[(int(r["seed"]), r["arm"], r["cell"])] = (int(r["right"]), int(r["n"]))
    paired = {}
    for r in rows("paired.csv"):
        paired[(int(r["seed"]), r["arm_a"], r["arm_b"])] = (
            int(r["n"]), int(r["only_a"]), int(r["only_b"]))
    return arms, cells, paired


def check(arms, cells, paired):
    for seed in SEEDS:
        for arm in ARMS:
            right, n = arms[(seed, arm)]
            if n != EXAM_ITEMS:
                fail(f"{seed} {arm}: {n} items, expected {EXAM_ITEMS}")
            for axis, _ in AXES:
                seen = cells[(seed, arm, f"{axis}_seen")]
                new = cells[(seed, arm, f"{axis}_new")]
                if seen[1] != CELL_ITEMS or new[1] != CELL_ITEMS:
                    fail(f"{seed} {arm} {axis}: cell sizes {seen[1]}, {new[1]}")
                if seen[0] + new[0] != right:
                    fail(f"{seed} {arm} {axis}: cells do not add up to the arm total")
        for a, b in PAIRS:
            n, only_a, only_b = paired[(seed, a, b)]
            if n != EXAM_ITEMS:
                fail(f"{seed} {a} vs {b}: {n} paired items")
            if arms[(seed, a)][0] - arms[(seed, b)][0] != only_a - only_b:
                fail(f"{seed} {a} vs {b}: discordant items disagree with the right counts")
    for seed in SEEDS:
        low, high = wilson(*arms[(seed, "tuned")])
        recorded = RECORDED_WILSON_TUNED[seed]
        if (round(low, 3), round(high, 3)) != recorded:
            fail(f"{seed}: Wilson interval of the tuned arm {low:.3f}, {high:.3f} != {recorded}")
        for (s, a, b), spec in RECORDED_P.items():
            if s != seed:
                continue
            _, only_a, only_b = paired[(s, a, b)]
            p = upper_tail(only_a, only_a + only_b)
            if spec[0] == "lt":
                if not p < spec[1]:
                    fail(f"{s} {a} over {b}: p = {p} is not below {spec[1]}")
            else:
                _, value, decimals, discordant = spec
                if round(p, decimals) != value:
                    fail(f"{s} {a} over {b}: p = {p} does not round to {value}")
                if discordant is not None and only_a + only_b != discordant:
                    fail(f"{s} {a} over {b}: {only_a + only_b} discordant items != {discordant}")
        for other in ("base", "named", "principles"):
            _, only_a, only_b = paired[(seed, "tuned", other)]
            if not upper_tail(only_a, only_a + only_b) < 1e-10:
                fail(f"{seed}: tuned over {other} is not below 1e-10, the paper states it is")
        _, only_a, only_b = paired[(seed, "retrieval", "base")]
        if not upper_tail(only_a, only_a + only_b) < 1e-6:
            fail(f"{seed}: retrieval over base is not below 1e-6, the paper states it is")
        wins, discordant = RECORDED_DISCORDANT_TUNED_RETRIEVAL[seed]
        _, only_a, only_b = paired[(seed, "tuned", "retrieval")]
        if (only_a, only_a + only_b) != (wins, discordant):
            fail(f"{seed}: tuned over retrieval is {only_a} of {only_a + only_b}")


# ---------------------------------------------------------------- macros

class Macros:
    def __init__(self):
        self.lines = []
        self.seen = set()

    def add(self, name, value):
        if name in self.seen:
            fail(f"macro {name} defined twice")
        self.seen.add(name)
        self.lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    def text(self):
        return "\n".join(self.lines) + "\n"


def cap(arm):
    return arm.capitalize()


def build_macros(arms, cells, paired):
    m = Macros()
    m.add("ExamItems", str(EXAM_ITEMS))
    m.add("CellItems", str(CELL_ITEMS))
    for seed in SEEDS:
        L = LETTER[seed]
        for arm in ARMS:
            right, n = arms[(seed, arm)]
            low, high = wilson(right, n)
            m.add(f"Right{L}{cap(arm)}", str(right))
            m.add(f"Acc{L}{cap(arm)}", fmt3(right / n))
            m.add(f"Lo{L}{cap(arm)}", fmt3(low))
            m.add(f"Hi{L}{cap(arm)}", fmt3(high))
        for a, b in PAIRS:
            _, only_a, only_b = paired[(seed, a, b)]
            p = upper_tail(only_a, only_a + only_b)
            key = f"{cap(a)}{cap(b)}{L}"
            m.add(f"OnlyFirst{key}", str(only_a))
            m.add(f"OnlySecond{key}", str(only_b))
            m.add(f"Disc{key}", str(only_a + only_b))
            m.add(f"P{key}", fmt_p(p))
            m.add(f"Lead{key}", str(only_a - only_b))
            m.add(f"LeadPts{key}", fmt_points((only_a - only_b) / EXAM_ITEMS))
        tuned_cells = [cells[(seed, "tuned", c)][0] / CELL_ITEMS for c in CELLS]
        best_other = max(
            cells[(seed, arm, c)][0] / CELL_ITEMS
            for arm in ("base", "named", "principles") for c in CELLS)
        m.add(f"MinTunedCell{L}", fmt3(min(tuned_cells)))
        m.add(f"MaxPromptedCell{L}", fmt3(best_other))
        if not min(tuned_cells) > best_other:
            fail(f"{seed}: a tuned cell is not above every cell of base, named and principles")
    for a, b in POOLED_PAIRS:
        only_a = sum(paired[(s, a, b)][1] for s in SEEDS)
        only_b = sum(paired[(s, a, b)][2] for s in SEEDS)
        key = f"{cap(a)}{cap(b)}"
        m.add(f"PooledFirst{key}", str(only_a))
        m.add(f"PooledSecond{key}", str(only_b))
        m.add(f"PooledP{key}", fmt_p(upper_tail(only_a, only_a + only_b)))
    # Multiplicity: the number of paired tests run, and what the claims
    # made from them survive.
    tests = {k: upper_tail(v[1], v[1] + v[2]) for k, v in paired.items()}
    m.add("PairedTests", str(len(tests)))
    claimed = [p for (s, a, b), p in tests.items()
               if (a, b) in [("tuned", "base"), ("tuned", "named"), ("tuned", "principles"),
                             ("retrieval", "base")]
               or (a, b) == ("tuned", "retrieval") and s != 101]
    m.add("ClaimedTests", str(len(claimed)))
    worst = max(claimed) * len(tests)
    if worst >= 0.05:
        fail("a claimed difference does not survive a Bonferroni factor over all paired tests")
    m.add("BonferroniClaimedMax", fmt_p(worst))
    m.add("BonferroniPrinciplesA", fmt_p(min(1.0, tests[(101, 'principles', 'base')] * len(tests))))
    m.add("BonferroniPrinciplesB", f"{min(1.0, tests[(202, 'principles', 'base')] * len(tests)):.2f}")
    # Retrieval and principles lifts over the base, in items.
    for seed in SEEDS:
        L = LETTER[seed]
        base = arms[(seed, "base")][0]
        for arm in ("named", "principles", "retrieval", "tuned"):
            m.add(f"Lift{L}{cap(arm)}", str(arms[(seed, arm)][0] - base))
    # Gaps between seen and new cells of the tuned arm.
    for seed in SEEDS:
        L = LETTER[seed]
        for axis, _ in AXES:
            seen = cells[(seed, "tuned", f"{axis}_seen")]
            new = cells[(seed, "tuned", f"{axis}_new")]
            d, low, high = newcombe_difference(*seen, *new)
            m.add(f"Gap{L}{cap(axis)}", fmt_points(d))
            m.add(f"GapLo{L}{cap(axis)}", fmt_points(low))
            m.add(f"GapHi{L}{cap(axis)}", fmt_points(high))
    excluding_zero = [
        (seed, axis) for seed in SEEDS for axis, _ in AXES
        if not newcombe_difference(*cells[(seed, "tuned", f"{axis}_seen")],
                                   *cells[(seed, "tuned", f"{axis}_new")])[1]
        <= 0 <= newcombe_difference(*cells[(seed, "tuned", f"{axis}_seen")],
                                    *cells[(seed, "tuned", f"{axis}_new")])[2]]
    if excluding_zero != [(202, "tool")]:
        fail(f"the paper states that only the tool axis of author 202 excludes zero: {excluding_zero}")
    # Corpora, audit, training, cost.
    corpora = {int(r["seed"]): r for r in rows("corpora.csv")}
    check_corpora(corpora)
    for seed, r in corpora.items():
        L = LETTER[seed]
        for key in CORPUS_KEYS:
            m.add(f"Corpus{L}{key.title().replace('_', '')}", thousands(r[key]))
        m.add(f"Corpus{L}MeanWords", thousands(round(int(r["words"]) / int(r["essays"]))))
        m.add(f"Corpus{L}GateRejections", str(sum(int(r[k]) for k in GATE_KEYS)))
        m.add(f"Corpus{L}FailedRequests", str(int(r["writer_failed"]) + int(r["reader_failed"])))
    m.add("CorpusMinWords", thousands(min(int(r["min_words"]) for r in corpora.values())))
    m.add("CorpusMaxWords", thousands(max(int(r["max_words"]) for r in corpora.values())))
    kept_all = agree_all = 0
    rates = []
    for r in rows("audit.csv"):
        seed = int(r["seed"])
        L = LETTER[seed]
        kept, agree = int(r["kept"]), int(r["agree"])
        if kept != int(corpora[seed]["cases_kept"]):
            fail(f"{seed}: the audit covers {kept} decisions, the corpus kept {corpora[seed]['cases_kept']}")
        if not 0 <= agree <= kept:
            fail(f"{seed}: audit agreement {agree} out of range")
        kept_all, agree_all = kept_all + kept, agree_all + agree
        low, high = wilson(agree, kept)
        rates.append(100 * (kept - agree) / kept)
        m.add(f"AuditAgree{L}", str(agree))
        m.add(f"AuditKept{L}", str(kept))
        m.add(f"AuditRate{L}", f"{100 * agree / kept:.1f}")
        m.add(f"AuditDisagree{L}", str(kept - agree))
        m.add(f"AuditDisagreeRate{L}", f"{100 * (kept - agree) / kept:.1f}")
        m.add(f"AuditLo{L}", f"{100 * low:.1f}")
        m.add(f"AuditHi{L}", f"{100 * high:.1f}")
    m.add("AuditDisagreeRateMin", f"{min(rates):.1f}")
    m.add("AuditDisagreeRateMax", f"{max(rates):.1f}")
    m.add("AuditAgreeAll", str(agree_all))
    m.add("AuditDisagreeAll", str(kept_all - agree_all))
    m.add("AuditKeptAll", str(kept_all))
    m.add("AuditDisagreeRateAll", f"{100 * (kept_all - agree_all) / kept_all:.1f}")
    for r in rows("training.csv"):
        seed = int(r["seed"])
        L = LETTER[seed]
        if int(r["train_records"]) + int(r["held_out_records"]) != int(corpora[seed]["records"]):
            fail(f"{seed}: trained and held-out records do not add up to the records made")
        draws = int(r["steps"]) * int(r["records_per_step"])
        m.add(f"TrainRecords{L}", r["train_records"])
        m.add(f"HeldOutRecords{L}", r["held_out_records"])
        m.add(f"TrainPasses{L}", f"{draws / int(r['train_records']):.2f}")
        m.add(f"TrainHours{L}", f"{int(r['elapsed_at_last_step_s']) / 3600:.1f}")
        m.add(f"TrainLoss{L}", r["loss_at_last_step"])
        if r["learning_rate"]:
            m.add("TrainLearningRate", r["learning_rate"])
        if r["held_out_loss_before"]:
            m.add("HeldOutLossBefore", r["held_out_loss_before"])
            m.add("HeldOutLossAfter", r["held_out_loss_after"])
            m.add("HeldOutAccBefore", r["held_out_token_accuracy_before"])
            m.add("HeldOutAccAfter", r["held_out_token_accuracy_after"])
    for r in rows("cost.csv"):
        L = LETTER[int(r["seed"])]
        m.add(f"CostInput{L}", f"{int(r['estimated_input_tokens']) / 1e6:.2f}")
        m.add(f"CostOutput{L}", f"{int(r['estimated_output_tokens']) / 1e6:.2f}")
    spend = {r["seeds"]: float(r["daily_usage_usd_after"]) - float(r["daily_usage_usd_before"])
             for r in rows("spend.csv")}
    if any(v <= 0 for v in spend.values()):
        fail("a run's spend is not positive")
    m.add("SpendA", f"{spend['101']:.2f}")
    m.add("SpendBC", f"{spend['202+303']:.2f}")
    # Divergence.
    for r in rows("cross.csv"):
        total = int(r["situations"])
        own, other, neither = int(r["own"]), int(r["other"]), int(r["neither"])
        if own + other + neither != total:
            fail(f"cross {r['trained_on']} vs {r['compared_with']}: counts do not add up")
        if not own > max(other, neither):
            fail(f"cross {r['trained_on']} vs {r['compared_with']}: own is not the largest count")
    return m


# ---------------------------------------------------------------- tables

def table(spec, header, body):
    out = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    out += [line.removesuffix(GAP) + r" \\" + (r" \addlinespace" if line.endswith(GAP) else "")
            for line in body]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tex_name(name):
    return name.replace("_", r"\_")


def authors_table():
    body = []
    for r in rows("authors.csv"):
        beliefs = "; ".join(
            f"{tex_name(b.split()[0])} ({b.split()[1]})" for b in r["beliefs"].split("; "))
        body.append(f"{r['seed']} & {r['name']} & {beliefs}")
    return table(r"@{}rll@{}", "Seed & Name & Beliefs (tenet, reading), highest priority first", body)


def corpora_table():
    body = []
    for r in rows("corpora.csv"):
        def n(key):
            return thousands(r[key])
        mean = thousands(round(int(r["words"]) / int(r["essays"])))
        body.append(
            f"{r['seed']} & {n('essays')} of {n('planned_articles')} & {n('words')} ({mean}) & "
            f"{n('writer_requests')} & {n('reader_requests')} & {n('rejection_log_lines')} & "
            f"{n('essays_read')} & {n('cases_kept')} of {n('cases_read')} & {n('records')}")
    return table(
        r"@{}rrrrrrrrr@{}",
        r"Seed & Essays & \shortstack{Words\\(mean)} & \shortstack{Writing\\requests} & "
        r"\shortstack{Reading\\requests} & \shortstack{Rejection-\\log lines} & "
        r"\shortstack{Essays\\mined} & \shortstack{Decisions kept\\of cases read} & Records",
        body)


def rejections_table():
    labels = [("too_short", "too short"), ("belief_quoted", "belief quoted"),
              ("brief_copied", "brief copied"), ("case_missing", "case missing"),
              ("wrong_decision", "decision read otherwise"), ("fact_misstated", "fact read otherwise"),
              ("writer_failed", "writing request failed"), ("reader_failed", "reading request failed")]
    data = rows("corpora.csv")
    body = [f"{label} & " + " & ".join(r[key] for r in data) for key, label in labels]
    body.append("total & " + " & ".join(r["rejection_log_lines"] for r in data))
    header = "Line of the rejection log & " + " & ".join(f"Author {r['seed']}" for r in data)
    return table(r"@{}lrrr@{}", header, body)


def main_table(arms):
    body = []
    for arm in ARMS:
        cellsx = []
        for seed in SEEDS:
            right, n = arms[(seed, arm)]
            low, high = wilson(right, n)
            cellsx.append(f"{right} ({fmt3(right / n)}) [{fmt3(low)}, {fmt3(high)}]")
        body.append(f"{ARM_LABEL[arm]} & " + " & ".join(cellsx))
    return table(r"@{}lccc@{}", "Arm & Author 101 & Author 202 & Author 303", body)


def paired_table(paired):
    header = ("Comparison & \\multicolumn{2}{c}{Author 101} & \\multicolumn{2}{c}{Author 202} & "
              "\\multicolumn{2}{c}{Author 303} & \\multicolumn{2}{c}{Pooled}")
    sub = ("& first--second & $p$ & first--second & $p$ & first--second & $p$ & first--second & $p$")
    body = []
    for a, b in PAIRS:
        parts = []
        for seed in SEEDS:
            _, only_a, only_b = paired[(seed, a, b)]
            parts.append(f"{only_a}--{only_b} & {fmt_p(upper_tail(only_a, only_a + only_b))}")
        if (a, b) in POOLED_PAIRS:
            only_a = sum(paired[(s, a, b)][1] for s in SEEDS)
            only_b = sum(paired[(s, a, b)][2] for s in SEEDS)
            parts.append(f"{only_a}--{only_b} & {fmt_p(upper_tail(only_a, only_a + only_b))}")
        else:
            parts.append("\\multicolumn{2}{c}{}")
        body.append(f"{a} over {b} & " + " & ".join(parts))
    out = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule", header + r" \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
           sub + r" \\", r"\midrule"]
    out += [line + r" \\" for line in body]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def cells_table(cells):
    header = "Author & Arm & " + " & ".join(CELL_LABEL[c] for c in CELLS)
    body = []
    for seed in SEEDS:
        for k, arm in enumerate(ARMS):
            vals = " & ".join(str(cells[(seed, arm, c)][0]) for c in CELLS)
            body.append(f"{seed if k == 0 else ''} & {ARM_LABEL[arm]} & {vals}")
        if seed != SEEDS[-1]:
            body[-1] += GAP
    return table(r"@{}rlrrrrrr@{}", header, body)


def gaps_table(cells):
    header = "Author & " + " & ".join(label for _, label in AXES)
    body = []
    for seed in SEEDS:
        parts = []
        for axis, _ in AXES:
            seen = cells[(seed, "tuned", f"{axis}_seen")]
            new = cells[(seed, "tuned", f"{axis}_new")]
            d, low, high = newcombe_difference(*seen, *new)
            parts.append(f"{fmt_points(d)} [{fmt_points(low)}, {fmt_points(high)}]")
        body.append(f"{seed} & " + " & ".join(parts))
    return table(r"@{}rccc@{}", header, body)


def cross_table():
    header = ("Trained on & Compared with & Items & Its own author & The other author & Neither")
    body = []
    previous = None
    for r in rows("cross.csv"):
        total = int(r["situations"])
        def cellf(key):
            return f"{int(r[key])} ({100 * int(r[key]) / total:.0f}\\%)"
        name = r["trained_on"] if r["trained_on"] != previous else ""
        previous = r["trained_on"]
        kind = " (opposite beliefs)" if r["rival_kind"] == "opposite" else ""
        body.append(
            f"{name} & {r['compared_with']}{kind} & {total} & {cellf('own')} & "
            f"{cellf('other')} & {cellf('neither')}")
    return table(r"@{}llrrrr@{}", header, body)


def audit_table():
    body = []
    names = {r["seed"]: r["name"] for r in rows("authors.csv")}
    for r in rows("audit.csv"):
        kept, agree = int(r["kept"]), int(r["agree"])
        low, high = wilson(agree, kept)
        body.append(
            f"{r['seed']} & {names[r['seed']]} & {kept} & {agree} & "
            f"{100 * agree / kept:.1f}\\% [{100 * low:.1f}, {100 * high:.1f}] & {kept - agree}")
    return table(r"@{}rlrrrr@{}",
                 "Seed & Author & Decisions kept & Equal to the oracle & Share [95\\% Wilson] & Disagreeing",
                 body)


def training_table():
    body = []
    for r in rows("training.csv"):
        def change(before, after):
            return f"{r[before]} $\\to$ {r[after]}" if r[before] else "--"
        body.append(
            f"{r['seed']} & {r['train_records']} & {r['held_out_records']} & "
            f"{thousands(r['elapsed_at_last_step_s'])} & {r['loss_at_last_step']} & "
            f"{change('held_out_loss_before', 'held_out_loss_after')} & "
            f"{change('held_out_token_accuracy_before', 'held_out_token_accuracy_after')}")
    return table(
        r"@{}rrrrrcc@{}",
        r"Seed & \shortstack{Records\\trained on} & \shortstack{Records\\held out} & "
        r"\shortstack{Elapsed at\\step 240 (s)} & \shortstack{Loss at\\step 240} & "
        r"\shortstack{Held-out\\token loss} & \shortstack{Held-out\\token accuracy}",
        body)


# ---------------------------------------------------------------- figure data

def figure_data(arms):
    lines = ["x," + ",".join(f"{a},{a}_minus,{a}_plus" for a in ARMS)]
    for k, seed in enumerate(SEEDS):
        parts = [str(k)]
        for arm in ARMS:
            right, n = arms[(seed, arm)]
            low, high = wilson(right, n)
            p = right / n
            parts += [f"{p:.4f}", f"{p - low:.4f}", f"{high - p:.4f}"]
        lines.append(",".join(parts))
    return "\n".join(lines) + "\n"


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    arms, cells, paired = load()
    check(arms, cells, paired)
    files = {
        "macros.tex": build_macros(arms, cells, paired).text(),
        "tab_authors.tex": authors_table(),
        "tab_corpora.tex": corpora_table(),
        "tab_rejections.tex": rejections_table(),
        "tab_main.tex": main_table(arms),
        "tab_paired.tex": paired_table(paired),
        "tab_cells.tex": cells_table(cells),
        "tab_gaps.tex": gaps_table(cells),
        "tab_cross.tex": cross_table(),
        "tab_audit.tex": audit_table(),
        "tab_training.tex": training_table(),
        "accuracy.csv": figure_data(arms),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")


main()
