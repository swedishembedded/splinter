#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of self-improving model
# pipelines for its clients. If your team needs expertise in turning raw run
# records into tables whose every number is checked against the claims they
# support, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Builds every number, table and the limitations text of the paper from data/runs.

    build_data.py --out DIR     write macros.tex, tables and limitations.tex into DIR
    build_data.py --check       verify the ledger against the runs and the text

Nothing measured is typed by hand into the paper. A MEASURED claim names its
evidence as `runs/<file>#<dotted.path>` or `runs/<file>#<metric>`; `--check`
resolves each against the raw record, requires the claim's text to contain the
value, and requires that the paper's text cites the claim. An experiment whose
record is absent is printed as NOT MEASURED, never as zero.

Raw records (data/runs):
  e1/tasks.json          the task report of `tasks generate` (E1)
  e1/command.txt, e1/time.txt
  e2/summary.json        `flywheel_cycle.py summary` (per fact answers per step)
  e2/raw/*.json          each splinter command's output (releases, gate reports)
  e2/build.json          seed of the invented facts
  e2/exits.json          exit status of the two learn commands and the rollback
  qual/*                 training-substrate qualification files, copied verbatim
  throughput/*.log       a later probe of the 27B architecture's training step rate
  ../results.csv         paired outcomes of the registered coding-task experiments (none yet)

Claim kinds (data/claims.csv): STRUCTURE (verify_structure.py), MEASURED and COMPUTED (a
metric of this script, whose value the claim text must contain), REGISTERED (a phrase the
dated registration files contain), PLANNED (an experiment with no result row yet; false as
soon as one exists) and CITED (a key of references.bib).
"""

import argparse
import csv
import json
import re
import sys
import tomllib
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import mcnemar_exact, wilson  # noqa: E402
from classify_rejections import classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
SECTIONS = ROOT / "sections"


def load(rel):
    path = RUNS / rel
    return json.loads(path.read_text()) if path.exists() else None


def dig(data, dotted):
    for part in dotted.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def pct(k, n):
    return f"{100 * k / n:.0f}"


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * lo:.0f}--{100 * hi:.0f}\\%)"


# ---------------------------------------------------------------- metrics
def e1_metrics():
    d = load("e1/tasks.json")
    if d is None:
        return {}
    m = {"e1.tasks": d["tasks"], "e1.parts": d["parts"], "e1.parts_total": d["parts_total"],
         "e1.elapsed_secs": d["elapsed_secs"]}
    for kind, v in d["per_kind"].items():
        m[f"e1.admitted.{kind}"] = v["admitted"]
        m[f"e1.rejected.{kind}"] = v["rejected"]
    for reason, n in d["rejected"].items():
        m[f"e1.reason.{reason}"] = n
    m["e1.rejected_total"] = sum(d["rejected"].values())
    return m


W2_E1 = [("e1a", "Brain pages", "Qwen3-4B"), ("e1b", "Sven pages", "Qwen3-4B"),
         ("e1c", "Brain pages", "Qwen3-8B")]
CLASSES = ["absent", "apart", "unsourced", "markup"]


def e1w2_metrics():
    """The funnel of each second-wave E1 run: proposals, admissions, rejections by reason,
    and the `no_subject` rejections read against the rule."""
    m = {}
    for tag, _, _ in W2_E1:
        d = load(f"w2/{tag}/tasks.json")
        if d is None:
            continue
        admitted = sum(v["admitted"] for k, v in d["per_kind"].items() if k != "denoise")
        rejected = sum(d["rejected"].values())
        m[f"{tag}.tasks"] = d["tasks"]
        m[f"{tag}.proposals"] = admitted + rejected
        m[f"{tag}.admitted_proposals"] = admitted
        m[f"{tag}.elapsed_secs"] = d["elapsed_secs"]
        for reason, n in d["rejected"].items():
            m[f"{tag}.reason.{reason}"] = n
        no_subject = d["rejected"].get("no_subject", 0)
        m[f"{tag}.share.no_subject"] = ci(no_subject, admitted + rejected)
        kinds = [classify(r["detail"]) for r in d["rejections"] if r["reason"] == "no_subject"]
        for c in CLASSES:
            m[f"{tag}.class.{c}"] = kinds.count(c)
    return m


def e1w2_table():
    rows = []
    for tag, corpus, generator in W2_E1:
        d = load(f"w2/{tag}/tasks.json")
        if d is None:
            continue
        m = e1w2_metrics()
        other = sum(d["rejected"].values()) - d["rejected"].get("no_subject", 0)
        rows.append(f"{corpus}, {generator} & {m[f'{tag}.proposals']} & {m[f'{tag}.admitted_proposals']} & "
                    f"{m[f'{tag}.reason.no_subject']} & {other} & "
                    + " / ".join(str(m[f"{tag}.class.{c}"]) for c in CLASSES) + r" \\")
    if not rows:
        return r"\textbf{NOT MEASURED}"
    return "\n".join([r"\begin{tabular}{lrrrrr}", r"\toprule",
                      r"Run & Proposals & Admitted & \code{no\_subject} & Other & "
                      r"absent / apart / unsourced / markup \\", r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"])


W2_E2 = [("e2r", "w2/e2", "registered run"), ("e2b", "w2/e2b", "E2b"), ("e2c", "w2/e2c", "repaired rerun"),
          ("e2d", "w2/e2d", "E2d, 1.7B policy")]
CHECKS = ["improvement", "retention", "anchor", "serve"]


def gate_of(report):
    release = report.get("release") or {}
    return release.get("gate")


def e2w2_metrics():
    """Per run: facts answered at each step, McNemar of each candidate against the base, the
    release gate's four checks with their numbers, the stage counts and the exits."""
    m = {}
    for tag, rel, _ in W2_E2:
        rows = load(f"{rel}/summary.json")
        if not rows:
            continue
        steps_ = {r["step"]: r for r in rows}
        for name, r in steps_.items():
            key = name.replace("-", "_")
            m[f"{tag}.{key}.anchor"] = r["anchor_correct"]
            for g in range(3):
                m[f"{tag}.{key}.g{g}"] = r[f"g{g}_answered"]
        for cand, g in (("candidate-g0", 0), ("candidate-g1", 1)):
            for h in range(3):
                if "base" in steps_ and cand in steps_:
                    gained, lost = discordant(steps_["base"], steps_[cand], h)
                    pre = f"{tag}.base__{cand.replace('-', '_')}.g{h}"
                    m[pre + ".gained"], m[pre + ".lost"] = gained, lost
                    m[pre + ".p"] = f"{mcnemar_exact(gained, lost):.3g}"
        regrade = load(f"{rel}/regrade/summary.json")
        if regrade:
            for r in regrade:
                key = r["step"].replace("-", "_")
                m[f"{tag}.regrade.{key}.anchor"] = r["anchor_correct"]
                for g in range(3):
                    m[f"{tag}.regrade.{key}.g{g}"] = r[f"g{g}_answered"]
            rs = {r["step"]: r for r in regrade}
            for cand in ("candidate-g0", "candidate-g1"):
                for h in range(3):
                    if "base" in rs and cand in rs:
                        gained, lost = discordant(rs["base"], rs[cand], h)
                        pre = f"{tag}.regrade.base__{cand.replace('-', '_')}.g{h}"
                        m[pre + ".gained"], m[pre + ".lost"] = gained, lost
                        m[pre + ".p"] = f"{mcnemar_exact(gained, lost):.3g}"
        for g in (0, 1):
            learn = load(f"{rel}/raw/learn-g{g}.json")
            if not learn:
                continue
            r = learn["report"]
            m[f"{tag}.g{g}.exit"] = learn["exit"]
            m[f"{tag}.g{g}.tasks"] = r["tasks"]["tasks"]
            dist = r["frontier"]["distribution"]
            m[f"{tag}.g{g}.taught"], m[f"{tag}.g{g}.never"] = dist["taught"], dist["never"]
            m[f"{tag}.g{g}.variants"] = r["variants"]["variants"]
            m[f"{tag}.g{g}.records"] = r["dataset"]["records"]
            tr = r["candidate"]["training_record"]
            m[f"{tag}.g{g}.train_records"] = tr["train_records"]
            m[f"{tag}.g{g}.steps"], m[f"{tag}.g{g}.rank"] = tr["steps"], tr["rank"]
            gate = gate_of(r)
            if gate:
                m[f"{tag}.g{g}.released"] = "yes" if r["release"].get("release") else "no"
                for c in CHECKS:
                    m[f"{tag}.g{g}.gate.{c}"] = ("passed" if gate[c]["passed"] else
                                             "refused" if gate[c]["measured"] is not None else "not measured")
                imp = gate["improvement"]["measured"]
                rec = imp["recall"]
                m[f"{tag}.g{g}.imp.variants_n"] = rec["comparison"]["items"]
                m[f"{tag}.g{g}.imp.variants_right"] = round(rec["comparison"]["candidate_accuracy"] * rec["comparison"]["items"])
                m[f"{tag}.g{g}.imp.sign_wins"] = rec["sign_test"]["candidate_wins"]
                m[f"{tag}.g{g}.imp.sign_discordant"] = rec["sign_test"]["discordant"]
                m[f"{tag}.g{g}.imp.sign_p"] = f"{rec['sign_test']['p_value']:.2g}"
                an = gate["anchor"]["measured"]
                if an:
                    m[f"{tag}.g{g}.anchor.drop"] = f"{an['drop']:.2f}"
                    m[f"{tag}.g{g}.anchor.bound"] = f"{an['bound']:.2f}"
                ret = gate["retention"]["measured"]
                if ret and ret["suites"]:
                    m[f"{tag}.g{g}.retention.tasks"] = ret["suites"][0]["suite"]["tasks"]
                    m[f"{tag}.g{g}.retention.right"] = round(ret["suites"][0]["comparison"]["candidate_accuracy"]
                                                            * ret["suites"][0]["comparison"]["items"])
                sv = gate["serve"]["measured"]
                if sv:
                    m[f"{tag}.g{g}.serve.agreed"], m[f"{tag}.g{g}.serve.sampled"] = sv["agreed"], sv["sampled"]
        rollback = load(f"{rel}/raw/rollback.json")
        if rollback:
            m[f"{tag}.rollback.exit"] = rollback["exit"]
            m[f"{tag}.rollback.message"] = rollback["report"].get("error", "")
        releases = load(f"{rel}/raw/releases-final.json")
        if releases is not None:
            m[f"{tag}.releases_final"] = len(releases["report"]["releases"])
    return m


def e2w2_table():
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Run and state & & Gen.~0 facts & Gen.~1 facts & Gen.~2 facts & Anchor \\", r"\midrule"]
    any_row = False
    for tag, rel, label in W2_E2:
        rows = load(f"{rel}/summary.json")
        if not rows:
            continue
        regrade = {r["step"]: r for r in (load(f"{rel}/regrade/summary.json") or [])}
        for r in rows:
            if r["step"] not in ("base", "candidate-g0", "candidate-g1"):
                continue
            use = regrade.get(r["step"], r)
            cells = [ci(use[f"g{g}_answered"], use[f"g{g}_n"]) for g in range(3)]
            lines.append(f"{esc(label)} & {esc(r['step'])} & {cells[0]} & {cells[1]} & {cells[2]} & "
                         f"{use['anchor_correct']}/{use['anchor_n']} \\\\")
            any_row = True
    if not any_row:
        return r"\textbf{NOT MEASURED}"
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def gate_table():
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Run & Learn & Improvement & Retention & Anchor & Serve \\", r"\midrule"]
    m = e2w2_metrics()
    for tag, _, label in W2_E2:
        for g in (0, 1):
            if f"{tag}.g{g}.gate.improvement" in m:
                cells = [m[f"{tag}.g{g}.gate.{c}"] for c in CHECKS]
                lines.append(f"{esc(label)} & generation {g} & " + " & ".join(cells) + r" \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def steps():
    rows = load("e2/summary.json")
    return {r["step"]: r for r in rows} if rows else {}


def e2_metrics():
    s = steps()
    m = {}
    for name, r in s.items():
        key = name.replace("-", "_")
        m[f"e2.{key}.anchor"] = r["anchor_correct"]
        for g in range(3):
            m[f"e2.{key}.g{g}"] = r[f"g{g}_answered"]
    for a, b, g in [("base", "after-g0", 0), ("after-g0", "after-g1", 0),
                    ("base", "after-g1", 1), ("base", "after-g1", 2), ("base", "after-g0", 2),
                    ("after-g1", "after-rollback", 0), ("after-g1", "after-rollback", 1)]:
        if a in s and b in s:
            gained, lost = discordant(s[a], s[b], g)
            tag = f"e2.{a.replace('-', '_')}__{b.replace('-', '_')}.g{g}"
            m[tag + ".gained"] = gained
            m[tag + ".lost"] = lost
            m[tag + ".p"] = f"{mcnemar_exact(gained, lost):.3g}"
    exits = load("e2/exits.json")
    if exits:
        for k, v in exits.items():
            m[f"e2.exit.{k}"] = v
    rollback = load("e2/raw/rollback.json")
    if rollback:
        m["e2.rollback.message"] = rollback["report"]["error"]
    releases = load("e2/raw/releases-final.json")
    if releases is not None:
        m["e2.releases_final"] = len(releases["report"]["releases"])
    return m


def discordant(before, after, g):
    b, a = before[f"g{g}_facts"], after[f"g{g}_facts"]
    gained = sum(1 for k in b if not b[k] and a[k])
    lost = sum(1 for k in b if b[k] and not a[k])
    return gained, lost


LEARN_STAGES = [
    ("tasks", r"\[tasks\] (\d+) task"),
    ("attempts", r"\[solve\] (\d+) solved"),
    ("attempts_passed", r"\[verify\] (\d+) passed"),
    ("attempts_failed", r"\[verify\] \d+ passed, (\d+) failed"),
    ("teacher_verified", r"the teacher's answer verified for (\d+)"),
    ("variants", r"\[variants\] (\d+) variant"),
    ("critiqued", r"\[critique\] (\d+) critiqued"),
    ("dataset_records", r"\[dataset\] (\d+) record"),
]


def learn_metrics():
    """Stage counts of each `learn` command, read from the stderr the run kept."""
    path = RUNS / "e2" / "raw" / "commands.log"
    if not path.exists():
        return {}
    blocks = re.split(r"^\$ ", path.read_text(), flags=re.M)
    learns = [b for b in blocks if " learn " in b.split("\n", 1)[0]]
    m = {}
    for g, block in enumerate(learns):
        for name, pattern in LEARN_STAGES:
            found = re.search(pattern, block)
            if found:
                m[f"e2.g{g}.{name}"] = int(found.group(1))
        refused = re.search(r"^splinter: (.*)$", block, flags=re.M)
        if refused:
            m[f"e2.g{g}.refusal"] = refused.group(1)
    return m


QUAL_PROBES = {"nf4_8192": "r1-nf4-64L-8192", "bf16_8192": "r2-bf16-64L-8192",
               "bf16_16384": "r2-bf16-64L-16384"}
QUAL_CLI = {"nf4_1step": "r3-real-nf4-1step", "nf4_40step": "r3-real-nf4-40step",
            "bf16_1step": "r4-real-bf16-1step", "nf4_batch32": "r3a-real-nf4-default-batch32-OOM"}


def qual_metrics():
    """Numbers the qualification logs print: probe lines and the wrapper's MEASURE line."""
    m = {}
    qdir = RUNS / "qual"
    if not (qdir / "r1-nf4-64L-8192.log").exists():
        return m
    commit = re.search(r"# commit: (\w+)", (qdir / "r1-nf4-64L-8192.log").read_text())
    m["e4.commit"] = commit.group(1)[:8]
    for tag, name in QUAL_PROBES.items():
        text = (qdir / f"{name}.log").read_text()
        for field in ("built_mib", "peak_mib"):
            m[f"e4.probe_{tag}.{field}"] = int(re.search(rf"PROBE {field}=(\d+)", text).group(1))
        for field in ("step_s", "tokens_per_s"):
            m[f"e4.probe_{tag}.{field}"] = re.search(rf"{field}=([\d.]+)", text).group(1)
        m[f"e4.probe_{tag}.exit"] = int(re.search(r"# exit=(\d+)", text).group(1))
    for tag, name in QUAL_CLI.items():
        text = (qdir / f"{name}.log").read_text()
        meas = re.search(r"MEASURE exit=(\d+) wall_s=([\d.]+) host_peak_rss_MiB=(\d+) device_peak_MiB=(\d+)", text)
        m[f"e4.cli_{tag}.exit"] = int(meas.group(1))
        m[f"e4.cli_{tag}.wall_s"] = meas.group(2)
        m[f"e4.cli_{tag}.host_rss_mib"] = int(meas.group(3))
        m[f"e4.cli_{tag}.device_mib"] = int(meas.group(4))
        loss = re.search(r"loss ([\d.]+) -> ([\d.]+)", text)
        if loss:
            m[f"e4.cli_{tag}.loss_first"] = loss.group(1)
            m[f"e4.cli_{tag}.loss_last"] = loss.group(2)
    return m


ALPHA = 0.05


def upper_tail(wins, n):
    """P(X >= wins) for X ~ Binomial(n, 1/2): the exact one-sided sign test."""
    return sum(comb(n, i) for i in range(wins, n + 1)) / 2**n if n else 1.0


def critical_wins(n):
    """Fewest wins out of n non-tied families with one-sided p <= ALPHA, or None."""
    return next((w for w in range(n + 1) if upper_tail(w, n) <= ALPHA), None)


def power(n, q):
    """P(reject) when each non-tied family is a win with probability q."""
    w = critical_wins(n)
    return 0.0 if w is None else sum(comb(n, i) * q**i * (1 - q) ** (n - i) for i in range(w, n + 1))


def design_metrics():
    m = {}
    for n in (15, 20, 25, 30):
        m[f"design.n{n}.wins"] = critical_wins(n)
        for q in (70, 80, 90):
            m[f"design.n{n}.power{q}"] = f"{power(n, q / 100):.2f}"
    return m


def throughput_metrics():
    """The probe's warm rate (steps two and three), step time and peak, and the card hours one
    million tokens take at that rate."""
    path = RUNS / "throughput" / "probe-nf4-8192.log"
    if not path.exists():
        return {}
    text = path.read_text()
    rate = int(re.search(r"PROBE warm_tokens_per_s=(\d+)", text).group(1))
    return {"thr.nf4_8192.tokens_per_s": rate,
            "thr.nf4_8192.step_s": re.findall(r"step=3 .*?step_s=([\d.]+)", text)[0],
            "thr.nf4_8192.peak_mib": int(re.search(r"PROBE peak_mib=(\d+)", text).group(1)),
            "thr.hours_per_million_tokens": f"{1e6 / (rate * 3600):.2f}"}


def ledger_metrics():
    with open(ROOT / "data" / "claims.csv", newline="", encoding="utf-8") as f:
        claims = list(csv.DictReader(f))
    checks = [line for line in (ROOT / "data" / "structure_checks.jsonl").read_text().splitlines() if line.strip()]
    return {"ledger.claims": len(claims), "ledger.structure": sum(c["kind"] == "STRUCTURE" for c in claims),
            "ledger.checks": len(checks)}


def metrics():
    m = {}
    m.update(design_metrics())
    m.update(throughput_metrics())
    m.update(ledger_metrics())
    m.update(qual_metrics())
    m.update(e1_metrics())
    m.update(e1w2_metrics())
    m.update(e2w2_metrics())
    m.update(e2_metrics())
    m.update(learn_metrics())
    return m


# ---------------------------------------------------------------- outputs
def esc(text):
    return text.replace("_", r"\_")


def e1_table():
    d = load("e1/tasks.json")
    if d is None:
        return r"\textbf{NOT MEASURED}"
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Task kind & Admitted & Rejected \\", r"\midrule"]
    for kind, v in d["per_kind"].items():
        lines.append(f"{esc(kind)} & {v['admitted']} & {v['rejected']} \\\\")
    lines += [r"\midrule", r"Rejection reason & \multicolumn{2}{r}{Count} \\", r"\midrule"]
    for reason, n in sorted(d["rejected"].items(), key=lambda kv: -kv[1]):
        lines.append(f"{esc(reason)} & \\multicolumn{{2}}{{r}}{{{n}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def e2_table():
    s = steps()
    if not s:
        return r"\textbf{NOT MEASURED}"
    names = [n for n in ("base", "after-g0", "after-g1", "after-rollback") if n in s]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"State of \code{policy:default} & Facts of gen.~0 & Facts of gen.~1 & "
             r"Facts of gen.~2 (never taught) & Anchor (of 40) \\", r"\midrule"]
    for n in names:
        r = s[n]
        cells = [ci(r[f"g{g}_answered"], r[f"g{g}_n"]) for g in range(3)]
        lines.append(f"{esc(n)} & {cells[0]} & {cells[1]} & {cells[2]} & {r['anchor_correct']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def learn_table():
    m = learn_metrics()
    if not m:
        return r"\textbf{NOT MEASURED}"
    labels = [("tasks", "tasks admitted"), ("attempts", "closed-book attempts (k=4)"),
              ("attempts_passed", "attempts verified correct"), ("teacher_verified", "teacher answers verified"),
              ("variants", "variants written"), ("critiqued", "attempts critiqued"),
              ("dataset_records", "records in the SFT dataset")]
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Stage & Gen.~0 & Gen.~1 \\", r"\midrule"]
    for key, label in labels:
        lines.append(f"{label} & {m.get(f'e2.g0.{key}', '--')} & {m.get(f'e2.g1.{key}', '--')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def e4_table():
    m = qual_metrics()
    if not m:
        return r"\textbf{NOT MEASURED}"
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Run & Base & Tokens & Peak MiB & Seconds & Outcome \\", r"\midrule"]
    for tag, base, tokens in [("nf4_8192", "NF4", 8192), ("bf16_8192", "bf16", 8192), ("bf16_16384", "bf16", 16384)]:
        lines.append(f"probe, 64 layers, random init & {base} & {tokens} & {m[f'e4.probe_{tag}.peak_mib']} "
                     f"& {m[f'e4.probe_{tag}.step_s']} & step done \\\\")
    for tag, base, tokens, what in [("nf4_1step", "NF4", 1024, "27B, 1 step"), ("bf16_1step", "bf16", 1024, "27B, 1 step"),
                                    ("nf4_40step", "NF4", 256, "27B, 40 steps")]:
        lines.append(f"{what} & {base} & {tokens} & {m[f'e4.cli_{tag}.device_mib']} "
                     f"& {m[f'e4.cli_{tag}.wall_s']} & finished \\\\")
    lines.append(f"27B, default batch & NF4 & 32768 & {m['e4.cli_nf4_batch32.device_mib']} "
                 f"& {m['e4.cli_nf4_batch32.wall_s']} & out of memory \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- coding-task tables
# The registered coding-task experiments (prereg/coding-tasks.md). A status is derived, never typed:
# an experiment is PENDING until a row of data/results.csv or a directory data/runs/<id> exists.
REGISTERED = [
    ("CE1", "Untrained agent: resolve rate, failure profile, cost, discordance", "descriptive, no threshold"),
    ("CE4", "SFT on verified trajectories against the base (headline)", "sign test, qualification, alpha 0.05"),
    ("CE6", "27B training and deployment, ten stages", "gates G7 and G8"),
    ("CE7", "Replication on a second, independent repository", "same test, independently"),
]
COMPARISONS = ("CE4", "CE7")


def read_results():
    path = ROOT / "data" / "results.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def experiment_status(key):
    rows = [r for r in read_results() if r["experiment"] == key]
    if rows or (RUNS / key.lower()).exists():
        return f"{len(rows)} result rows" if rows else "recorded"
    return r"\textbf{PENDING}"


def registered_table():
    lines = [r"\begin{tabular}{lp{6.2cm}p{3.6cm}l}", r"\toprule", r"Id & Question & Primary measure & Status \\", r"\midrule"]
    for key, question, measure in REGISTERED:
        lines.append(f"{key} & {question} & {measure} & {experiment_status(key)} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def results_table():
    rows = read_results()
    lines = [r"\begin{tabular}{lllrrrrl}", r"\toprule",
             r"Exp. & Repository & Arm & Families & Wins & Losses & $p$ & Status \\", r"\midrule"]
    for key in COMPARISONS:
        mine = [r for r in rows if r["experiment"] == key]
        if not mine:
            lines.append(rf"{key} & \multicolumn{{7}}{{l}}{{\textbf{{PENDING}}: no run exists}} \\")
        for r in mine:
            wins, losses = int(r["wins"]), int(r["losses"])
            lines.append(f"{key} & {r['repository']} & {r['arm']} & {r['exam_families']} & {wins} & {losses} "
                         f"& {upper_tail(wins, wins + losses):.3f} & {r['status']} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def design_table():
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"Non-tied & Wins needed & Smallest $p$ & \multicolumn{3}{c}{Power if $P(\text{win})=$} \\",
             r"families $n$ & for $p\le 0.05$ & (all wins) & 0.7 & 0.8 & 0.9 \\", r"\midrule"]
    for n in (4, 5, 6, 8, 10, 15, 20, 25, 30):
        w = critical_wins(n)
        lines.append(f"{n} & {w if w is not None else 'none'} & {upper_tail(n, n):.2g} & "
                     f"{power(n, 0.7):.2f} & {power(n, 0.8):.2f} & {power(n, 0.9):.2f} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def power_curve():
    return "n,q70,q80,q90\n" + "".join(
        f"{n},{power(n, 0.7):.4f},{power(n, 0.8):.4f},{power(n, 0.9):.4f}\n" for n in range(2, 41))


def catalog():
    """The authored repository families, from the catalogue beside this paper in the repository."""
    path = ROOT.parent.parent / "tasks" / "catalog.toml"
    return [f for f in tomllib.loads(path.read_text())["family"] if f.get("kind") == "repository"]


def families_table():
    lines = [r"\begin{tabular}{llrr}", r"\toprule", r"Family & Language & Tests & Admitted tasks \\", r"\midrule"]
    for f in catalog():
        lines.append(f"\\code{{{f['id']}}} & {f['language']} & {f['tests']} & {f['admitted']} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def macro_name(key):
    digits = "Zero One Two Three Four Five Six Seven Eight Nine".split()
    words = re.split(r"[._\-]+", key)
    spelled = [re.sub(r"\d", lambda d: digits[int(d.group())], w.capitalize()) for w in words]
    return "m" + "".join(spelled)


def write_outputs(out):
    out.mkdir(parents=True, exist_ok=True)
    m = metrics()
    macros = "".join(f"\\newcommand{{\\{macro_name(k)}}}{{{v}}}\n" for k, v in sorted(m.items()))
    macros += f"\\newcommand{{\\mExperimentsMeasured}}{{{len({k.split('.')[0] for k in m})}}}\n"
    (out / "macros.tex").write_text(macros)
    (out / "e1_table.tex").write_text(e1_table() + "\n")
    (out / "e1w2_table.tex").write_text(e1w2_table() + "\n")
    (out / "e2w2_table.tex").write_text(e2w2_table() + "\n")
    (out / "gate_table.tex").write_text(gate_table() + "\n")
    (out / "e2_table.tex").write_text(e2_table() + "\n")
    (out / "e4_table.tex").write_text(e4_table() + "\n")
    (out / "learn_table.tex").write_text(learn_table() + "\n")
    (out / "registered_table.tex").write_text(registered_table() + "\n")
    (out / "results_table.tex").write_text(results_table() + "\n")
    (out / "design_table.tex").write_text(design_table() + "\n")
    (out / "power_curve.csv").write_text(power_curve())
    (out / "families_table.tex").write_text(families_table() + "\n")
    macros_extra = f"\\newcommand{{\\alphalevel}}{{{ALPHA}}}\n"
    with open(out / "macros.tex", "a") as handle:
        handle.write(macros_extra)
    (out / "limitations.tex").write_text(limitations())


def limitations():
    """One sentence per experiment whose record is absent: not measured, never zero."""
    absent = {
        "e1/tasks.json": "E1 (corpus-expansion funnel) was not measured.",
        "e2/summary.json": "E2 and E3 (the two-generation turn) were not measured.",
        "qual/r1-nf4-64L-8192.log": "E4: the 27B training qualification was not measured; the substrate "
                             "section states only what the code contains.",
    }
    lines = [text for rel, text in absent.items() if not (RUNS / rel).exists()]
    return "\n\n".join(lines) + "\n" if lines else "\n"


# ---------------------------------------------------------------- check
def cited_ids():
    used = set()
    for path in SECTIONS.glob("*.tex"):
        for group in re.findall(r"\\cl\{([^}]*)\}", path.read_text()):
            used |= {i.strip() for i in group.split(",")}
    return used


def normalise(text):
    return " ".join(text.split())


def check_evidence(c, m, problems):
    kind = c["kind"]
    entries = [e for e in c["evidence"].split(";") if e]
    if kind in ("MEASURED", "COMPUTED"):
        for ev in entries:
            key = ev.removeprefix("runs/")
            if key not in m:
                problems.append(f"{c['id']}: metric {key} has no record")
            elif str(m[key]) not in c["claim"]:
                problems.append(f"{c['id']}: claim text lacks {m[key]} for {key}")
    elif kind == "REGISTERED":
        for ev in entries:
            rel, _, phrase = ev.partition("#")
            path = ROOT / rel
            if not path.exists() or normalise(phrase) not in normalise(path.read_text()):
                problems.append(f"{c['id']}: {rel} does not contain {phrase!r}")
    elif kind == "PLANNED":
        for ev in entries:
            key = ev.removeprefix("exp:")
            if experiment_status(key) != r"\textbf{PENDING}":
                problems.append(f"{c['id']}: {key} has results; the planned claim is false")
    elif kind == "CITED":
        bib = (ROOT / "references.bib").read_text()
        for key in entries:
            if "{" + key + "," not in bib:
                problems.append(f"{c['id']}: {key} is not in references.bib")


def check():
    with open(ROOT / "data" / "claims.csv", newline="", encoding="utf-8") as f:
        claims = list(csv.DictReader(f))
    ids = [c["id"] for c in claims]
    problems = []
    if len(set(ids)) != len(ids):
        problems.append("duplicate claim ids")
    used = cited_ids()
    for missing in sorted(used - set(ids)):
        problems.append(f"text cites {missing}, not in the ledger")
    for unused in sorted(set(ids) - used):
        problems.append(f"ledger row {unused} is cited nowhere in the text")
    m = metrics()
    counts = {}
    for c in claims:
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1
        allowed = "planned" if c["kind"] == "PLANNED" else "verified"
        if c["status"] != allowed:
            problems.append(f"{c['id']}: status {c['status']!r}, expected {allowed!r}")
        check_evidence(c, m, problems)
    print(f"{len(claims)} claims ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}), {len(problems)} problems")
    for p in problems:
        print("FAIL", p)
    sys.exit(1 if problems else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true")
    p.add_argument("--list-metrics", action="store_true")
    args = p.parse_args()
    if args.list_metrics:
        for k, v in sorted(metrics().items()):
            print(k, v)
    elif args.check:
        check()
    elif args.out:
        write_outputs(args.out)
    else:
        p.error("one of --out, --check, --list-metrics")


if __name__ == "__main__":
    main()
