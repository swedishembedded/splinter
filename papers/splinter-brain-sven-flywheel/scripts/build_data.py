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
import re
import sys
import tomllib
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
SECTIONS = ROOT / "sections"


# ---------------------------------------------------------------- metrics
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
    return m


# ---------------------------------------------------------------- outputs
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
    (out / "e4_table.tex").write_text(e4_table() + "\n")
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
