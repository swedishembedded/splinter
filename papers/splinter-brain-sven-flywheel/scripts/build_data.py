#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of self-improving model
# pipelines for its clients. If your team needs expertise in turning raw run
# records into tables whose every number is checked against the claims they
# support, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Builds every number and table of the paper from data/measurements.json and the other files of data/.

    build_data.py --out DIR     write macros.tex and the tables into DIR
    build_data.py --check       verify the ledger against the measurements and the text
    build_data.py --check --evidence DIR   also compare data/measurements.json with the raw logs in DIR
                                (default ~/project/shared/containers/runs/paper-evidence/flywheel; absent = skipped)
    build_data.py --check --final   also fail while any REGISTERED or PLANNED claim remains
                                or the paper is still marked draft (status.txt)

A draft may contain registered but not yet validated results; a final paper contains
validated results only. status.txt holds `draft` or `final`; a `final` paper is always
checked with --final.

Nothing measured is typed by hand into the paper. A MEASURED claim names its
evidence as a dotted metric of data/measurements.json; `--check`
resolves each against that file, requires the claim's text to contain the
value, and requires that the paper's text cites the claim. An experiment whose
record is absent is printed as NOT MEASURED, never as zero.

Raw logs (run on a GPU, generated output, kept outside the repository; see DEFAULT_EVIDENCE):
  qual/*                 training-substrate qualification logs
  throughput/*.log       a later probe of the 27B architecture's training step rate
data/measurements.json records, per log, the values the claims cite, the brain commit, the exact
command and the log's sha256. ../results.csv holds paired outcomes of the registered
coding-task experiments (none yet).

Claim kinds (data/claims.csv): STRUCTURE (verify_structure.py), MEASURED and COMPUTED (a
metric of this script, whose value the claim text must contain), REGISTERED (a phrase the
dated registration files contain), PLANNED (an experiment with no result row yet; false as
soon as one exists) and CITED (a key of references.bib).
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import tomllib
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"


# ---------------------------------------------------------------- metrics
# The raw logs behind the MEASURED claims are generated output of GPU runs and are not kept in the
# repository. data/measurements.json holds the values the claims cite, with each log's sha256, the
# brain commit and the exact command; `--evidence DIR` re-extracts every value from the logs.
MEASUREMENTS = ROOT / "data" / "measurements.json"
DEFAULT_EVIDENCE = "~/project/shared/containers/runs/paper-evidence/flywheel"
E4_COMMIT_RUN = "r1-nf4-64L-8192"


def load_measurements():
    return json.loads(MEASUREMENTS.read_text(encoding="utf-8"))["runs"]


def qual_metrics():
    """E4: the values the qualification runs' logs printed, read from data/measurements.json."""
    runs = load_measurements()
    m = {"e4.commit": runs[E4_COMMIT_RUN]["brain_commit"][:8]}
    for run in runs.values():
        if run["metric_prefix"].startswith("e4."):
            m.update({f"{run['metric_prefix']}.{k}": v for k, v in run["metrics"].items()})
    return m


def prefixed_metrics(prefix):
    """The values of every run in data/measurements.json whose metric prefix starts with `prefix`,
    keyed `<metric prefix>.<name>`."""
    m = {}
    for run in load_measurements().values():
        if run["metric_prefix"].startswith(prefix):
            m.update({f"{run['metric_prefix']}.{k}": v for k, v in run["metrics"].items()})
    return m


def traj_metrics():
    """The trajectory-capture findings: values a log printed on a `TJ key=value ...` line."""
    return prefixed_metrics("tj.")


def ce4_metrics():
    return prefixed_metrics("ce4.")


def ce1_metrics():
    """CE1 pass 1: the counts of every finished development set."""
    return prefixed_metrics("ce1.")


def ce8_metrics():
    """CE8 training runs: the loss curve's ends, the monitoring evaluations and the guard's verdict.
    A `ce8.eval-` run is an evaluation of a candidate against the base under the coding-task
    protocol: its paired families and the exact sign test, from an `experiences compare` report."""
    return prefixed_metrics("ce8.")


def g8_metrics():
    """G8: the served-versus-in-process agreement report of an adapter arm. The values are
    formatted where the report is read (parse_log) so a claim quoting them quotes the report."""
    return prefixed_metrics("g8.")


def g8_report_metrics(report):
    """The agreement report of one G8 arm (`g8-compare.py`): the two sides' mean |served -
    in-process| log-probability difference, their top-1 agreement, the served mean
    log-probability, and the base-normalised ratio and point difference of the registration."""
    base, adapter = report["sides"]["base"], report["sides"]["adapter"]
    ratio = report["derived"]["diff_ratio_adapter_over_base"]
    points = report["derived"]["top1_points_adapter_minus_base"]
    return {
        "records": base["records"], "positions": base["positions"],
        "base_diff": f"{base['mean_abs_logprob_diff']:.3e}",
        "adapter_diff": f"{adapter['mean_abs_logprob_diff']:.3e}",
        "diff_ratio": f"{ratio:.3f}",
        "base_top1": f"{100 * base['top1_agreement']:.2f}",
        "adapter_top1": f"{100 * adapter['top1_agreement']:.2f}",
        "top1_drop": f"{-100 * points:.2f}",
        "served_base_logprob": f"{base['served_mean_logprob']:.6f}",
        "served_adapter_logprob": f"{adapter['served_mean_logprob']:.6f}",
        "verdict": report["verdict"],
        # the report carries the emitter's manifest, so the split the scores were taken on is
        # part of the record rather than a number retyped from it
        **({"held_out_records": report["provenance"]["held_out"]["records"],
            "held_out_groups": report["provenance"]["held_out"]["groups"]}
           if "provenance" in report else {}),
    }


def ce8_train_metrics(text):
    """The CE8 trainer's output: `step N loss L ...` lines, `step N lr R train T eval E` evaluation
    lines and the run record's base score. The memorisation guard of the registration: memorised when
    the final training loss is below half the first step's while the final monitoring loss is above
    its minimum."""
    steps = [(int(a), b) for a, b in re.findall(r"^step\s+(\d+)\s+loss ([\d.]+)", text, re.M)]
    evals = [(int(a), b) for a, b in re.findall(r"^step\s+(\d+)\s+lr \S+\s+train [\d.]+\s+eval ([\d.]+)", text, re.M)]
    base = re.search(r'"base_score": \{\s*"loss": ([\d.]+)', text).group(1)
    record = json.loads(text[text.index("{"):])
    if "memorisation" in record:
        # a run judged by the training record's own verdict (the guard on the kept evaluation)
        m = record["memorisation"]
        return {"records": record["records"], "steps_run": steps[-1][0],
                "kept_step": m["kept_step"], "kept_monitor": f"{m['monitor_loss']:.4f}",
                "kept_train": f"{m['train_loss']:.4f}", "base_monitor": f"{m['base_loss']:.4f}",
                "verdict": m["verdict"].replace("_", " ")}
    best = min(evals, key=lambda e: float(e[1]))
    memorised = float(steps[-1][1]) < float(steps[0][1]) / 2 and float(evals[-1][1]) > float(best[1])
    return {"steps_run": steps[-1][0], "loss_first": steps[0][1], "loss_last": steps[-1][1],
            "eval_best": best[1], "eval_best_step": best[0], "eval_last": evals[-1][1],
            "base_monitor": f"{float(base):.4f}", "memorised": "yes" if memorised else "no"}


def ce4_compare_metrics(compare):
    """The values a claim may cite from an `experiences compare` report (logs/pq-S1-compare.json of lane ce)."""
    families = compare["families"]
    return {
        "tasks": compare["tasks"], "paired": compare["paired"], "families": len(families),
        "wins": compare["clusters"]["wins"], "losses": compare["clusters"]["losses"], "ties": compare["clusters"]["ties"],
        "discordant": compare["sign_test"]["discordant"], "p_value": compare["sign_test"]["p_value"],
        "adapter_verified": compare["candidate"]["competence"]["counts"]["verified"],
        "adapter_exposed": compare["candidate"]["exposed"],
        "base_verified": compare["baseline"]["competence"]["counts"]["verified"],
        "base_exposed": compare["baseline"]["exposed"],
    }


def ce1_summary_metrics(summary):
    """The values a claim may cite from one CE1 summary (runs/ce/ce1-evidence.py)."""
    return {
        "tasks": summary["tasks"], "solved": summary["solved"],
        "success": summary["conclusions"].get("success", 0),
        "cancelled": summary["conclusions"].get("cancelled", 0),
        "passed_clean": summary["passed_clean"], "failed": summary["failed"],
        "undecided": summary["undecided"], "exposed": summary["exposed"],
        "hidden_path": summary["exposed_by"].get("hidden_path", 0),
        "outside_workspace": sum(n for k, n in summary["exposed_by"].items() if k.startswith("outside_workspace")),
        **({"raw_pass": summary["yield"]["raw_pass"], "audit_removed": summary["yield"]["removed"]} if summary.get("yield") else {}),
    }


def parse_log(prefix, text):
    """The metrics of one raw log, by the shape its prefix names: a probe, a command-line run or
    the throughput probe. Used only to check measurements.json against the logs."""
    m = {}
    if prefix.startswith("ce1."):
        return ce1_summary_metrics(json.loads(text))
    if prefix.startswith("ce4."):
        return ce4_compare_metrics(json.loads(text))
    if prefix.startswith("ce8.eval"):
        return ce4_compare_metrics(json.loads(text))
    if prefix.startswith("ce8.build"):
        # a dataset build report: what it kept and what it excluded, by reason
        d = json.loads(text)
        ex = d.get("excluded") or {}
        return {"records": d["records"], **{k: ex.get(k, 0) for k in ("failed", "repeated_action", "compacted", "over_block")}}
    if prefix.startswith("ce8.shape"):
        # the arm-shape summary is written flat, so the file IS the metrics dict
        return json.loads(text)
    if prefix.startswith("ce8."):
        return ce8_train_metrics(text)
    if prefix.startswith("g8."):
        return g8_report_metrics(json.loads(text))
    if prefix.startswith("tj."):
        line = next(l for l in text.splitlines() if l.startswith("TJ "))
        return {k: int(v) for k, v in (kv.split("=") for kv in line.split()[1:])}
    if prefix.startswith("e4.probe_"):
        for field in ("built_mib", "peak_mib"):
            m[field] = int(re.search(rf"PROBE {field}=(\d+)", text).group(1))
        for field in ("step_s", "tokens_per_s"):
            m[field] = re.search(rf"{field}=([\d.]+)", text).group(1)
        m["exit"] = int(re.search(r"# exit=(\d+)", text).group(1))
    elif prefix.startswith("e4.cli_"):
        meas = re.search(r"MEASURE exit=(\d+) wall_s=([\d.]+) host_peak_rss_MiB=(\d+) device_peak_MiB=(\d+)", text)
        m.update(exit=int(meas.group(1)), wall_s=meas.group(2), host_rss_mib=int(meas.group(3)),
                 device_mib=int(meas.group(4)))
        loss = re.search(r"loss ([\d.]+) -> ([\d.]+)", text)
        if loss:
            m.update(loss_first=loss.group(1), loss_last=loss.group(2))
    else:
        m["tokens_per_s"] = int(re.search(r"PROBE warm_tokens_per_s=(\d+)", text).group(1))
        m["step_s"] = re.findall(r"step=3 .*?step_s=([\d.]+)", text)[0]
        m["peak_mib"] = int(re.search(r"PROBE peak_mib=(\d+)", text).group(1))
    return m


def check_evidence_dir(evidence, problems):
    """Compare measurements.json with the raw logs in `evidence`: the sha256 of each log, the values
    extracted from it, and (for the qualification logs) the commit and command of its header. An
    absent directory is not checked and not a failure."""
    if not evidence.is_dir():
        print(f"evidence directory {evidence} absent: log digests not checked")
        return
    for name, run in load_measurements().items():
        path = evidence / run["log"]
        if not path.is_file():
            problems.append(f"{name}: {run['log']} is missing from {evidence}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["sha256"]:
            problems.append(f"{name}: sha256 of {run['log']} differs from data/measurements.json")
            continue
        text = path.read_text(encoding="utf-8")
        if parse_log(run["metric_prefix"], text) != run["metrics"]:
            problems.append(f"{name}: values extracted from {run['log']} differ from data/measurements.json")
        if run["metric_prefix"].startswith("e4."):
            if f"# commit: {run['brain_commit']}" not in text or f"# command: {run['command']}" not in text:
                problems.append(f"{name}: commit or command differs from the header of {run['log']}")
    print(f"evidence directory {evidence}: {len(load_measurements())} logs checked")


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
    """The probe's warm rate, step time and peak, and the card hours one million tokens take at that rate."""
    run = load_measurements()["probe-nf4-8192"]
    rate = run["metrics"]["tokens_per_s"]
    return {**{f"{run['metric_prefix']}.{k}": v for k, v in run["metrics"].items()},
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
    m.update(traj_metrics())
    m.update(ce1_metrics())
    m.update(ce4_metrics())
    m.update(ce8_metrics())
    m.update(g8_metrics())
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
# an experiment is PENDING until a row of data/results.csv exists.
REGISTERED = [
    ("CE1", "Untrained agent: resolve rate, failure profile, cost, discordance", "descriptive, no threshold"),
    ("CE4", "8B SFT on verified 27B-teacher trajectories against the untrained 8B (headline)", "sign test, qualification, alpha 0.05"),
    ("CE6", "27B training and deployment, ten stages", "gates G7 and G8"),
    ("CE7", "Replication on a second, independent repository", "same test, independently"),
    ("CE8", "27B SFT on its own verified trajectories against the 27B base", "sign test, qualification, alpha 0.05"),
]
COMPARISONS = ("CE4", "CE7", "CE8")


def ce1_table():
    """One row per finished CE1 set, from the measurements; empty until a set is finished."""
    rows = {}
    for key, value in ce1_metrics().items():
        prefix, field = key.rsplit(".", 1)
        rows.setdefault(prefix, {})[field] = value
    if not rows:
        return r"\textbf{NOT MEASURED}"
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Set & Tasks & Success & Stopped & Hidden pass & Audit removed & Passed clean & Failed \\", r"\midrule"]
    for prefix, r in sorted(rows.items()):
        name = prefix.removeprefix('ce1.').replace('_', ' ')
        lines.append(f"{name} & {r['tasks']} & {r['success']} & {r['solved'] - r['success']} & {r.get('raw_pass', '--')} "
                     f"& {r.get('audit_removed', '--')} & {r['passed_clean']} & {r['failed']} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def read_results():
    path = ROOT / "data" / "results.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def experiment_status(key):
    rows = [r for r in read_results() if r["experiment"] == key]
    if rows:
        return f"{len(rows)} result rows"
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
    macros += f"\\newcommand{{\\mPaperStatus}}{{{paper_status()}}}\n"
    macros += f"\\newcommand{{\\mExperimentsMeasured}}{{{len({k.split('.')[0] for k in m})}}}\n"
    (out / "macros.tex").write_text(macros)
    (out / "e4_table.tex").write_text(e4_table() + "\n")
    (out / "registered_table.tex").write_text(registered_table() + "\n")
    (out / "results_table.tex").write_text(results_table() + "\n")
    (out / "ce1_table.tex").write_text(ce1_table() + "\n")
    (out / "design_table.tex").write_text(design_table() + "\n")
    (out / "power_curve.csv").write_text(power_curve())
    (out / "families_table.tex").write_text(families_table() + "\n")
    macros_extra = f"\\newcommand{{\\alphalevel}}{{{ALPHA}}}\n"
    with open(out / "macros.tex", "a") as handle:
        handle.write(macros_extra)


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


def paper_status():
    status = (ROOT / "status.txt").read_text().strip()
    if status not in ("draft", "final"):
        sys.exit(f"status.txt must say draft or final, not {status!r}")
    return status


def check(final, evidence):
    final = final or paper_status() == "final"
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
    check_evidence_dir(evidence, problems)
    if final:
        if paper_status() == "draft":
            problems.append("status.txt says draft; a final paper is marked final")
        for c in claims:
            if c["kind"] in ("REGISTERED", "PLANNED"):
                problems.append(f"{c['id']}: {c['kind']} claim remains; a final paper holds validated results only")
    print(f"{len(claims)} claims ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}), {len(problems)} problems")
    for p in problems:
        print("FAIL", p)
    sys.exit(1 if problems else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true")
    p.add_argument("--evidence", type=Path, help="with --check: directory of the raw logs "
                   f"(default {DEFAULT_EVIDENCE}); skipped when absent")
    p.add_argument("--final", action="store_true", help="with --check: refuse a draft")
    p.add_argument("--list-metrics", action="store_true")
    args = p.parse_args()
    if args.list_metrics:
        for k, v in sorted(metrics().items()):
            print(k, v)
    elif args.check:
        check(args.final, (args.evidence or Path(DEFAULT_EVIDENCE)).expanduser())
    elif args.out:
        write_outputs(args.out)
    else:
        p.error("one of --out, --check, --list-metrics")


if __name__ == "__main__":
    main()
