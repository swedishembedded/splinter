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
  rsi{2,3,4}/verdicts/   the independent examiner's signed step and plan verdicts of the RSI series
  rsi{3,4}/lineages/*/rounds.jsonl   the learner's round records of those series
  rsi1-*/                RSI1's registered analyses and arm R/P/P2 evaluations
  (the RSI records are folders of the evidence dataset splinter-flywheel-evidence)
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


def verify_metrics(report):
    """The values a claim may cite from an `experiences verify` report: how many attempts passed
    and failed, and how many were exposed (a pass on a visible check that fails the hidden one)."""
    return {"experiences": report["experiences"], "passed": report["passed"],
            "failed": report["failed"], "exposed": report["exposed"],
            "undecided": report["undecided"]}


def time_metrics(text):
    """The values a claim may cite from a training time record (`train-*.time`): the wall time,
    the steps and, when the record names it, the training binary's SHA-256."""
    m = {}
    for key in ("wall_s", "steps"):
        found = re.search(rf"\b{key}=(\d+)", text)
        if found:
            m[key] = int(found.group(1))
    binary = re.search(r"binary_sha256=([0-9a-f]{64})", text)
    if binary:
        m["binary_sha256"] = binary.group(1)
    return m


def behaviour_metrics(text):
    """The values a claim may cite from the behaviour table of the 2026-10-25 session record:
    how many of the 46 D46 attempts edited a file and how many ended in the repetition stop,
    per arm. The table is the one under `Behaviour is seed-stable`; the arms are named there."""
    starts = text.find("Behaviour is seed-stable")
    if starts < 0:
        return {}
    section = text[starts:].split("###")[0]
    names = {"base": "base", "decisive seed 1": "decisive_s1", "decisive seed 7": "decisive_s7",
             "decisive seed 13": "decisive_s13", "every action, seed 1": "all_actions_s1"}
    m = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] not in names:
            continue
        edited = re.match(r"(\d+)/46", cells[1])
        aborted = re.match(r"(\d+)", cells[2])
        if edited and aborted:
            m[f"{names[cells[0]]}_edited"] = int(edited.group(1))
            m[f"{names[cells[0]]}_aborted"] = int(aborted.group(1))
    return m


def edit_rate_metrics(text):
    """The values a claim may cite from the edit-rate record over the stored D46 attempts: how
    many attempts edited a file and how many ended in the repetition stop, per arm."""
    names = {"pilot-d46-base.verify.json": "base",
             "pilot-d46-decisive.verify.json": "decisive_s1",
             "pilot-d46-decisive2.verify.json": "decisive_s7",
             "pilot-s13-d46-decisive-s13.verify.json": "decisive_s13",
             "pilot-d46-allnew.verify.json": "all_actions_s1"}
    m = {}
    for line in text.splitlines():
        found = re.match(r"(\S+)\s+attempts\s+(\d+) edited\s+(\d+) \(\s*\d+%\) repetition-aborted\s+(\d+)", line)
        if found and found.group(1) in names:
            arm = names[found.group(1)]
            m[f"{arm}_attempts"] = int(found.group(2))
            m[f"{arm}_edited"] = int(found.group(3))
            m[f"{arm}_aborted"] = int(found.group(4))
    return m


def repeat_metrics(report):
    """The values a claim may cite from the derived repeat record: the three runs' pass counts and
    the number of the 46 task verdicts that differ between each pair."""
    m = {"tasks": report["tasks"]}
    for name, passed in report["passes"].items():
        m[f"{name}_passed"] = passed
    for pair, count in report["verdict_differences"].items():
        m[f"diff_{pair}"] = count
    return m


def det_metrics(prefix, record):
    """The values a claim may cite from the serving-determinism record: how many identical greedy
    requests were sent one at a time and at once, how many distinct completions came back, and the
    position of the first divergent token (`determinism-record.json`, derived from the probe
    outputs by the same scripts that produced them)."""
    if prefix.startswith("g8.det.probe"):
        bm = record["batch_matrix"]
        return {
            "sequential_requests": record["sequential_requests"],
            "sequential_distinct": record["sequential_distinct_token_sequences"],
            "concurrent_requests": record["concurrent_requests_96"],
            "concurrent_distinct": record["concurrent_distinct_96"],
            "first_divergent_token": int(record["concurrent_first_diff_96"]),
            "concurrent2_distinct": bm["2"]["concurrent_distinct"],
            "concurrent8_distinct": bm["8"]["concurrent_distinct"],
            "short_prompt_concurrent_distinct": record["warm_concurrent_distinct_short"],
            "long_prompt_concurrent_distinct": record["warm_concurrent_distinct_long"],
        }
    arms = {
        "subset_tasks": record["arm_subset_tasks"],
        "c1a_passed": record["arm_c1a_passed"],
        "c1b_passed": record["arm_c1b_passed"],
        "c1_verdict_differences": record["arm_c1_verdict_differences"],
    }
    if "arm_c8a_passed" in record:
        arms["c8a_passed"] = record["arm_c8a_passed"]
    if "arm_c8_verdict_differences" in record:
        arms["c8_verdict_differences"] = record["arm_c8_verdict_differences"]
    return arms


def proto6_metrics(record):
    """The values a claim may cite from the six-decoding-seed protocol read: each arm's mean
    pass@1 over the registered decoding seeds and each comparison's per-family sign test."""
    keys = ["seed_count", "base_mean_pass1", "decisive_mean_pass1", "all_mean_pass1"]
    for pair in ("decisive_vs_base", "decisive_vs_all", "base_vs_all"):
        keys += [f"{pair}_wins", f"{pair}_losses", f"{pair}_ties", f"{pair}_p"]
    return {k: record[k] for k in keys if k in record}


# ---------------------------------------------------------------- RSI series
# The recursive-improvement series (prereg/recursive-improvement.md, rsi2.md, rsi3.md, rsi4.md). RSI2-4 are
# decided by an independent examiner's signed aggregate verdicts (one per lineage and step, and one plan
# verdict over the registered rule); the learner's side is each lineage's rounds.jsonl. RSI1's arms were
# measured by their own registered analysis scripts.
RSI_SERIES = (("rsi3", "RSI3"), ("rsi4", "RSI4"))
RSI_LINEAGES = ("r0", "r1", "r2", "r3")
RSI_STEPS = (1, 2, 3)


def fmt_p(p):
    """A p value as the paper prints it: three significant digits from 0.001 up, two below."""
    return f"{p:.3g}" if p >= 1e-3 else f"{p:.2g}"


def rsi_verdict_metrics(v):
    """One step verdict of the examiner: items right for the candidate and its predecessor, the paired
    wins and losses over the compared items, the one-sided exact sign test and whether the step holds,
    the exam's identity, the training-overlap audit and, where the record carries them, the exclusion
    counts and the serving and protocol settings. An absent exclusion record stays absent."""
    m = {"right": v["items_right"], "predecessor_right": v["predecessor_items_right"],
         "wins": v["wins"], "losses": v["losses"], "p": fmt_p(v["p_one_sided"]),
         "holds": "yes" if v["step_holds"] else "no",
         "exam": v["collection"]["exam_sha256"][:12], "items": v["collection"]["items"],
         "exact_matches": v["overlap"]["exact_matches"], "flags": len(v["integrity_flags"]),
         "signature": v["signature"]["alg"]}
    if "exclusion" in v:
        m.update(excluded=v["exclusion"]["excluded"], compared=v["exclusion"]["compared"])
    if "brain_sha256" in v.get("serving_build", {}):
        m["serving_brain"] = v["serving_build"]["brain_sha256"][:9]
    if "kv_cache" in v.get("serving", {}):
        m["kv"] = v["serving"]["kv_cache"]
    for key in ("max_tokens", "truncated"):
        if key in v["protocol"]:
            m[key] = v["protocol"][key]
    return m


def rsi_plan_metrics(plan):
    """The plan verdict: whether the registered rule is met, how many lineages succeeded and failed,
    and the pooled sign test of every step."""
    m = {"plan_met": "true" if plan["plan_met"] else "false",
         "succeeded": plan["series_succeeded"], "failed": plan["series_failed"]}
    for step, pooled in sorted(plan["pooled_steps"].items()):
        m.update({f"g{step}_wins": pooled["wins"], f"g{step}_losses": pooled["losses"],
                  f"g{step}_p": fmt_p(pooled["p_one_sided"]), f"g{step}_series": pooled["series"]})
    return m


def rsi_rounds_metrics(text):
    """A lineage's rounds.jsonl (the learner's side): per generation what the champion proposed and
    mispredicted, what was replayed and trained, the frontier measurement, the dev gate's paired
    counts and rates and its retention losses, and the lineage's learner hours in all."""
    m, seconds = {}, 0.0
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g, d = f"g{r['generation']}", r["decision"]
        m.update({f"{g}_proposed": r["proposed"], f"{g}_mispredicted": r["mispredicted"],
                  f"{g}_replayed": r["replayed"], f"{g}_steps": r["trained"]["steps"],
                  f"{g}_dev_wins": d["wins"], f"{g}_dev_losses": d["losses"],
                  f"{g}_dev_p": fmt_p(d["p_value"]), f"{g}_dev_before": d["champion_rate"],
                  f"{g}_dev_after": d["candidate_rate"], f"{g}_retention_losses": d["retention_losses"],
                  f"{g}_advanced": "yes" if r["advanced"] else "no"})
        if r.get("trained_records") is not None:
            m[f"{g}_trained_records"] = r["trained_records"]
        if r.get("frontier"):
            m.update({f"{g}_{k}": r["frontier"][k] for k in ("measured", "on_frontier", "requests")})
        seconds += r["seconds"]
    m["hours"] = f"{seconds / 3600:.1f}"
    return m


def rsi1_analysis_metrics(a):
    """An RSI1 arm's registered analysis: per step the pooled wins, losses and sign test and whether a
    replicate fell, per replicate its items right at each generation and its step counts (arms H, HP and
    HR); for arm HL's per-lineage analysis, each lineage's steps, whether it succeeds and the pooled steps."""
    m = {}
    if "lineages" in a:
        for lineage in a["lineages"]:
            r = lineage["lineage"]
            for i, step in enumerate(lineage["steps"], 1):
                m.update({f"{r}_g{i}_wins": step["wins"], f"{r}_g{i}_losses": step["losses"],
                          f"{r}_g{i}_p": fmt_p(step["p"]), f"{r}_g{i}_right": step["right"]})
            m[f"{r}_succeeds"] = "yes" if lineage["succeeds"] else "no"
        m.update(succeeding=a["lineages_succeeding"], met="yes" if a["criterion_met"] else "no")
        for step, pooled in sorted(a["pooled"].items()):
            m.update({f"{step}_wins": pooled["wins"], f"{step}_losses": pooled["losses"], f"{step}_p": fmt_p(pooled["p"])})
        return m
    for key, step in sorted(a.items()):
        g = int(key.split("-")[0].removeprefix("gen"))
        m.update({f"g{g}_wins": step["wins"], f"g{g}_losses": step["losses"], f"g{g}_p": fmt_p(step["sign_p"]),
                  f"g{g}_falls": "no" if step["no_replicate_falls"] else "yes"})
        for rep, counts in sorted(step["per_replicate"].items()):
            m.update({f"r{rep}_g{g - 1}_right": counts["from"], f"r{rep}_g{g}_right": counts["to"],
                      f"r{rep}_g{g}_wins": counts["wins"], f"r{rep}_g{g}_losses": counts["losses"]})
    return m


def rsi1_rl_metrics(e):
    """One iteration's evaluation of RSI1 arms R, P and P2: pass@1 and pass@4 on the fresh items of the
    trained families and on the never-trained families, and the items right on arm H's exam."""
    return {"seen_p1": e["test-seen"]["pass@1"], "seen_p4": e["test-seen"]["pass@4"],
            "held_p1": e["test-held"]["pass@1"], "held_p4": e["test-held"]["pass@4"],
            "exam_right": e["exam-h"]["right"], "exam_items": e["exam-h"]["items"]}


def rsi_pooled_metrics(series, m):
    """COMPUTED: each step's wins and losses summed over the lineages' step verdicts, with the exact
    one-sided sign test of the sums (the plan verdict states the same figures; check() compares them)."""
    out = {}
    for step in RSI_STEPS:
        wins = sum(v for k, v in m.items() if re.fullmatch(rf"{series}\.v\.r\d\.g{step}\.wins", k))
        losses = sum(v for k, v in m.items() if re.fullmatch(rf"{series}\.v\.r\d\.g{step}\.losses", k))
        if wins + losses:
            out.update({f"{series}.pooled.g{step}.wins": wins, f"{series}.pooled.g{step}.losses": losses,
                        f"{series}.pooled.g{step}.p": fmt_p(upper_tail(wins, wins + losses))})
    return out


def rsi_gate_metrics(series, m):
    """COMPUTED: how many steps the learner's dev gate accepted and how many of those the examiner's
    step verdict found not to hold, and per generation the proposals and mispredicted (trained-on)
    programs summed over the lineages."""
    accepted = not_held = 0
    for r in RSI_LINEAGES:
        for step in RSI_STEPS:
            if m.get(f"{series}.dev.{r}.g{step}_advanced") == "yes":
                accepted += 1
                not_held += m.get(f"{series}.v.{r}.g{step}.holds") == "no"
    out = {f"{series}.gate.accepted": accepted, f"{series}.gate.not_held": not_held}
    for step in RSI_STEPS:
        for field in ("proposed", "mispredicted"):
            values = [m[k] for r in RSI_LINEAGES if (k := f"{series}.dev.{r}.g{step}_{field}") in m]
            if values:
                out[f"{series}.signal.g{step}.{field}"] = sum(values)
    return out


class NoParser(Exception):
    """A run whose metric prefix names no log shape this script reads: its values cannot be re-extracted."""


def parse_log(prefix, text):
    """The metrics of one raw log, by the shape its prefix names: a probe, a command-line run or
    the throughput probe. Used only to check measurements.json against the logs."""
    m = {}
    if prefix.startswith("rsi1.arm."):
        return rsi1_analysis_metrics(json.loads(text))
    if prefix.startswith("rsi1.rl."):
        return rsi1_rl_metrics(json.loads(text))
    if re.match(r"rsi\d\.v\.", prefix):
        return rsi_verdict_metrics(json.loads(text))
    if re.match(r"rsi\d\.plan$", prefix):
        return rsi_plan_metrics(json.loads(text))
    if re.match(r"rsi\d\.dev\.", prefix):
        return rsi_rounds_metrics(text)
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
    if prefix.startswith(("ce8.shape", "gen27b")):
        # the arm-shape summary is written flat, so the file IS the metrics dict
        return json.loads(text)
    if prefix.startswith("ce8."):
        return ce8_train_metrics(text)
    if prefix.startswith(("d46.", "rcv.", "gen2.")):
        if ".repeat_differences" in prefix:
            return repeat_metrics(json.loads(text))
        if ".compare" in prefix:
            return ce4_compare_metrics(json.loads(text))
        if ".train" in prefix:
            return time_metrics(text)
        if ".behaviour" in prefix:
            return behaviour_metrics(text)
        if ".edit_rate" in prefix:
            return edit_rate_metrics(text)
        return verify_metrics(json.loads(text))
    if prefix.startswith("g8.det."):
        return det_metrics(prefix, json.loads(text))
    if prefix.startswith(("g8.proto6", "g8.qual")):
        return proto6_metrics(json.loads(text))
    if prefix.startswith("g8.traindet"):
        d = json.loads(text)
        return {
            "steps": int(d["run_a"]["steps"] and 6),
            "distinct_adapters": 2 if d["run_a_adapter"] != d["run_b_adapter"] else 1,
            "run_a_monitor": d["run_a"]["monitor"],
            "run_b_monitor": d["run_b"]["monitor"],
        }
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
    elif prefix.startswith("thr."):
        m["tokens_per_s"] = int(re.search(r"PROBE warm_tokens_per_s=(\d+)", text).group(1))
        m["step_s"] = re.findall(r"step=3 .*?step_s=([\d.]+)", text)[0]
        m["peak_mib"] = int(re.search(r"PROBE peak_mib=(\d+)", text).group(1))
    else:
        raise NoParser(f"no parser reads the log shape of metric prefix {prefix!r}")
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
        try:
            extracted = parse_log(run["metric_prefix"], text)
        except NoParser as error:
            problems.append(f"{name}: {error}; its values are not re-checked against {run['log']}")
            continue
        if extracted != run["metrics"]:
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
    m.update(prefixed_metrics("d46."))
    m.update(prefixed_metrics("rcv."))
    m.update(prefixed_metrics("gen2."))
    m.update(prefixed_metrics("lua."))
    m.update(prefixed_metrics("gen27b"))
    m.update(prefixed_metrics("rsi"))
    for series in ("rsi2", "rsi3", "rsi4"):
        m.update(rsi_pooled_metrics(series, m))
        m.update(rsi_gate_metrics(series, m))
    m.update(frontier_entry_metrics())
    return m


def frontier_entry_metrics(samples=4):
    """COMPUTED: the chance that RSI4's frontier rule (a program enters when at least one of `samples`
    sampled predictions is wrong) admits a program the champion predicts right with probability q per
    sample, 1 - q**samples, at q = 0.5 and 0.9."""
    return {f"rsi4.frontier.entry_q{int(100 * q)}": f"{1 - q ** samples:.2f}" for q in (0.5, 0.9)}


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
# The registered comparisons that have a row in data/results.csv (prereg/coding-tasks.md).
COMPARISONS = ("CE4", "CE8")


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


# Development probes, not registered experiments: rows of data/results.csv whose experiment is one
# of these are rendered by pilot_results_table, and their status must stay PILOT.
PILOTS = ("D46-SS", "D46-RCV", "D46-GEN2", "S1-GEN2")


def pilot_results_table(rows=None):
    """The paired outcomes of the development probes, from their rows of data/results.csv (or the
    rows passed, for the test): the exact one-sided sign test is computed from the wins and losses,
    never typed."""
    rows = read_results() if rows is None else rows
    mine = [r for r in rows if r["experiment"] in PILOTS]
    if not mine:
        return r"\textbf{NOT MEASURED}"
    lines = [r"\begin{tabular}{lp{5.4cm}rrrrl}", r"\toprule",
             r"Probe & Arm & Families & Wins & Losses & $p$ & Status \\", r"\midrule"]
    for r in mine:
        wins, losses = int(r["wins"]), int(r["losses"])
        lines.append(f"{r['experiment']} & {r['arm']} & {r['exam_families']} & {wins} & {losses} "
                     f"& {upper_tail(wins, wins + losses):.3g} & {r['status']} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def tex_p(p):
    """A p value string of fmt_p as LaTeX: 1.8e-22 as 1.8 times ten to the -22."""
    mantissa, _, exponent = p.partition("e")
    return f"${mantissa}{{\\times}}10^{{{int(exponent)}}}$" if exponent else p


def rsi_step_rows(series, label, m):
    """One row per lineage of a series: items right at the base and each generation (the base as the
    examiner measured it at step 1), each step's wins/losses and sign test, and whether every step held."""
    rows = []
    for r in RSI_LINEAGES:
        key = f"{series}.v.{r}"
        if f"{key}.g1.right" not in m:
            continue
        right = [m[f"{key}.g1.predecessor_right"]] + [m[f"{key}.g{g}.right"] for g in RSI_STEPS]
        steps = [f"{m[f'{key}.g{g}.wins']}/{m[f'{key}.g{g}.losses']} & {tex_p(m[f'{key}.g{g}.p'])}" for g in RSI_STEPS]
        held = all(m[f"{key}.g{g}.holds"] == "yes" for g in RSI_STEPS)
        rows.append(f"{r} & " + " & ".join(map(str, right)) + " & " + " & ".join(steps)
                    + f" & {'yes' if held else 'no'} \\\\")
    return "\n".join(rows)


def rsi_table(m=None):
    """The per-step outcomes of RSI3 and RSI4 from the examiner's verdicts, with the pooled steps
    computed from them and the plan verdict's outcome."""
    m = metrics() if m is None else m
    lines = [r"\begin{tabular}{@{}l rrrr rr rr rr l@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Items right of 889} & \multicolumn{2}{c}{Gen.~1 vs base}"
             r" & \multicolumn{2}{c}{Gen.~2 vs gen.~1} & \multicolumn{2}{c}{Gen.~3 vs gen.~2} & \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"Lineage & base & g1 & g2 & g3 & W/L & $p$ & W/L & $p$ & W/L & $p$ & All hold \\", r"\midrule"]
    for series, label in RSI_SERIES:
        if f"{series}.plan.plan_met" not in m:
            lines.append(rf"\multicolumn{{12}}{{l}}{{{label}: \textbf{{NOT MEASURED}}}} \\")
            continue
        lines.append(rf"\multicolumn{{12}}{{@{{}}l}}{{\emph{{{label}}}}} \\")
        lines.append(rsi_step_rows(series, label, m))
        # summed over lineages that share the exam's items: a description, so no p is printed
        pooled = " & ".join(f"{m[f'{series}.pooled.g{g}.wins']}/{m[f'{series}.pooled.g{g}.losses']} & --"
                            for g in RSI_STEPS)
        verdict = "met" if m[f"{series}.plan.plan_met"] == "true" else "not met"
        lines.append(rf"pooled & \multicolumn{{4}}{{l}}{{{m[f'{series}.plan.succeeded']} of 4 lineages succeed}} & {pooled}"
                     rf" & {verdict} \\")
        lines.append(r"\addlinespace")
    lines[-1:] = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def rsi_items_csv(series, m):
    """Items right per generation (0 = the base) and lineage, for the figure."""
    out = ["gen," + ",".join(RSI_LINEAGES)]
    for g in (0, *RSI_STEPS):
        cells = []
        for r in RSI_LINEAGES:
            key = f"{series}.v.{r}.g{max(g, 1)}.{'predecessor_right' if g == 0 else 'right'}"
            cells.append(str(m.get(key, "nan")))
        out.append(f"{g}," + ",".join(cells))
    return "\n".join(out) + "\n"


def rsi_steps_csv(series, m):
    """Each lineage step's wins and losses (losses negative) on one axis position each, for the figure."""
    out = ["x,lineage,step,wins,losses,holds"]
    x = 0
    for r in RSI_LINEAGES:
        for g in RSI_STEPS:
            key = f"{series}.v.{r}.g{g}"
            if f"{key}.wins" in m:
                out.append(f"{x},{r},{g},{m[key + '.wins']},{-m[key + '.losses']},{1 if m[key + '.holds'] == 'yes' else 0}")
            x += 1
        x += 1
    return "\n".join(out) + "\n"


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
    (out / "results_table.tex").write_text(results_table() + "\n")
    (out / "pilot_results_table.tex").write_text(pilot_results_table() + "\n")
    (out / "ce1_table.tex").write_text(ce1_table() + "\n")
    (out / "design_table.tex").write_text(design_table() + "\n")
    (out / "power_curve.csv").write_text(power_curve())
    (out / "families_table.tex").write_text(families_table() + "\n")
    (out / "rsi_table.tex").write_text(rsi_table(m) + "\n")
    for series, _ in RSI_SERIES:
        (out / f"{series}_items.csv").write_text(rsi_items_csv(series, m))
        (out / f"{series}_steps.csv").write_text(rsi_steps_csv(series, m))
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


def check_rsi_consistency(m, problems):
    """The pooled steps computed from the lineages' step verdicts must equal the plan verdict's, and the
    lineages whose every step holds must be the number the plan verdict counts as succeeded."""
    for series in ("rsi2", "rsi3", "rsi4"):
        if f"{series}.plan.plan_met" not in m:
            continue
        for g in RSI_STEPS:
            for field in ("wins", "losses"):
                computed = m.get(f"{series}.pooled.g{g}.{field}", 0)
                stated = m.get(f"{series}.plan.g{g}_{field}")
                if stated is not None and computed != stated:
                    problems.append(f"{series} step {g}: pooled {field} {computed} from the lineages, {stated} in the plan verdict")
        held = sum(all(m.get(f"{series}.v.{r}.g{g}.holds") == "yes" for g in RSI_STEPS) for r in RSI_LINEAGES)
        if held != m[f"{series}.plan.succeeded"]:
            problems.append(f"{series}: {held} lineages hold at every step, the plan verdict counts {m[f'{series}.plan.succeeded']}")


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
    check_rsi_consistency(m, problems)
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
