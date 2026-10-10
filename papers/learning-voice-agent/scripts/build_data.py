#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements voice agents that learn how to speak from
# their users, with every claim backed by a reproducible measurement, for its
# clients. If your team needs expertise in evaluating spoken dialogue systems,
# you can procure our services by sending an email to info@swedishembedded.com.

"""Builds everything the paper's tables and macros show from data/.

Nothing is typed by hand into the paper. data/runs.csv names every measured
run and the file holding its report; data/claims.csv holds the status of each
claim and the run that supports it. A claim may be marked `met` only with an
evidence run that exists, and a claim with no evidence prints as pending: an
unmeasured number is absent, never zero. Standard library only.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "generated"
STATUSES = {"pending", "partial", "met", "failed", "limitation"}


def rows(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fail(message):
    sys.exit(f"build_data: {message}")


def tex(text):
    """`text` safe in running LaTeX."""
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("<=", "$\\leq$"), (">=", "$\\geq$")):
        text = text.replace(a, b)
    return text


def spread(report, key):
    column = report.get(key)
    if not column or "p50" not in column or "p95" not in column:
        fail(f"a latency report lacks {key}")
    return column["p50"], column["p95"]


def main():
    runs = {r["id"]: r for r in rows("runs.csv")}
    for run in runs.values():
        if not (ROOT / run["file"]).is_file():
            fail(f"run {run['id']}: {run['file']} does not exist")

    claims = rows("claims.csv")
    for claim in claims:
        status, evidence = claim["status"], claim["evidence"]
        if status not in STATUSES:
            fail(f"claim {claim['id']}: status {status!r} is not one of {sorted(STATUSES)}")
        if status in {"met", "partial"} and evidence not in runs:
            fail(f"claim {claim['id']} is {status} but its evidence {evidence!r} is not a run in runs.csv")
        if status == "pending" and evidence:
            fail(f"claim {claim['id']} is pending yet names evidence")

    OUT.mkdir(exist_ok=True)
    macros = {}

    # Claims table.
    lines = ["\\begin{tabular}{@{}l p{6.2cm} l p{4.6cm}@{}}", "\\toprule", "Claim & What is claimed & Status & Gate\\\\", "\\midrule"]
    for claim in claims:
        lines.append(f"{claim['id']} & {tex(claim['claim'])} & {claim['status']} & {tex(claim['gate'])}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "claims_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # First-sound latency, from the stored run.
    run = runs["latency-stream-20261009"]
    report = json.loads((ROOT / run["file"]).read_text(encoding="utf-8"))
    n = report["turns"]
    if str(n) != run["n"]:
        fail(f"latency run: the report holds {n} turns, runs.csv says {run['n']}")
    table = ["\\begin{tabular}{@{}l r r@{}}", "\\toprule", "Stage (seconds from the end of the question) & p50 & p95\\\\", "\\midrule"]
    for label, key in (("first word written", "first_text_seconds"), ("first piece handed to be spoken", "first_piece_seconds"), ("first sound", "first_audio_seconds"), ("whole turn", "total_seconds")):
        p50, p95 = spread(report, key)
        table.append(f"{label} & {p50:.2f} & {p95:.2f}\\\\")
        macros[{"first_audio_seconds": "FirstSound", "first_text_seconds": "FirstWord", "first_piece_seconds": "FirstPiece", "total_seconds": "WholeTurn"}[key] + "Pfifty"] = f"{p50:.2f}"
        macros[{"first_audio_seconds": "FirstSound", "first_text_seconds": "FirstWord", "first_piece_seconds": "FirstPiece", "total_seconds": "WholeTurn"}[key] + "Pninetyfive"] = f"{p95:.2f}"
    table += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "latency_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    macros["LatencyTurns"] = str(n)
    macros["LatencyGpu"] = tex(run["gpu"])
    macros["LatencyRunId"] = tex(run["id"])

    # Pilot: the recogniser as a judge of pronunciation. Counted here, not by hand.
    pilot = runs["pilot-names-20261009"]
    items = json.loads((ROOT / pilot["file"]).read_text(encoding="utf-8"))["items"]
    if str(len(items)) != pilot["n"]:
        fail(f"pilot run: the report holds {len(items)} sentences, runs.csv says {pilot['n']}")

    def words(text):
        return "".join(c.lower() if c.isalnum() or c == " " else " " for c in text).split()

    differing = [i for i in items if words(i["text"]) != words(i["heard"])]
    empty = [i for i in differing if not i["heard"].strip()]
    macros["PilotSentences"] = str(len(items))
    macros["PilotDiffering"] = str(len(differing))
    macros["PilotEmpty"] = str(len(empty))
    macros["PilotNonEmptyDiffering"] = str(len(differing) - len(empty))

    # Pilot: a tail of silence before recognition, two recognisers.
    takes = runs["takes-pilot-20261009"]
    summary = json.loads((ROOT / takes["file"]).read_text(encoding="utf-8"))
    if str(summary["takes"]) != takes["n"]:
        fail(f"takes run: the summary holds {summary['takes']} takes, runs.csv says {takes['n']}")
    first, second = "nvidia/nemotron-3.5-asr-streaming-0.6b", "Qwen/Qwen3-ASR-1.7B"
    macros["TakesN"] = str(summary["takes"])
    macros["TakesAgreed"] = str(summary["agreed"])
    macros["TakesContested"] = str(summary["contested"])
    for key, name in (("FirstRecogniser", first), ("SecondRecogniser", second)):
        macros[f"{key}Raw"] = str(summary["differing_raw"][name])
        macros[f"{key}Tailed"] = str(summary["differing_tailed"][name])

    # Machinery gate: the policy learns a planted reward and generalizes to held-out sentences.
    gate = runs["rl-speed-gate-20261010"]
    report = json.loads((ROOT / gate["file"]).read_text(encoding="utf-8"))
    if str(report["held_out_sentences"]) != gate["n"]:
        fail(f"gate run: the report holds {report['held_out_sentences']} held-out sentences, runs.csv says {gate['n']}")
    first_eval, last_eval = report["evals"][0], report["evals"][-1]
    macros["GateUpdates"] = str(last_eval["after_updates"])
    macros["GateGroup"] = str(report["group"])
    macros["GateFramesBefore"] = f"{first_eval['mean_frames_per_word']:.2f}"
    macros["GateFramesAfter"] = f"{last_eval['mean_frames_per_word']:.2f}"
    macros["GateWerBefore"] = f"{first_eval['word_error_rate']:.3f}"
    macros["GateWerAfter"] = f"{last_eval['word_error_rate']:.3f}"
    macros["GateKlLast"] = f"{report['steps'][-1]['mean_kl']:.3f}"

    # The same planted reward and budget under three training methods.
    methods = [("GRPO (group advantages)", "rl-speed-gate-20261010"), ("DPO (best vs worst take)", "rl-speed-dpo-20261010"), ("Best-of-n (imitate the best take)", "rl-speed-best-of-n-20261010")]
    lines = ["\\begin{tabular}{@{}l r r r r@{}}", "\\toprule", "Method & Frames/word before & after & WER before & after\\\\", "\\midrule"]
    for label, rid in methods:
        ev = json.loads((ROOT / runs[rid]["file"]).read_text(encoding="utf-8"))["evals"]
        lines.append(f"{label} & {ev[0]['mean_frames_per_word']:.2f} & {ev[-1]['mean_frames_per_word']:.2f} & {ev[0]['word_error_rate']:.3f} & {ev[-1]['word_error_rate']:.3f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "methods_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Extraction study: the rule-based readers of a user's words, on the frozen test split.
    study = runs["extraction-rules-20261010"]
    readers = json.loads((ROOT / study["file"]).read_text(encoding="utf-8"))["readers"]
    names = {"respell": "pronounce X as Y", "imitate": "talk as follows", "reject": "that sounded wrong", "approve": "that sounded good",
             "retract": "forget that", "confirm_yes": "confirm", "confirm_no": "refuse", "compare_new": "second take better", "compare_old": "first take better"}
    lines = ["\\begin{tabular}{@{}l r r r@{}}", "\\toprule", "Reader & Utterances & Precision & Recall\\\\", "\\midrule"]
    for key, label in names.items():
        t = readers[key]["test"]
        precision = "n/a" if t["precision"] is None else f"{t['precision']:.2f}"
        recall = "n/a" if t["recall"] is None else f"{t['recall']:.2f}"
        lines.append(f"{label} & {t['positives']} & {precision} & {recall}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "extraction_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macros["ExtractFalsePositives"] = str(sum(r["test"]["false_positives"] for r in readers.values()))
    macros["ExtractTestSize"] = study["n"]

    (OUT / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8")


if __name__ == "__main__":
    main()
