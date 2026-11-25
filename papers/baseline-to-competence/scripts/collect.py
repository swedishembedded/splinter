#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines and reporting for its clients. If your team needs
# expertise in building papers whose every number can be regenerated from the repository, you can procure our
# services by sending an email to info@swedishembedded.com.
"""Writes the paper's measured counts (data/exam.csv, train.csv, behaviour.csv) from splinter run directories.

usage: collect.py SPEC.json DATA_DIR

SPEC maps a model key to its runs, e.g.
  {"qwen3-4b": {"exam": {"run": "runs/vim-sft-qwen3-4b-full", "arm": "full"},
                "train": [{"run": "runs/vim-base-train-qwen3-4b", "stage": "base", "arm": "base"}], ...}}
Every count comes from `scripts/run-metrics.py --json` of the run, so the paper's numbers are the run's own.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

METRICS = Path(__file__).resolve().parents[3] / "scripts" / "run-metrics.py"


def metrics(run):
    out = subprocess.run([sys.executable, "-I", str(METRICS), run, "--json"], capture_output=True, text=True, check=True)
    return {r["arm"]: r for r in json.loads(out.stdout)}


def within(arms, arm):
    """Tasks solved within k=1..5: the first attempt, then the reruns of the failed."""
    first = arms[f"eval-exam/{arm}"]
    sets = [set(first["solved_sets"][0])]
    rerun = arms.get(f"eval-exam/{arm}-reruns")
    counts = [len(sets[0])]
    if rerun:
        for s in rerun["solved_sets"]:
            counts.append(len(sets[0] | set(s)))
    while len(counts) < 5:
        counts.append(counts[-1])
    return counts[:5]


def main(spec_path, data):
    spec = json.load(open(spec_path))
    exam, train, behaviour = [], [], []
    for model, runs in spec.items():
        for e in runs.get("exam", []):
            arms = metrics(e["run"])
            # the baseline arm is the base model unless the run names what it was measured against
            for side, name in (("baseline", e.get("baseline_arm", "base")), ("candidate", e["arm"])):
                if f"eval-exam/{side}" in arms and not (side == "baseline" and e.get("skip_base")):
                    exam.append([model, name, *within(arms, side)])
                    a, r = arms[f"eval-exam/{side}"], arms.get(f"eval-exam/{side}-reruns")
                    trials = a["slots"] + (r["slots"] if r else 0)
                    loops = a["loop_stops"] + (r["loop_stops"] if r else 0)
                    behaviour.append([model, name, "exam", loops, trials, a["repeat_after_error"]["share"]])
        for t in runs.get("train", []):
            arms = metrics(t["run"])
            a = arms[t["stage"]]
            train.append([model, t["arm"], a["passed"], a["slots"]])
            behaviour.append([model, t["arm"], "train", a["loop_stops"], a["slots"], a["repeat_after_error"]["share"]])
    curve = []
    for model, runs in spec.items():
        for c in runs.get("curve", []):
            t = json.load(open(Path(c["run"]) / "lora-r16" / "training.json"))
            o = t.get("outputs", t)
            base = (o.get("base_score") or {}).get("loss")
            curve.append([model, c["arm"], 0, base])
            curve.extend([model, c["arm"], p["step"], p["monitor_loss"]] for p in o["curve"])
    data = Path(data)
    for name, header, rows in (
        ("exam.csv", ["model", "arm", "k1", "k2", "k3", "k4", "k5"], exam),
        ("train.csv", ["model", "arm", "passed", "trials"], train),
        ("behaviour.csv", ["model", "arm", "set", "loop_trials", "trials", "repeat_after_error"], behaviour),
        ("curve.csv", ["model", "arm", "step", "loss"], curve),
    ):
        with open(data / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:3])
