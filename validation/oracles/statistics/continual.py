# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for measuring what a continually trained model keeps and loses, for its
# clients. If your team needs expertise in continual-learning evaluation then you can procure our services by sending
# an email to info@swedishembedded.com.
"""Oracle `statistics/continual`: the continual-learning metrics of a results matrix (plan 12E), by numpy, written
from their documentation (crates/eval/src/continual.rs), never from its code.

`A[k][i]` is the score on task `i` after generation `k`, an unmeasured cell absent (NaN here, never 0); task `i` was
learned at generation `g_i` and belongs to a family (the unit); its base score may be unmeasured. Each metric is a
per-task value over the tasks that have what it needs, then the mean over units of each unit's mean (units in name
order), with the counts of tasks and units; a metric no task has what it needs for is not measured.

  * ACC: A[K-1][i];
  * BWT: A[K-1][i] - A[g_i][i], tasks with g_i < K-1;
  * forgetting: max over g_i <= k < K-1 of A[k][i] (measured cells only) - A[K-1][i], tasks with g_i < K-1;
  * forward transfer: A[g_i - 1][i] - base_i, tasks with g_i > 0;
  * learning gain: A[g_i][i] - base_i;
  * the gain curve at k: A[k][i] - base_i over tasks with g_i <= k.

Usage: python -I continual.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

import numpy

NAME = "statistics/continual"
REFERENCE = "numpy-continual"


def draw(rng):
    generations = rng.randint(1, 5)
    tasks = []
    for t in range(rng.randint(1, 7)):
        tasks.append({"id": f"t{t}", "unit": rng.choice(["fa", "fb", "fc"]),
                      "learned_at": rng.randrange(generations),
                      "base": None if rng.random() < 0.2 else rng.choice([0.0, 0.25, 0.5, 0.125])})
    cells = [[None if rng.random() < 0.15 else rng.randint(0, 16) / 16 for _ in tasks] for _ in range(generations)]
    return {"generations": generations, "tasks": tasks, "cells": cells}


def measure(values, units):
    """The metric of per-task `values` (NaN where a task lacks what it needs)."""
    present = ~numpy.isnan(values)
    if not present.any():
        return {"status": "not_measured"}
    names = sorted({u for u, p in zip(units, present) if p})
    unit_values = [float(numpy.mean(values[present & (units == name)])) for name in names]
    return {"status": "measured", "value": float(numpy.mean(unit_values)), "tasks": int(present.sum()),
            "units": len(names), "unit_values": unit_values}


def metrics(case):
    k_last = case["generations"] - 1
    a = numpy.array([[numpy.nan if c is None else c for c in row] for row in case["cells"]], dtype=float)
    learned = numpy.array([t["learned_at"] for t in case["tasks"]])
    base = numpy.array([numpy.nan if t["base"] is None else t["base"] for t in case["tasks"]], dtype=float)
    units = numpy.array([t["unit"] for t in case["tasks"]])
    n = len(case["tasks"])
    cols = numpy.arange(n)
    early = learned < k_last
    bwt = numpy.where(early, a[k_last] - a[learned, cols], numpy.nan)
    peak = numpy.full(n, numpy.nan)
    for i in cols:
        window = a[learned[i]:k_last, i]
        if early[i] and (~numpy.isnan(window)).any():
            peak[i] = numpy.nanmax(window)
    forgetting = numpy.where(early, peak - a[k_last], numpy.nan)
    before = numpy.where(learned > 0, learned - 1, 0)
    forward = numpy.where(learned > 0, a[before, cols] - base, numpy.nan)
    curve = [{"generation": k, "gain": measure(numpy.where(learned <= k, a[k] - base, numpy.nan), units)}
             for k in range(case["generations"])]
    return {
        "acc": measure(a[k_last].copy(), units),
        "bwt": measure(bwt, units),
        "forgetting": measure(forgetting, units),
        "forward_transfer": measure(forward, units),
        "learning_gain": measure(a[learned, cols] - base, units),
        "gain_curve": curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=400)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261130 if args.seed is None else args.seed)
    cases = []
    for _ in range(args.cases):
        case = draw(rng)
        cases.append({**case, "metrics": metrics(case)})
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
