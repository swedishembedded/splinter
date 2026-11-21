#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements paired comparisons of agent models whose
# statistics are proven against exact references, for its clients. If your
# team needs expertise in telling a real gain of a fine-tuned agent from the
# luck of a small exam, you can procure our services by sending an email to
# info@swedishembedded.com.
"""Write arm_compare.jsonl (trials of three arms) and arm_compare.json (what
`experiences compare-arms` must report of them), for
crates/pipelines/src/arm_compare_tests.rs.

Every expected value is exact rational arithmetic from the definitions
(fractions, math.comb), independent of splinter's code:

* pass@k per arm: the mean over tasks with n >= k graded attempts of
  1 - C(n - c, k) / C(n, k); timeouts are graded, infrastructure errors leave
  the denominator;
* per contrast over the tasks graded in both arms: the mean per-task
  difference of pass rates, the better / worse / tied counts, the exact
  one-sided P(X >= better) and two-sided min(1, 2 min(P(X >= b), P(X <= b)))
  sign tests, Holm over the contrasts;
* the exact bootstrap distributions of the mean difference: over tasks
  (multinomial counts of the distinct per-task differences) and over whole
  families (multinomial counts of the families, the statistic the mean over
  every task drawn), as CDFs whose 2.5% and 97.5% quantiles the reported
  bounds must be within Monte Carlo error of.

The trials are drawn from a fixed table, not at random, so the file is the
same on every run.

Run: python3 -I crates/pipelines/tests/fixtures/conformance/generate_arm_compare.py
(standard library only; rewrites both files next to this script).
"""
import json
import math
import os
from fractions import Fraction

# Per task: family, then per arm the verdicts of its attempts ("p" pass, "f"
# fail, "t" timeout that failed, "T" timeout that passed, "x" infrastructure
# error).
TASKS = {
    "t01": ("fa", {"base": "ffff", "a": "pfpf", "b": "pppp"}),
    "t02": ("fa", {"base": "pfff", "a": "ppff", "b": "pppf"}),
    "t03": ("fb", {"base": "ffft", "a": "ffff", "b": "pfpf"}),
    "t04": ("fb", {"base": "pppp", "a": "pppp", "b": "pppp"}),
    "t05": ("fc", {"base": "ppff", "a": "pppT", "b": "ffff"}),
    "t06": ("fc", {"base": "fffx", "a": "pffx", "b": "pfff"}),
    "t07": ("fd", {"base": "ffff", "a": "ffff", "b": "ffff"}),
    "t08": ("fd", {"base": "pfff", "a": "pppf", "b": "ppff"}),
    "t09": ("fe", {"base": "ppfx", "a": "ppff", "b": "pppp"}),
    "t10": ("fe", {"base": "ffff", "a": "pfff", "b": "ffff"}),
    "t11": ("ff", {"base": "ppff", "a": "ffff", "b": "pppp"}),
    "t12": ("ff", {"base": "pfff", "a": "ppfx", "b": "pfpf"}),
    # Graded only in arm a: not paired with base.
    "t13": ("ff", {"a": "pppp"}),
}
REFERENCE = "base"
KS = [1, 2, 4]
LEVEL = Fraction(95, 100)


def graded(verdicts):
    return [v in "pT" for v in verdicts if v != "x"]


def pass_at_k(n, c, k):
    return 1 - Fraction(math.comb(n - c, k), math.comb(n, k))


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def cdf(pmf):
    out, running = [], Fraction(0)
    for key in sorted(pmf):
        running += pmf[key]
        out.append([float(key), float(running)])
    return out


def task_bootstrap(diffs):
    values = sorted(set(diffs))
    n = len(diffs)
    pmf = {}
    for counts in compositions(n, len(values)):
        coef = math.factorial(n)
        for c in counts:
            coef //= math.factorial(c)
        p = Fraction(coef, n**n)
        for c, v in zip(counts, values):
            p *= diffs.count(v) ** c
        mean = sum(c * v for c, v in zip(counts, values)) / n
        pmf[mean] = pmf.get(mean, Fraction(0)) + p
    return cdf(pmf)


def family_bootstrap(families):
    k = len(families)
    pmf = {}
    for counts in compositions(k, k):
        coef = math.factorial(k)
        for c in counts:
            coef //= math.factorial(c)
        n = sum(c * len(f) for c, f in zip(counts, families))
        ratio = sum(c * sum(f) for c, f in zip(counts, families)) / n
        pmf[ratio] = pmf.get(ratio, Fraction(0)) + Fraction(coef, k**k)
    return cdf(pmf)


def sign(better, worse):
    n = better + worse
    if n == 0:
        return Fraction(1), Fraction(1)
    upper = Fraction(sum(math.comb(n, i) for i in range(better, n + 1)), 2**n)
    lower = Fraction(sum(math.comb(n, i) for i in range(0, better + 1)), 2**n)
    return upper, min(Fraction(1), 2 * min(upper, lower))


def holm(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted, running = [None] * m, Fraction(0)
    for rank, i in enumerate(order):
        running = max(running, min(Fraction(1), (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    lines = []
    for task, (family, arms) in TASKS.items():
        for arm, verdicts in arms.items():
            for attempt, v in enumerate(verdicts):
                line = {"task": task, "family": family, "arm": arm, "attempt": attempt}
                if v == "x":
                    line["status"] = "infra_error"
                else:
                    line["status"] = "timeout" if v in "tT" else "graded"
                    # Half the lines give a reward, half a pass flag.
                    if attempt % 2:
                        line["reward"] = 1.0 if v in "pT" else 0.0
                    else:
                        line["pass"] = v in "pT"
                lines.append(line)
    with open(os.path.join(here, "arm_compare.jsonl"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    arms = sorted({a for _, (_, per) in TASKS.items() for a in per})
    expected = {"reference": REFERENCE, "k": KS, "level": float(LEVEL), "arms": {}, "contrasts": {}}
    by_arm = {a: {t: graded(per[a]) for t, (_, per) in TASKS.items() if a in per} for a in arms}
    for arm in arms:
        verdicts = "".join(per[arm] for _, (_, per) in TASKS.items() if arm in per)
        tasks = {t: v for t, v in by_arm[arm].items() if v}
        pk = {}
        for k in KS:
            values = [pass_at_k(len(v), sum(v), k) for v in tasks.values() if len(v) >= k]
            pk[str(k)] = {"value": float(sum(values) / len(values)), "tasks": len(values)}
        expected["arms"][arm] = {
            "trials": len(verdicts),
            "graded": sum(v != "x" for v in verdicts),
            "passed": sum(v in "pT" for v in verdicts),
            "timeouts": sum(v in "tT" for v in verdicts),
            "timeouts_passed": verdicts.count("T"),
            "infra_errors": verdicts.count("x"),
            "tasks": len(tasks),
            "pass_at_k": pk,
        }
    p_one, p_two, names = [], [], []
    for arm in arms:
        if arm == REFERENCE:
            continue
        paired = sorted(set(by_arm[arm]) & set(by_arm[REFERENCE]))
        rate = lambda v: Fraction(sum(v), len(v))
        diffs = [rate(by_arm[arm][t]) - rate(by_arm[REFERENCE][t]) for t in paired]
        better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        one, two = sign(better, worse)
        fam = {}
        for t, d in zip(paired, diffs):
            fam.setdefault(TASKS[t][0], []).append(d)
        expected["contrasts"][arm] = {
            "paired_tasks": len(paired),
            "arm_pass_at_1": float(sum(rate(by_arm[arm][t]) for t in paired) / len(paired)),
            "reference_pass_at_1": float(sum(rate(by_arm[REFERENCE][t]) for t in paired) / len(paired)),
            "difference": float(sum(diffs) / len(diffs)),
            "better": better,
            "worse": worse,
            "ties": len(diffs) - better - worse,
            "families": len(fam),
            "p_one_sided": float(one),
            "p_two_sided": float(two),
            "task_cdf": task_bootstrap(diffs),
            "family_cdf": family_bootstrap(list(fam.values())),
        }
        p_one.append(one)
        p_two.append(two)
        names.append(arm)
    for name, h1, h2 in zip(names, holm(p_one), holm(p_two)):
        expected["contrasts"][name]["p_one_sided_holm"] = float(h1)
        expected["contrasts"][name]["p_two_sided_holm"] = float(h2)
    expected["provenance"] = (
        "crates/pipelines/tests/fixtures/conformance/generate_arm_compare.py: exact rational "
        "arithmetic from the definitions; independent of splinter's code"
    )
    with open(os.path.join(here, "arm_compare.json"), "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
