# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for knowing exactly which attempts a preference or feedback dataset
# trains on, for its clients. If your team needs expertise in validating agent training data then you can procure
# our services by sending an email to info@swedishembedded.com.
"""Oracle `data/attempt-pairs`: a second, deliberately plain implementation of which attempts the preference view
pairs and which the feedback view keeps (plan 6B), written from the views' documentation
(crates/data/src/views/{attempts,preference,feedback}.rs), never from their code.

A case is the views' minimum strength and a list of attempts of one task version in corpus order. Each names its
trial, the model that ran it (the policy), its verdict (pass, fail, or an infrastructure failure the examiner did
not count) and the strength it was decided at. The expected value is, for the preference view, the set of pairs it
writes (chosen trial, rejected trial) and what it leaves out, and for the feedback view each record's trial and
whether it is desirable, in order, what it leaves out, and the weight of each class.

The rules (each a sentence of the documentation):
  * an attempt is verified when its verdict decides it at the minimum strength or above; an infrastructure failure
    is left out as InfrastructureFailure, a pass decided below it as UnverifiedPass, a fail below it as TooWeak;
  * a verified attempt belongs to a group: one version of one task attempted by one policy;
  * preference: a verified pass and a verified fail of one group, decided at the same strength, are a pair, the
    pass chosen; every such pair is written (no cap); a verified attempt with nothing of the other verdict to pair
    with is left out as NoCounterpart;
  * feedback: each verified attempt is one record, desirable when it passed; the smaller class is weighed up by the
    ratio of the counts so both weigh the same in all, each weight one when a class is empty.

Usage: python -I attempt_pairs.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

NAME = "data/attempt-pairs"
REFERENCE = "reference-projector"
RANK = {"judged": 0, "executable": 1}


def draw_case(rng):
    minimum = rng.choice(["executable", "judged"])
    attempts = []
    for t in range(rng.randint(1, 7)):
        strength = rng.choice(["executable", "executable", "judged"])
        attempts.append({
            "trial": f"t{t}",
            "model": rng.choice(["policy-a", "policy-b"]),
            "verdict": rng.choice(["pass", "pass", "fail", "fail", "infrastructure"]),
            "strength": strength,
        })
    return {"minimum": minimum, "attempts": attempts}


def count(excluded, reason):
    excluded[reason] = excluded.get(reason, 0) + 1


def verified(case, excluded):
    out = []
    for a in case["attempts"]:
        if a["verdict"] == "infrastructure":
            count(excluded, "InfrastructureFailure")
        elif RANK[a["strength"]] < RANK[case["minimum"]]:
            count(excluded, "UnverifiedPass" if a["verdict"] == "pass" else "TooWeak")
        else:
            out.append(a)
    return out


def preference(case):
    excluded = {}
    attempts = verified(case, excluded)
    pairs = []
    for a in attempts:
        partners = [b for b in attempts if b["model"] == a["model"] and b["strength"] == a["strength"]
                    and b["verdict"] != a["verdict"]]
        if not partners:
            count(excluded, "NoCounterpart")
        if a["verdict"] == "pass":
            pairs.extend([a["trial"], b["trial"]] for b in partners)
    return {"pairs": sorted(pairs), "excluded": excluded}


def feedback(case):
    excluded = {}
    attempts = verified(case, excluded)
    records = [{"trial": a["trial"], "desirable": a["verdict"] == "pass"} for a in attempts]
    desirable = sum(r["desirable"] for r in records)
    undesirable = len(records) - desirable
    def weight(fewer, more):
        return more / fewer if fewer and more > fewer else 1.0
    balance = {"desirable": desirable, "undesirable": undesirable,
               "desirable_weight": weight(desirable, undesirable),
               "undesirable_weight": weight(undesirable, desirable)}
    return {"records": records, "excluded": excluded, "balance": balance}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=400)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261126 if args.seed is None else args.seed)
    cases = []
    for _ in range(args.cases):
        case = draw_case(rng)
        cases.append({**case, "preference": preference(case), "feedback": feedback(case)})
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
