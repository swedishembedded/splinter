# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for knowing exactly which recorded judgements a preference dataset trains
# on, for its clients. If your team needs expertise in validating preference data then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Oracle `data/recorded-preferences`: a second, deliberately plain implementation of the preference view's first
source, recorded preferences (plan 6B), written from its documentation (crates/data/src/views/preference.rs, item 1),
never from its code.

A case is a list of experiences. Each attempted one of two tasks, sometimes as a retry given a critique of another
attempt; it has a final output or none, its last agent step calls a tool or not, it is in the corpus or not, and it
records that it is preferred over some others (by index; possibly one not in the corpus, possibly the same twice,
possibly itself). No experience carries a verdict, so the view's other sources make nothing. The expected value is
the records the view writes, in order, and what the recorded source leaves out, by reason.

The rules (each a sentence of the documentation):
  * every relation of every experience in the corpus, in corpus order, is a candidate; one relation recorded twice
    is one judgement, the second left out as Duplicate;
  * it is left out as RelatedMissing when the rejected experience is not in the corpus, as DifferentTask when the
    two attempted different tasks (a critique alone does not make another task), as NoFinalOutput when either lacks
    a final output, as IdenticalOutputs when the two replies are the same;
  * otherwise the prompt is the chosen experience's instruction and each side is its final output, carrying the
    tool calls its last agent step made.

Usage: python -I recorded_preferences.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

NAME = "data/recorded-preferences"
REFERENCE = "reference-projector"
INSTRUCTIONS = {"A": "What is two plus two?", "B": "Name a prime above ten."}
OUTPUTS = ["4", "four", "5", "11", None]


def draw_case(rng):
    experiences = []
    count = rng.randint(1, 6)
    for _ in range(count):
        experiences.append({
            "task": rng.choice(["A", "A", "B"]),
            "critique": rng.random() < 0.3,
            "output": rng.choice(OUTPUTS),
            "calls": rng.choice([[], [], ["submit"]]),
            "in_corpus": rng.random() < 0.85,
            "prefers": [rng.randrange(count) for _ in range(rng.choice([0, 1, 1, 2, 3]))],
        })
    return {"experiences": experiences}


def project(case):
    experiences = case["experiences"]
    excluded, records, seen = {}, [], set()

    def leave_out(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for i, chosen in enumerate(experiences):
        if not chosen["in_corpus"]:
            continue
        for j in chosen["prefers"]:
            rejected = experiences[j]
            if not rejected["in_corpus"]:
                leave_out("RelatedMissing")
                continue
            if (i, j) in seen:
                leave_out("Duplicate")
                continue
            seen.add((i, j))
            if chosen["task"] != rejected["task"]:
                leave_out("DifferentTask")
                continue
            if chosen["output"] is None or rejected["output"] is None:
                leave_out("NoFinalOutput")
                continue
            better = {"content": chosen["output"], "calls": chosen["calls"]}
            worse = {"content": rejected["output"], "calls": rejected["calls"]}
            if better == worse:
                leave_out("IdenticalOutputs")
                continue
            records.append({"chosen": i, "rejected": j, "prompt": INSTRUCTIONS[chosen["task"]],
                            "better": better, "worse": worse})
    return {**case, "records": records, "excluded": excluded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=400)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261128 if args.seed is None else args.seed)
    cases = [project(draw_case(rng)) for _ in range(args.cases)]
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
