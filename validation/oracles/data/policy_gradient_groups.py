# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for knowing exactly which attempts a reinforcement-learning dataset
# compares with which, for its clients. If your team needs expertise in validating agent training data then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Oracle `data/policy-gradient-groups`: a second, deliberately plain implementation of how the policy-gradient view
groups attempts and which of them it writes (plan 6B), written from the view's documentation
(crates/data/src/views/policy_gradient.rs), never from its code.

A case is a list of conversations in corpus order. Each names its trial, the task version it attempted, the model
that ran it (the policy), the trial's verdict (pass or fail) and its role; a trial may have several conversations,
one per role. Every conversation is one tool call followed by a final answer. The expected value is the records the
view writes, in order, each as its trial, its reward and its group (numbered in the order groups first appear), and
how many experiences it left out and why.

The rules (each a sentence of the documentation):
  * the reward of a trial is 1.0 when it passed and 0.0 when it failed (no shaping, no over-length penalty);
  * a group is the trials of one task version made by one policy; each trial counts once, however many
    conversations it had;
  * a group of one trial, or whose trials all earned the same, says nothing: each of its experiences is left out as
    NoGroupSignal;
  * every action of every conversation of a written trial is a record carrying the trial's reward, its group and
    the trial's name, in the corpus's order; an action is a step with calls, or the last agent step when it calls
    nothing, so each conversation here has two.

Usage: python -I policy_gradient_groups.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

NAME = "data/policy-gradient-groups"
REFERENCE = "reference-projector"
ACTIONS_PER_CONVERSATION = 2


def draw_case(rng):
    trials = []
    for t in range(rng.randint(2, 8)):
        trials.append({
            "trial": f"t{t}",
            "version": rng.choice(["1", "2"]),
            "model": rng.choice(["m1", "m2"]),
            "verdict": rng.choice(["pass", "fail"]),
            "roles": rng.choice([["designer"], ["designer"], ["designer", "checker"]]),
        })
    conversations = [
        {"trial": t["trial"], "version": t["version"], "model": t["model"], "verdict": t["verdict"], "role": role}
        for t in trials for role in t["roles"]
    ]
    rng.shuffle(conversations)
    return {"conversations": conversations}


def project(case):
    conversations = case["conversations"]
    reward = {}
    group_of = {}
    for c in conversations:
        reward[c["trial"]] = 1.0 if c["verdict"] == "pass" else 0.0
        group_of[c["trial"]] = (c["version"], c["model"])
    members = {}
    for trial, group in group_of.items():
        members.setdefault(group, set()).add(trial)
    def says_something(group):
        trials = members[group]
        return len(trials) > 1 and len({reward[t] for t in trials}) > 1
    numbers = {}
    records = []
    left_out = 0
    for c in conversations:
        group = group_of[c["trial"]]
        if not says_something(group):
            left_out += 1
            continue
        number = numbers.setdefault(group, len(numbers))
        for _ in range(ACTIONS_PER_CONVERSATION):
            records.append({"trial": c["trial"], "reward": reward[c["trial"]], "group": number})
    excluded = {"NoGroupSignal": left_out} if left_out else {}
    return {**case, "records": records, "excluded": excluded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=400)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261125 if args.seed is None else args.seed)
    cases = [project(draw_case(rng)) for _ in range(args.cases)]
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
