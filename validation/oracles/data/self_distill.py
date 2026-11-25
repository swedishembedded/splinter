# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for knowing exactly which failed attempts a self-distillation dataset
# teaches from, and with what, for its clients. If your team needs expertise in on-policy distillation data then you
# can procure our services by sending an email to info@swedishembedded.com.
"""Oracle `data/self-distill`: a second, deliberately plain implementation of which attempts the policy-gradient view
distils and which carry their group's reward (plan 6B), written from the view's documentation
(crates/data/src/views/policy_gradient.rs, its self-distillation section, and crates/data/src/views/privileged.rs),
never from its code.

A case is a distill setting (off, cliff, every) and conversations in corpus order, as in data/policy-gradient-groups:
each names its trial, the task version, the policy, the verdict, its role, and what the verifier said of the trial
(a reason, or nothing). The expected value is the records the view writes, in order, and what it leaves out, by
reason. A record is its trial and either its reward and group (numbered in the order the rewarded groups first
appear) or the teacher context's two facts: whether it says the attempt passed, and the verifier's reason.

The rules (each a sentence of the documentation):
  * groups, rewards and a group's signal are as in data/policy-gradient-groups; a group with no signal whose trials
    all failed is the cliff (all-fail), one with no signal of which some trial passed is all-pass;
  * off: a group with a signal writes each action with its reward; any other group writes nothing and each of its
    experiences is left out as NoGroupSignal;
  * cliff: the trials of an all-fail group are distilled, a group with a signal keeps its rewards, an all-pass group
    writes nothing (NoGroupSignal);
  * every: every trial of every group is distilled, none carries a reward;
  * a distilled attempt is written with a teacher context only when the verifier said something of it (here, a
    reason; these tasks carry no reference and no hidden check); otherwise each of its experiences is left out as
    NoPrivilegedContext. The context says whether the attempt passed and quotes the reason.

Usage: python -I self_distill.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

NAME = "data/self-distill"
REFERENCE = "reference-projector"
ACTIONS_PER_CONVERSATION = 2
REASONS = [None, None, "the bracket is 2 mm too thin", "no part file was written"]


def draw_case(rng):
    trials = []
    for t in range(rng.randint(1, 7)):
        trials.append({
            "trial": f"t{t}",
            "version": rng.choice(["1", "2"]),
            "model": rng.choice(["m1", "m2"]),
            "verdict": rng.choice(["pass", "fail", "fail"]),
            "reason": rng.choice(REASONS),
            "roles": rng.choice([["designer"], ["designer"], ["designer", "checker"]]),
        })
    conversations = [
        {k: t[k] for k in ("trial", "version", "model", "verdict", "reason")} | {"role": role}
        for t in trials for role in t["roles"]
    ]
    rng.shuffle(conversations)
    return {"distill": rng.choice(["off", "cliff", "every"]), "conversations": conversations}


def project(case):
    conversations = case["conversations"]
    reward, group_of = {}, {}
    for c in conversations:
        reward[c["trial"]] = 1.0 if c["verdict"] == "pass" else 0.0
        group_of[c["trial"]] = (c["version"], c["model"])
    members = {}
    for trial, group in group_of.items():
        members.setdefault(group, set()).add(trial)

    def kind(group):
        trials = members[group]
        if len(trials) > 1 and len({reward[t] for t in trials}) > 1:
            return "graded"
        return "all-pass" if any(reward[t] == 1.0 for t in trials) else "all-fail"

    def distilled(group):
        return case["distill"] == "every" or (case["distill"] == "cliff" and kind(group) == "all-fail")

    numbers, records, excluded = {}, [], {}

    def leave_out(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for c in conversations:
        group = group_of[c["trial"]]
        if distilled(group):
            if c["reason"] is None:
                leave_out("NoPrivilegedContext")
                continue
            record = {"trial": c["trial"], "passed": c["verdict"] == "pass", "reason": c["reason"]}
        elif kind(group) == "graded":
            record = {"trial": c["trial"], "reward": reward[c["trial"]],
                      "group": numbers.setdefault(group, len(numbers))}
        else:
            leave_out("NoGroupSignal")
            continue
        records.extend(dict(record) for _ in range(ACTIONS_PER_CONVERSATION))
    return {**case, "records": records, "excluded": excluded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=400)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261127 if args.seed is None else args.seed)
    cases = [project(draw_case(rng)) for _ in range(args.cases)]
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
