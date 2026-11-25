# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for knowing exactly what a teacher's repair of a failed attempt teaches,
# for its clients. If your team needs expertise in preference data from corrected agent runs then you can procure our
# services by sending an email to info@swedishembedded.com.
"""Oracle `data/repair-pairs`: a second, deliberately plain implementation of the preference view's third source,
repairs (plan 6B), written from its documentation (crates/data/src/views/preference.rs, item 3 and the paragraph on
where two conversations part), never from its code.

A case is the attempts of one task in corpus order. A student's attempt failed or was an infrastructure failure; a
teacher's continuation copied the first one or two actions of some trajectory and then went on, and passed or failed.
Every attempt is shell commands, each an action with its result, then a final answer (`Booked.` after a pass,
`Done.` after a fail). The expected value is the records the view writes, in order, each the chosen and the rejected
trial, the shared prompt and each side's first message, and what it leaves out, by reason.

The rules (each a sentence of the documentation):
  * an infrastructure failure is never a candidate (InfrastructureFailure);
  * the students' verified fails form one group with nothing that passed: each is left out as NoCounterpart;
  * a continuation of another trajectory's prefix that failed is never a candidate (Failed); one that passed is a
    repair, paired with every verified fail of the task in corpus order, or left out as NoCounterpart when there is
    none;
  * a pair is made from where the two conversations first differ: the messages before are the shared prompt and
    each side from there is a candidate; when they never differ it is left out as IdenticalOutputs, or as
    NoDivergence when one is the other's beginning; when they differ before the repair's own part (inside its copied
    prefix, so the prefix is not the failure's own) it is left out as PrefixMismatch.

A conversation is the system turn, the instruction, each command (an assistant turn whose content is its arguments
as compact JSON) with its result (`ran <command>`), and the final answer.

Usage: python -I repair_pairs.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

NAME = "data/repair-pairs"
REFERENCE = "reference-projector"
COMMANDS = ["cat ledger.csv", "ls", "book accrual", "book wrong"]


def commands(rng):
    return [rng.choice(COMMANDS) for _ in range(rng.randint(1, 3))]


def draw_case(rng):
    students = []
    for s in range(rng.randint(0, 3)):
        students.append({"trial": f"s{s}", "teacher": False, "copied": 0, "commands": commands(rng),
                         "verdict": rng.choice(["fail", "fail", "infrastructure"])})
    repairs = []
    for r in range(rng.randint(1, 3)):
        copied = rng.randint(1, 2)
        ours = commands(rng)
        if students and rng.random() < 0.7:
            # Most repairs continue one student's own prefix.
            prefix = rng.choice(students)["commands"][:copied]
        else:
            prefix = [rng.choice(COMMANDS) for _ in range(copied)]
        repairs.append({"trial": f"r{r}", "teacher": True, "copied": len(prefix), "commands": prefix + ours,
                        "verdict": rng.choice(["pass", "pass", "fail"])})
    attempts = students + repairs
    rng.shuffle(attempts)
    return {"attempts": attempts}


def messages(attempt):
    out = ["system:You keep the books.", "user:Book the accrual."]
    for c in attempt["commands"]:
        out.append("assistant:" + json.dumps({"command": c}, separators=(",", ":")))
        out.append(f"tool:ran {c}")
    out.append("assistant:" + ("Booked." if attempt["verdict"] == "pass" else "Done."))
    return out


def project(case):
    excluded, records = {}, []

    def leave_out(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    fails, repairs = [], []
    for a in case["attempts"]:
        if a["verdict"] == "infrastructure":
            leave_out("InfrastructureFailure")
        elif a["teacher"]:
            if a["verdict"] == "pass":
                repairs.append(a)
            else:
                leave_out("Failed")
        else:
            fails.append(a)
    for _ in fails:
        leave_out("NoCounterpart")
    for repair in repairs:
        if not fails:
            leave_out("NoCounterpart")
            continue
        won = messages(repair)
        own_start = 2 + 2 * repair["copied"]
        for fail in fails:
            lost = messages(fail)
            parted = next((i for i, (a, b) in enumerate(zip(won, lost)) if a != b), None)
            if parted is None:
                leave_out("IdenticalOutputs" if len(won) == len(lost) else "NoDivergence")
                continue
            if parted < own_start:
                leave_out("PrefixMismatch")
                continue
            records.append({"chosen": repair["trial"], "rejected": fail["trial"], "prompt": won[:parted],
                            "chosen_first": won[parted], "rejected_first": lost[parted]})
    return {**case, "records": records, "excluded": excluded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=400)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261129 if args.seed is None else args.seed)
    cases = [project(draw_case(rng)) for _ in range(args.cases)]
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
