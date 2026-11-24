# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for knowing exactly which tokens a fine-tuning dataset supervises, for its
# clients. If your team needs expertise in validating agent training data then you can procure our services by
# sending an email to info@swedishembedded.com.
"""Oracle `data/sft-projection`: a second, deliberately plain implementation of the sft-step and sft-final views
(plan 6B), written from their documentation (crates/data/src/views/{sft_step,sft_final,rendering}.rs and
crates/data/src/trajectory.rs), never from their code: explicit loops over typed steps, no shared abstraction.

A case is one experience: its task's setting (a workspace, or a closed-book question), its steps after the
instruction, its verdict (pass, fail, or none) and step labels, and the view's options. The expected value is the
records the view writes (every message: role, text, supervised or not, tool calls with their scoped ids and
arguments, the call a tool reply answers) and what it counts as left out. The system turn every record opens with is
written as SYSTEM_PROMPT: the prompt's text is the core crate's, which the test supplies.

The rules (each a sentence of the documentation):
  * a record opens with the system prompt, then the student's turn (the instruction alone);
  * an agent step with calls is the assistant's turn (its trimmed text, its calls, ids scoped `s<step>-<id>`)
    followed by one tool turn per result; a reply is the assistant's turn; feedback is the user's turn;
  * an action is a step with calls, or the last agent step when it calls nothing and no feedback answered it;
  * sft-step: a pass trains every action; a failed or ungraded attempt trains only actions labelled good, and
    without any good label is left out (Failed, Undecided); among actions, the first reason applies: a bad label
    (BadStep), after a truncating history read (AfterRepositoryHistory), a history read not kept
    (RepositoryHistory), a call repeating an earlier call with the same result (RepeatedAction), a call of a tool
    not asked for (ActionNotWanted); no action at all is NoAction; unpacked, one record per trained action ending at
    it with only it supervised; packed, one record ending at the last trained action, each trained action supervised;
  * sft-final: a closed-book pass with an output is one record, the output supervised; a workspace episode is left
    out (WorkspaceEpisode), and an attempt that did not pass under its reason.

Usage: python -I sft_projection.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

NAME = "data/sft-projection"
REFERENCE = "reference-projector"

# What each kind of step does: (tool, arguments, result), or None for a reply or feedback.
CALLS = {
    "history": ("shell", {"command": "git log --oneline -3"}, "abc fix"),
    "read": ("read_file", {"path": "src/parse.rs"}, "fn parse() {}"),
    "read-other": ("read_file", {"path": "src/lib.rs"}, "pub mod parse;"),
    "write": ("shell", {"command": "sed -i s/a/b/ src/parse.rs"}, "ok"),
}
KINDS = ["history", "read", "read-other", "write", "say", "feedback"]


def step_id(i):
    return i + 2


def text_of(kind, i):
    if kind == "say":
        return f"  Reply {step_id(i)}.  "
    if kind == "feedback":
        return f"The checks still fail ({step_id(i)})."
    # A call's own text: some calls say something, most nothing.
    return "Looking." if i % 3 == 0 else ""


def messages_of(steps):
    """The conversation, and where each step's assistant turn is."""
    messages = [
        {"role": "system", "content": "SYSTEM_PROMPT", "train": False},
        {"role": "user", "content": "Fix the parser.", "train": False},
    ]
    at = {}
    for i, kind in enumerate(steps):
        sid = step_id(i)
        if kind == "feedback":
            messages.append({"role": "user", "content": text_of(kind, i), "train": False})
            continue
        at[i] = len(messages)
        if kind == "say":
            messages.append({"role": "assistant", "content": text_of(kind, i).strip(), "train": False})
            continue
        tool, arguments, result = CALLS[kind]
        call_id = f"s{sid}-c{sid}"
        messages.append({
            "role": "assistant",
            "content": text_of(kind, i).strip(),
            "train": False,
            "tool_calls": [{"id": call_id, "name": tool, "arguments": arguments}],
        })
        messages.append({"role": "tool", "content": result, "train": False, "tool_call_id": call_id})
    return messages, at


def count(excluded, reason, n=1):
    excluded[reason] = excluded.get(reason, 0) + n


def sft_step(case):
    steps, labels, verdict, options = case["steps"], case["labels"], case["verdict"], case["options"]
    excluded = {}
    any_good = "good" in labels
    if verdict != "pass" and not any_good:
        count(excluded, "Failed" if verdict == "fail" else "Undecided")
        return [], excluded
    messages, at = messages_of(steps)
    last_agent = None
    for i, kind in enumerate(steps):
        if kind != "feedback":
            last_agent = i
    actions = []
    for i, kind in enumerate(steps):
        if kind == "feedback":
            continue
        if kind == "say":
            answered = "feedback" in steps[i + 1:]
            if i == last_agent and not answered:
                actions.append(i)
            continue
        actions.append(i)
    if not actions:
        count(excluded, "NoAction")
        return [], excluded
    seen_calls = []
    repeated = {}
    for i, kind in enumerate(steps):
        if kind in CALLS:
            repeated[i] = kind in seen_calls
            seen_calls.append(kind)
    trained = []
    truncated = False
    for i in actions:
        kind = steps[i]
        after_cut = truncated
        flagged = kind == "history" and options["history"] != "keep"
        if flagged and options["history"] == "truncate":
            truncated = True
        only = options["only"]
        wanted = only is None or kind not in CALLS or CALLS[kind][0] == only
        if labels[i] == "bad":
            count(excluded, "BadStep")
        elif after_cut:
            count(excluded, "AfterRepositoryHistory")
        elif flagged:
            count(excluded, "RepositoryHistory")
        elif repeated.get(i, False):
            count(excluded, "RepeatedAction")
        elif not wanted:
            count(excluded, "ActionNotWanted")
        elif verdict == "pass" or labels[i] == "good":
            trained.append(at[i])
    records = []
    if options["packed"]:
        if trained:
            record = [dict(m) for m in messages[: trained[-1] + 1]]
            for position in trained:
                record[position]["train"] = True
            records.append(record)
    else:
        for position in trained:
            record = [dict(m) for m in messages[: position + 1]]
            record[position]["train"] = True
            records.append(record)
    return records, excluded


def sft_final(case):
    excluded = {}
    if case["setting"] == "workspace":
        count(excluded, "WorkspaceEpisode")
        return [], excluded
    if case["verdict"] != "pass":
        count(excluded, "Failed" if case["verdict"] == "fail" else "Undecided")
        return [], excluded
    return [[
        {"role": "system", "content": "SYSTEM_PROMPT", "train": False},
        {"role": "user", "content": "What is two plus two?", "train": False},
        {"role": "assistant", "content": case["output"], "train": True},
    ]], excluded


def expected(case):
    records, excluded = (sft_step if case["view"] == "sft-step" else sft_final)(case)
    return {**case, "records": records, "excluded": excluded}


def step_cases(rng, n):
    out = []
    for _ in range(n):
        length = rng.randint(0, 5)
        steps = [rng.choice(KINDS) for _ in range(length)]
        labels = [None if k == "feedback" else rng.choice([None, None, "good", "bad"]) for k in steps]
        out.append({
            "view": "sft-step",
            "setting": "workspace",
            "steps": steps,
            "labels": labels,
            "verdict": rng.choice(["pass", "pass", "fail", None]),
            "options": {
                "packed": rng.random() < 0.5,
                "history": rng.choice(["drop", "keep", "truncate"]),
                "only": rng.choice([None, None, "read_file", "shell"]),
            },
        })
    return out


def final_cases():
    out = []
    for setting in ["closed", "workspace"]:
        for verdict in ["pass", "fail", None]:
            for output in ["4", "  Four.  ", "It is 4, not 5."]:
                out.append({"view": "sft-final", "setting": setting, "verdict": verdict, "output": output})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=600)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261124 if args.seed is None else args.seed)
    cases = [expected(c) for c in step_cases(rng, args.cases) + final_cases()]
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
