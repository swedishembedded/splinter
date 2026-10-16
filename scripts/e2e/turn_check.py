#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements end-to-end acceptance tests for learning
# agent stacks, for its clients. If your team needs expertise in proving that
# a model, an agent runtime and a trainer work together, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Assertions of the cross-repo acceptance test (scripts/e2e/turn.sh).

Each subcommand reads JSON the pipeline printed and either prints a line
saying what was found (exit 0) or names the missing artifact (exit 1). It
asserts that stages ran and artifacts exist, never how good they are.

  practice REPORT.json       every stage of a practice run left its report
  experiences LIST.json      the stored experiences and their workspace verdicts
  tool-calls EXPERIENCE.json...  at least one trajectory recorded a tool call
  gate REPORT.json           a candidate was trained and the gate decided
  candidate REPORT.json      print "<adapter path> <adapter digest>" of the candidate
  served MODELS.json DIGEST  the served model id names the adapter digest
"""

import json
import sys

STAGES = ["snapshot", "profile", "tasks", "reserve", "solve", "verify"]
MIN_EXPERIENCES = 6
MIN_DIGEST_PREFIX = 12


def fail(message):
    print(f"FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def practice(report):
    for stage in STAGES:
        if not report.get(stage):
            fail(f"missing artifact: the practice report has no {stage} stage")
    print("practice stages present: " + ", ".join(STAGES))


def experiences(listing):
    found = listing.get("experiences", [])
    if len(found) < MIN_EXPERIENCES:
        fail(f"missing artifact: {len(found)} experiences stored, need {MIN_EXPERIENCES}")
    undecided = [e for e in found if not e.get("decision")]
    if undecided:
        fail(f"missing artifact: {len(undecided)} experiences have no workspace verdict")
    print(f"{len(found)} experiences with verdicts")


def count_tool_calls(node):
    """Tool calls anywhere in a trajectory: ATIF steps list them as `tool_calls`."""
    if isinstance(node, dict):
        own = node.get("tool_calls")
        return (len(own) if isinstance(own, list) else 0) + sum(
            count_tool_calls(v) for v in node.values()
        )
    if isinstance(node, list):
        return sum(count_tool_calls(v) for v in node)
    return 0


def tool_calls(paths):
    total = sum(count_tool_calls(load(p)) for p in paths)
    if total < 1:
        fail("missing artifact: no trajectory recorded a tool call")
    print(f"{total} tool calls recorded in {len(paths)} trajectories")


def candidate_of(report):
    candidate = report.get("candidate")
    if not candidate or not candidate.get("adapter") or not candidate.get("adapter_digest"):
        fail("missing artifact: no trained candidate adapter")
    return candidate


def gate(report):
    candidate_of(report)
    release = report.get("release")
    decision = (release or {}).get("gate")
    if not decision:
        fail("missing artifact: the release gate produced no decision")
    # Whatever the decision is, it is printed; the gate is never bypassed.
    print("release gate decision: " + json.dumps(decision.get("passed", decision)))


def candidate(report):
    c = candidate_of(report)
    print(f"{c['adapter']} {c['adapter_digest']}")


def served(models, digest):
    ids = [m.get("id", "") for m in models.get("data", [])]
    hex_digest = digest.split(":")[-1]
    wanted = hex_digest[:MIN_DIGEST_PREFIX]
    if not any(wanted in i for i in ids):
        fail(f"missing artifact: no served model id names the adapter digest {wanted} (ids: {ids})")
    print("served model ids: " + ", ".join(ids))


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    command, args = argv[1], argv[2:]
    if command == "tool-calls":
        tool_calls(args)
    elif command == "served" and len(args) == 2:
        served(load(args[0]), args[1])
    elif command in {"practice", "experiences", "gate", "candidate"} and len(args) == 1:
        {"practice": practice, "experiences": experiences, "gate": gate,
         "candidate": candidate}[command](load(args[0]))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
