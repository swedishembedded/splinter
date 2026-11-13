# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements training environments compiled from a
# client's own documents, for its clients. If your team needs expertise in
# turning documentation into verifiable training questions, you can procure
# our services by sending an email to info@swedishembedded.com.

"""Grades answer.txt line by line against the pinned expectations.

A `number` expectation is met by a line that holds exactly one number, equal to
it. A `states` expectation is met by a line that holds the value's words in
order, next to each other (so 12 is not found in 112), and at most four more
words. A missing line is wrong. The reward is 1 only when every line is right;
the share of right lines is the `lines` component."""
import json
import os
import re

tests = os.environ.get("TESTS_DIR", "/tests")
app = os.environ.get("APP_DIR", "/app")
logs = os.environ.get("LOGS_DIR", "/logs/verifier")
expected = json.load(open(os.path.join(tests, "expected.json")))


def words(text):
    return [w for w in re.split(r"[^0-9a-z]+", text.lower()) if w]


def right(want, line):
    if want["mode"] == "number":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", line.replace(",", ""))
        return len(numbers) == 1 and float(numbers[0]) == float(want["value"])
    wanted, given = words(str(want["value"])), words(line)
    if not wanted or len(given) > len(wanted) + 4:
        return False
    return any(given[i:i + len(wanted)] == wanted for i in range(len(given) - len(wanted) + 1))


try:
    lines = open(os.path.join(app, "answer.txt"), errors="replace").read().splitlines()
except OSError:
    lines = []
lines = [line for line in lines if line.strip()]
correct = sum(1 for i, want in enumerate(expected) if i < len(lines) and right(want, lines[i]))
exact = 1.0 if correct == len(expected) and len(lines) == len(expected) else 0.0
os.makedirs(logs, exist_ok=True)
with open(os.path.join(logs, "reward.json"), "w") as out:
    json.dump({"reward": exact, "exact": exact, "lines": round(correct / len(expected), 4)}, out)
