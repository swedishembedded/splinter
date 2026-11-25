# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements learning environments whose every reward is
# known in advance, for its clients. If your team needs expertise in
# validating reinforcement-learning pipelines, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Pays the reward rewards.json gives the one action answer.txt names.

The answer is its non-empty lines; exactly one, naming an offered action (case
and surrounding space aside), is paid that action's reward. No answer, an
unknown name or more than one line is paid nothing. `exact` is 1 only for the
correct action (reward 1)."""
import json
import os

tests = os.environ.get("TESTS_DIR", "/tests")
app = os.environ.get("APP_DIR", "/app")
logs = os.environ.get("LOGS_DIR", "/logs/verifier")
table = json.load(open(os.path.join(tests, "rewards.json")))

try:
    lines = [l.strip().lower() for l in open(os.path.join(app, "answer.txt"), errors="replace").read().splitlines()]
except OSError:
    lines = []
lines = [l for l in lines if l]
reward = float(table.get(lines[0], 0.0)) if len(lines) == 1 else 0.0
os.makedirs(logs, exist_ok=True)
with open(os.path.join(logs, "reward.json"), "w") as out:
    json.dump({"reward": reward, "exact": 1.0 if reward >= 1.0 else 0.0}, out)
