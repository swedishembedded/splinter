# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluations of whether an agent learns from
# its users, for its clients. If your team needs expertise in measuring that
# an agent kept what it was taught, you can procure our services by sending an
# email to info@swedishembedded.com.
"""Write the teaching scenarios: facts, corrections, rules, procedures, retractions.

The facts are about an invented company (Northwind Robotics), with values no
model can know, so a right answer is evidence of teaching and not of memory.
The first FACTS_DEV attributes and the first rules and procedures are the dev
split; the rest are the test split. Usage: teaching_set.py OUT_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

# attribute, statement template, first value, corrected value, three questions
FACTS = [
    ("the build server", "the build server is called {v}", "atlas-7", "borealis-3",
     ["What is the build server called?", "Which name does our build server have?", "I need to log in to the build server, what do I call it?"]),
    ("the deploy window", "the deploy window is {v}", "Thursday at 14:00", "Tuesday at 09:30",
     ["When is the deploy window?", "What day and time can we deploy?", "Remind me, when do deployments go out?"]),
    ("the on-call engineer", "the on-call engineer this week is {v}", "Marit Lindqvist", "Ottar Bergh",
     ["Who is the on-call engineer this week?", "Who do I page if production breaks this week?", "Which engineer is on call right now?"]),
    ("the staging database port", "the staging database port is {v}", "54871", "61203",
     ["Which port does the staging database use?", "What is the staging database port?", "I want to connect to staging, which port?"]),
    ("the office wifi network", "the office wifi network is called {v}", "NW-Guest-Fjord", "NW-Guest-Delta",
     ["What is the office wifi network called?", "Which wifi should visitors join?", "What name does the guest wifi have?"]),
    ("the release codename", "the release codename is {v}", "Kestrel", "Marlin",
     ["What is the release codename?", "Which codename does this release have?", "What are we calling the release?"]),
    ("the team chat channel", "the team chat channel is {v}", "#ops-heron", "#ops-lynx",
     ["Which chat channel does the team use?", "Where do I post updates for the team?", "What is the team channel?"]),
    ("the standup time", "the standup is at {v}", "09:45", "10:20",
     ["When is the standup?", "What time does standup start?", "At what time do we meet for standup?"]),
    ("the ticket prefix", "the ticket prefix is {v}", "NWR", "HRN",
     ["What is the ticket prefix?", "Which prefix do our tickets use?", "How do ticket numbers start?"]),
    ("the backup location", "the backups are stored in {v}", "the vault-oslo bucket", "the vault-turku bucket",
     ["Where are the backups stored?", "Which bucket holds the backups?", "I need a backup, where do I look?"]),
    ("the vpn host", "the vpn host is {v}", "gate.nw-robotics.example", "portal.nw-robotics.example",
     ["What is the vpn host?", "Which host do I connect the vpn to?", "What address does the vpn use?"]),
    ("the code freeze date", "the code freeze is on {v}", "the 14th", "the 21st",
     ["When is the code freeze?", "On which date does the code freeze start?", "What day is the freeze?"]),
    ("the parts supplier", "the parts supplier is {v}", "Halvorsen Components", "Brandt Mechatronik",
     ["Who is the parts supplier?", "Which company supplies our parts?", "Where do we order parts from?"]),
    ("the test rig", "the test rig is in {v}", "room B12", "room C07",
     ["Where is the test rig?", "Which room has the test rig?", "I need the test rig, which room?"]),
    ("the lab key", "the lab key is kept by {v}", "Sunniva", "Henrik",
     ["Who keeps the lab key?", "Who do I ask for the lab key?", "Which person holds the lab key?"]),
    ("the invoice approver", "invoices are approved by {v}", "Dagny Foss", "Leif Strand",
     ["Who approves invoices?", "Which person signs off on invoices?", "Who do I send an invoice to for approval?"]),
]
FACTS_DEV = 4

# How a right answer may be worded when the taught value can be spoken in more than one way.
SAID = {
    "Thursday at 14:00": ["Thursday"], "Tuesday at 09:30": ["Tuesday"],
    "09:45": ["09:45", "9:45", "nine forty-five", "quarter to ten"], "10:20": ["10:20", "ten twenty", "twenty past ten"],
    "the 14th": ["14th", "fourteenth"], "the 21st": ["21st", "twenty-first"],
    "the vault-oslo bucket": ["vault-oslo"], "the vault-turku bucket": ["vault-turku"],
    "room B12": ["B12"], "room C07": ["C07"],
}


def said(value):
    return SAID.get(value, [value])

UNRELATED = [
    ("What is two plus three?", ["5", "five"]),
    ("What is the capital of France?", ["Paris"]),
    ("How many days are in a week?", ["7", "seven"]),
]

RULES = [
    ("From now on, always end your answers with the word over.",
     {"ends_with": "over"}),
    ("Always call me captain.",
     {"any_of": ["captain"]}),
    ("From now on keep every answer to eight words or fewer.",
     {"max_words": 8}),
    ("From now on always say the word copy somewhere in your answer.",
     {"any_of": ["copy"]}),
]
RULE_QUESTIONS = ["What is the capital of France?", "Why is the sky blue?", "How do I boil an egg?", "What is a good name for a cat?"]

PROCEDURES = [
    ("deploy the site", "Here is how to deploy the site: first run the quartz checks, then seal the bundle, then push it to the harbor.",
     [("What is the first step to deploy the site?", ["quartz"]), ("How do I deploy the site?", ["harbor"]), ("What comes after sealing the bundle when deploying the site?", ["harbor"])]),
    ("rotate the keys", "Here is how to rotate the keys: first freeze the lantern, then mint the new pair, then retire the old pair.",
     [("What is the first step to rotate the keys?", ["lantern"]), ("How do I rotate the keys?", ["mint"]), ("What is the last step to rotate the keys?", ["retire"])]),
    ("close the month", "Here is how to close the month: first lock the ledger, then reconcile the tide table, then send the summary to Dagny.",
     [("What is the first step to close the month?", ["lock the ledger", "ledger"]), ("How do I close the month?", ["tide"]), ("Who gets the summary when closing the month?", ["Dagny"])]),
    ("reset the rig", "Here is how to reset the rig: first drain the coolant, then recalibrate the arm, then log it in the green book.",
     [("What is the first step to reset the rig?", ["coolant"]), ("How do I reset the rig?", ["green book"]), ("Where do I log the rig reset?", ["green book"])]),
]
PROCEDURES_DEV = 2


def fact_scenarios():
    for i, (subject, template, first, second, questions) in enumerate(FACTS):
        split = "dev" if i < FACTS_DEV else "test"
        teach = "Remember that " + template.format(v=first) + "."
        yield split, {"id": f"fact-{i:02d}", "kind": "fact", "teach": [teach],
                      "tests": [{"ask": q, "check": {"any_of": said(first)}} for q in questions]}
        correction = "Actually, " + template.format(v=second) + "."
        yield split, {"id": f"correction-{i:02d}", "kind": "correction", "teach": [teach, correction],
                      "tests": [{"ask": q, "check": {"any_of": said(second), "none_of": said(first)}} for q in questions]}
        yield split, {"id": f"retract-{i:02d}", "kind": "retract", "teach": [teach], "forget_last": True,
                      "tests": [{"ask": questions[0], "check": {"none_of": said(first)}}]}
        if i < 8:
            yield split, {"id": f"unrelated-{i:02d}", "kind": "unrelated", "teach": [teach],
                          "tests": [{"ask": q, "check": {"any_of": a}} for q, a in UNRELATED]}


# Questions that need a taught fact to be used, not looked up: attribute index, ask, first-value check follows said().
INDIRECT = [
    (0, "Write one sentence telling the team that the build is broken, naming the machine it ran on."),
    (2, "Draft a one-line page message to whoever is on call about the outage."),
    (10, "Give me the command line to connect to the vpn, with the host in it."),
    (12, "Write a one-line purchase request for ten servo motors, addressed to the supplier."),
]


def hard_scenarios():
    """Test-only scenarios that are harder than recalling one fact: many facts at once, facts that must be used, two rules at once."""
    teach_all = ["Remember that " + t.format(v=first) + "." for _, t, first, _, _ in FACTS]
    yield "test", {"id": "scale-00", "kind": "scale", "teach": teach_all,
                   "tests": [{"ask": qs[1], "check": {"any_of": said(first)}} for _, _, first, _, qs in FACTS]}
    for i, ask in INDIRECT:
        _, template, first, _, _ = FACTS[i]
        yield "test", {"id": f"indirect-{i:02d}", "kind": "indirect", "teach": ["Remember that " + template.format(v=first) + "."],
                       "tests": [{"ask": ask, "check": {"any_of": said(first)}}]}
    yield "test", {"id": "rules-two-00", "kind": "rules2",
                   "teach": [RULES[0][0], RULES[1][0]],
                   "tests": [{"ask": q, "check": {"ends_with": "over", "any_of": ["captain"]}} for q in RULE_QUESTIONS]}


def rule_scenarios():
    for i, (said, check) in enumerate(RULES):
        yield ("dev" if i % 2 == 0 else "test"), {"id": f"rule-{i:02d}", "kind": "rule", "teach": [said],
                                                    "tests": [{"ask": q, "check": check} for q in RULE_QUESTIONS]}


def procedure_scenarios():
    for i, (name, said, tests) in enumerate(PROCEDURES):
        yield ("dev" if i < PROCEDURES_DEV else "test"), {"id": f"procedure-{i:02d}", "kind": "procedure", "teach": [said],
                                                            "tests": [{"ask": q, "check": {"any_of": a}} for q, a in tests]}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    splits = {"dev": [], "test": []}
    for gen in (fact_scenarios, rule_scenarios, procedure_scenarios, hard_scenarios):
        for split, scenario in gen():
            splits[split].append(scenario)
    sums = []
    for split, scenarios in splits.items():
        text = "".join(json.dumps(s, sort_keys=True) + "\n" for s in scenarios)
        (out / f"scenarios-{split}.jsonl").write_text(text, encoding="utf-8")
        sums.append(f"{hashlib.sha256(text.encode()).hexdigest()}  scenarios-{split}.jsonl")
        print(split, len(scenarios), "scenarios")
    (out / "scenarios.sha256").write_text("\n".join(sums) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
