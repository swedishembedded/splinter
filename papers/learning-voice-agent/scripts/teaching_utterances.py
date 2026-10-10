# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements agents that learn facts, rules and workflows
# from what their users tell them, for its clients. If your team needs
# expertise in making an agent correctable by the people who work with it, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Write the utterances the readers of teaching are scored on.

Each line is what a user might say to an assistant, with what it is
(none, fact, rule, procedure) and, for a teaching, a key word its content must
contain. They are written by the implementer, in many phrasings and in the
lower-case, unpunctuated form a recogniser produces for about a third, before
any reader that is scored on them was run. The split is by position: every
third utterance is dev. Usage: teaching_utterances.py OUT.jsonl
"""
import hashlib
import json
import sys

# (kind, key, said, previous reply or "")
U = [
    # facts and corrections of facts
    ("fact", "monday", "Actually, the deadline is Monday.", "The deadline is Friday."),
    ("fact", "monday", "no no it's Monday not Friday", "The deadline is Friday."),
    ("fact", "atlas", "That's wrong, the build server is called atlas.", "The build server is Jenkins."),
    ("fact", "atlas", "you got that wrong the build server is atlas", "It is Jenkins."),
    ("fact", "marit", "Marit is on call this week, not Ottar.", "Ottar is on call."),
    ("fact", "thursday", "the deploy window is Thursday at 14:00 by the way", ""),
    ("fact", "sam", "Just so you know, my manager is Sam.", ""),
    ("fact", "paris", "No, our European office is in Paris.", "Your European office is in Berlin."),
    ("fact", "green", "Remember that the green binder holds the safety checklists.", ""),
    ("fact", "5432", "the production database listens on port 5432", ""),
    ("fact", "tuesday", "it's actually Tuesday, the meeting moved", "The meeting is on Monday."),
    ("fact", "nora", "for the record the project lead is Nora", ""),
    ("fact", "blue", "Not quite. The logo is blue, not green.", "The logo is green."),
    ("fact", "kestrel", "The release is called Kestrel.", ""),
    ("fact", "nine|9", "our standup starts at nine, not ten", "Standup is at ten."),
    ("fact", "oslo", "Please note the backups live in the Oslo bucket.", ""),
    ("fact", "lars", "That is incorrect. Lars owns the billing service.", "Mia owns billing."),
    ("fact", "friday", "the invoices go out on Fridays", ""),
    ("fact", "b12", "fyi the test rig is in room B12", ""),
    ("fact", "halvorsen", "wrong, the supplier is Halvorsen Components", "Your supplier is Brandt."),
    ("fact", "three|3", "Keep in mind there are three people on the support team.", ""),
    ("fact", "linux", "Our servers run Linux, not Windows.", "Your servers run Windows."),
    ("fact", "spring", "Correction: the conference is in spring.", "The conference is in autumn."),
    ("fact", "anna", "that's not right, Anna is the treasurer", "Ben is the treasurer."),
    ("fact", "helsinki", "I live in Helsinki now.", ""),
    # rules
    ("rule", "confirm", "From now on, always confirm before deleting a file.", ""),
    ("rule", "push", "Never push to main without asking me first.", ""),
    ("rule", "short", "keep your answers short from now on", ""),
    ("rule", "metric", "In the future use metric units.", ""),
    ("rule", "vendor", "don't ever touch the vendor directory", ""),
    ("rule", "summary", "Next time, start with a one line summary.", ""),
    ("rule", "formal", "going forward please be more formal with me", ""),
    ("rule", "ask", "whenever you are unsure, ask me instead of guessing", ""),
    ("rule", "tests", "Always run the tests before you tell me it is done.", ""),
    ("rule", "emoji", "Never use emoji in anything you write for me.", ""),
    ("rule", "backup", "from here on make a backup before you change a config file", ""),
    ("rule", "norwegian", "Please answer in Norwegian from now on.", ""),
    ("rule", "lock", "you shouldn't edit the lock file, leave it alone from now on", ""),
    ("rule", "ask", "that was the wrong file, next time ask me which one I mean", "I edited config.old."),
    ("rule", "dry", "do a dry run first every time you deploy", ""),
    ("rule", "bullet", "i want you to always use bullet points when you list things", ""),
    ("rule", "weekly", "when I say the usual, I mean the weekly report", ""),
    ("rule", "log", "Always write what you changed into the log.", ""),
    ("rule", "go", "don't act until I say go", ""),
    ("rule", "captain", "Always call me captain.", ""),
    ("rule", "friday", "Never deploy on a Friday.", ""),
    ("rule", "api", "never print the api key in a reply", ""),
    ("rule", "sources", "from now on cite your sources", ""),
    ("rule", "confirm", "you deleted the wrong folder, from now on show me the path and wait for a yes", "I deleted /tmp/build."),
    ("rule", "explanation", "stop adding explanations unless I ask for them", ""),
    ("rule", "copy", "Whenever you finish a task say the word copy.", ""),
    # procedures
    ("procedure", "tests", "Here is how to deploy the site: first run the tests, then build the image, then push it to the registry.", ""),
    ("procedure", "invoice", "to close the month you lock the ledger, then reconcile the invoices, then send the summary to Dagny", ""),
    ("procedure", "coolant", "This is how you reset the rig. Drain the coolant, recalibrate the arm, and log it in the green book.", ""),
    ("procedure", "ticket", "The process for a new hire is: create the account, open a ticket for the laptop, and book the intro meeting.", ""),
    ("procedure", "backup", "when a server fails, first fail over to the replica, then restore from backup, and finally write the incident report", ""),
    ("procedure", "review", "Here's how we ship a change. Get a review, wait for green checks, merge, then tag the release.", ""),
    ("procedure", "keys", "to rotate the keys: freeze deploys, mint the new pair, then retire the old pair", ""),
    ("procedure", "expense", "this is how expenses work, you photograph the receipt then you file it in the app then your manager approves it", ""),
    ("procedure", "onboard", "the way we onboard a customer is to send the welcome mail, schedule the kickoff, and create their workspace", ""),
    ("procedure", "release", "Here is how to cut a release: bump the version, update the changelog, tag it, and publish.", ""),
    ("procedure", "sample", "to prepare a sample you weigh it, dissolve it in the buffer, and then label the tube", ""),
    ("procedure", "audit", "here is the procedure for an audit. Collect the logs, redact the names, and send them to compliance.", ""),
    # not teaching: questions, requests, chat, near-misses
    ("none", "", "What is the deadline?", ""),
    ("none", "", "Can you tell me when the deploy window is?", ""),
    ("none", "", "Do you remember that the deadline is Friday?", ""),
    ("none", "", "Delete the old log file please.", ""),
    ("none", "", "Always a pleasure talking to you.", ""),
    ("none", "", "Never mind.", ""),
    ("none", "", "No, I meant the other file.", ""),
    ("none", "", "Actually, could you repeat that?", ""),
    ("none", "", "Here is the report I mentioned.", ""),
    ("none", "", "Thanks, that is helpful.", ""),
    ("none", "", "Good morning!", ""),
    ("none", "", "I think the deadline might be Friday, is that right?", ""),
    ("none", "", "How do I deploy the site?", ""),
    ("none", "", "Please summarise this document in one paragraph.", ""),
    ("none", "", "what time is it", ""),
    ("none", "", "I'm tired today.", ""),
    ("none", "", "Wow, that was fast.", ""),
    ("none", "", "that is wrong", ""),
    ("none", "", "yes", ""),
    ("none", "", "go ahead", ""),
    ("none", "", "Can you open the build log and tell me what failed?", ""),
    ("none", "", "I will be late tomorrow.", ""),
    ("none", "", "run the tests now", ""),
    ("none", "", "Remind me to call Anna at five.", ""),
    ("none", "", "What did you say the port was?", ""),
    ("none", "", "Is the server running?", ""),
    ("none", "", "From now on? What do you mean?", ""),
    ("none", "", "next time I will bring coffee", ""),
    ("none", "", "I always forget my password.", ""),
    ("none", "", "He never answers my emails.", ""),
    ("none", "", "First we need to talk about the budget.", ""),
    ("none", "", "ok then, what's next", ""),
    ("none", "", "Write a short poem about the sea.", ""),
    ("none", "", "How many people are on the support team?", ""),
    ("none", "", "Could you check whether the build server is up?", ""),
    ("none", "", "wait, say that again", ""),
    ("none", "", "I wonder if we should use Linux or Windows.", ""),
    ("none", "", "My manager is away this week.", ""),
    ("none", "", "I don't think that is right.", ""),
    ("none", "", "Tell me how the invoices get approved.", ""),
    ("none", "", "I need the report by Monday.", ""),
    ("none", "", "Please don't be late.", ""),
    ("none", "", "Is it true that the deadline moved?", ""),
    ("none", "", "that was great, thank you", ""),
    ("none", "", "let me think about it", ""),
    ("none", "", "Can you explain what a lock file is?", ""),
    ("none", "", "What is the capital of Norway?", ""),
    ("none", "", "I'll do it myself.", ""),
    ("none", "", "Could you not do that, please?", ""),
    ("none", "", "not now", ""),
    ("none", "", "He said the release is called Kestrel.", ""),
    ("none", "", "See you tomorrow.", ""),
    ("none", "", "Open the vendor directory and list the files.", ""),
    ("none", "", "Where are the backups stored?", ""),
    ("none", "", "that's interesting", ""),
    ("none", "", "How does the process for a new hire work?", ""),
    ("none", "", "I want a summary of the meeting.", ""),
    ("none", "", "Do it again, but faster.", ""),
]


def main():
    out = sys.argv[1]
    lines = []
    for i, (kind, key, said, previous) in enumerate(U):
        lines.append(json.dumps({"id": f"u{i:03d}", "split": "dev" if i % 3 == 0 else "test", "kind": kind, "key": key,
                                 "said": said, "previous": previous or None}, sort_keys=True))
    data = "\n".join(lines) + "\n"
    open(out, "w", encoding="utf-8").write(data)
    open(out + ".sha256", "w").write(f"{hashlib.sha256(data.encode()).hexdigest()}  {out.split('/')[-1]}\n")
    kinds = {}
    for k, *_ in U:
        kinds[k] = kinds.get(k, 0) + 1
    print(len(U), kinds)


if __name__ == "__main__":
    main()
