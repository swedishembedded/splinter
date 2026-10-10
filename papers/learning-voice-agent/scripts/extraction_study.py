#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements voice agents that learn how to speak from
# what their users tell them, for its clients. If your team needs expertise
# in measuring how reliably spoken feedback is understood, you can procure our
# services by sending an email to info@swedishembedded.com.

"""How well the readers of a user's words read them, per reader and per context.

    extraction_study.py VOICE_LOOP data/utterances/utterances.csv out.json

Each reader sees only the turns it would be listening to. The directive reader
and the retraction reader listen at every turn. The feedback reader listens
after the agent said something, so it is not asked about answers to a pending
offer or comparison; the confirmation reader listens while an offer is pending
and the comparison reader while two takes are being compared. The gold label is
what the user meant, written down before the readers were run. Standard
library only.
"""

import csv
import json
import subprocess
import sys
import tempfile

# reader -> (the gold classes it should fire on, the other classes it is shown)
CONTEXTS = {
    "respell": ("directive", {"respell"}, {"imitate", "reject", "approve", "retract", "confirm_yes", "confirm_no", "compare_new", "compare_old", "none"}),
    "imitate": ("directive", {"imitate"}, {"respell", "reject", "approve", "retract", "confirm_yes", "confirm_no", "compare_new", "compare_old", "none"}),
    "reject": ("feedback", {"reject"}, {"approve", "respell", "imitate", "retract", "none"}),
    "approve": ("feedback", {"approve"}, {"reject", "respell", "imitate", "retract", "none"}),
    "retract": ("retract", {"retract"}, {"respell", "imitate", "reject", "approve", "none"}),
    "confirm_yes": ("confirmation", {"confirm_yes"}, {"confirm_no", "respell", "imitate", "retract", "none"}),
    "confirm_no": ("confirmation", {"confirm_no"}, {"confirm_yes", "respell", "imitate", "retract", "none"}),
    "compare_new": ("comparison", {"compare_new"}, {"compare_old", "reject", "approve", "none"}),
    "compare_old": ("comparison", {"compare_old"}, {"compare_new", "reject", "approve", "none"}),
}
FIRE = {
    "respell": ("directive", "respell"), "imitate": ("directive", "imitate"), "reject": ("feedback", "reject"),
    "approve": ("feedback", "approve"), "retract": ("retract", True), "confirm_yes": ("confirmation", "yes"),
    "confirm_no": ("confirmation", "no"), "compare_new": ("comparison", "new"), "compare_old": ("comparison", "old"),
}


def main():
    binary, table, out = sys.argv[1:4]
    rows = list(csv.DictReader(open(table, newline="")))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(r["text"] for r in rows) + "\n")
        path = f.name
    # --backend is not needed: classifying loads no model.
    result = subprocess.run([binary, "teach", "--classify", path], capture_output=True, text=True, check=True)
    verdicts = [json.loads(l) for l in result.stdout.splitlines() if l.startswith("{")]
    assert len(verdicts) == len(rows), (len(verdicts), len(rows))
    report = {"table": table, "utterances": len(rows), "readers": {}}
    for split in ("test", "dev"):
        for reader, (key, positives, shown) in CONTEXTS.items():
            tp = fp = fn = 0
            misses, wrong = [], []
            for r, v in zip(rows, verdicts):
                if r["split"] != split or r["gold"] not in positives | shown:
                    continue
                field, want = FIRE[reader]
                fired = v[key] == want if key != "retract" else v[key] is True
                if r["gold"] in positives:
                    if fired:
                        tp += 1
                    else:
                        fn += 1
                        misses.append(r["text"])
                elif fired:
                    fp += 1
                    wrong.append(r["text"])
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if tp + fp + fn else None)
            report["readers"].setdefault(reader, {})[split] = {
                "positives": tp + fn, "true_positives": tp, "false_positives": fp, "false_negatives": fn,
                "precision": precision, "recall": recall, "f1": f1, "missed": misses, "wrongly_fired_on": wrong,
            }
    json.dump(report, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
