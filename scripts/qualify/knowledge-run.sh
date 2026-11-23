#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements qualification of learning agent systems
# for its clients. If your team needs expertise in proving that a model
# learned an unfamiliar domain, you can procure our services by sending an
# email to info@swedishembedded.com.

# A documented domain learned through the factory: the knowledge-learning
# template written into OUT/repo with the manual as its documents, run into
# OUT/run, and a summary printed as JSON for the qualification's assertions:
#
#   {"accepted": the run completed and trained a candidate,
#    "admitted_tasks": the tasks code admitted from the manual,
#    "passed_answers": the answers the tasks' verifiers passed}
#
# Usage: scripts/qualify/knowledge-run.sh SPLINTER MANUAL OUT [splinter run flags...]
# MANUAL is a file or a directory of documents. Exit status: 0 with the
# summary printed (accepted may be false); 2 for a usage error.
set -uo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 SPLINTER MANUAL OUT [splinter run flags...]" >&2
    exit 2
fi
# Absolute paths: the run is started from inside the repository written.
splinter=$(command -v "$1") || { echo "$0: no splinter binary at $1" >&2; exit 2; }
splinter=$(realpath "$splinter")
manual=$(realpath -m "$2")
out=$(realpath -m "$3")
shift 3
[ -e "$manual" ] || { echo "$0: no manual at $manual (a file or a directory)" >&2; exit 2; }

repo=$out/repo
run=$out/run
mkdir -p "$out"
"$splinter" init --template knowledge-learning "$repo" >&2 || exit 2
rm -rf "$repo/documents"
mkdir -p "$repo/documents"
cp -r "$manual" "$repo/documents/" || exit 2
(cd "$repo" && "$splinter" --json run recipes/knowledge-learning.yaml --out "$run" "$@") >&2
status=$?

python3 -I - "$run" "$status" <<'PY'
import json
import sys
from pathlib import Path

run, status = Path(sys.argv[1]), int(sys.argv[2])


def items(stage):
    index = run / stage / "items.json"
    if not index.is_file():
        return []
    return json.loads(index.read_text(encoding="utf-8")).get("items", [])


passed = 0
for item in items("answered"):
    for verdicts in (run / "answered" / item["path"]).glob("verdicts-*.json"):
        for verdict in json.loads(verdicts.read_text(encoding="utf-8")):
            if verdict.get("body", {}).get("outcome") == "pass":
                passed += 1
                break
print(json.dumps({
    "accepted": status == 0 and (run / "lora").is_dir(),
    "admitted_tasks": len(items("generated")),
    "passed_answers": passed,
}))
PY
