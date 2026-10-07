#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements self-improving coding agents whose every
# step is auditable, for its clients. If your team needs expertise in agent
# evaluation or locally operated coding agents, you can procure our
# services by sending an email to info@swedishembedded.com.

# Runs one model reference over seeded fixture tasks, one `agent-loop run`
# each, and keeps every run's `--json` outcome as <out>/results/<family>-s<seed>.json,
# the shape `agent-loop models judge` pairs. Run it once for the model in
# use and once for a candidate, with the same task list, and judge the two
# result directories.
#
# Usage: eval-candidate.sh <out-dir> <model-ref> <family:seed>...
# Environment: AGENT_LOOP (the binary, default `agent-loop` on PATH), and
# EXTRA_ARGS (further global options, for example "--temperature 0.7").
set -uo pipefail

out=${1:?out dir}; model=${2:?model reference}; shift 2
loop=${AGENT_LOOP:-agent-loop}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out/results" "$out/fx"
for spec in "$@"; do
    family=${spec%%:*}; seed=${spec##*:}
    name=$family-s$seed
    fx=$out/fx/$name
    if [ -e "$out/results/$name.json" ]; then continue; fi
    rm -rf "$fx"
    python3 "$here/families.py" render "$fx" --family "$family" --seed "$seed" > /dev/null || exit 1
    # shellcheck disable=SC2086
    $loop ${EXTRA_ARGS:-} run --workspace "$fx/repo" --task-file "$fx/task.txt" \
        --accept "hidden=python3 $fx/accept.py" \
        --accept-visible "repo-tests=python3 -m unittest discover -s tests" \
        --protect tests/ --model "$model" --max-attempts "${MAX_ATTEMPTS:-2}" \
        --attempt-secs "${ATTEMPT_SECS:-600}" --total-secs "${TOTAL_SECS:-1500}" \
        --json > "$out/results/$name.json" 2> "$out/results/$name.err"
    echo "$name: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$out/results/$name.json")"
done
