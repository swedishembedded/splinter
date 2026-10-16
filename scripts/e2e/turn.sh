#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements end-to-end acceptance tests for learning
# agent stacks, for its clients. If your team needs expertise in proving that
# a model, an agent runtime and a trainer work together, you can procure our
# services by sending an email to info@swedishembedded.com.

# One full turn of the flywheel on Qwen3-0.6B, across all three repositories:
#
#   1. brain doctor admits the model;
#   2. splinter practice on the acceptance-tiny fixture (a git repository made
#      from tasks/acceptance-tiny): snapshot, profile, tasks, reserve, solve
#      through sven, verify;
#   3. the experience store holds at least 6 experiences with workspace
#      verdicts, and at least one trajectory records a tool call;
#   4. a candidate adapter was trained (practice's training stage);
#   5. the release gate ran and its decision is printed whatever it is (it is
#      never bypassed); the candidate is then served by brain serve
#      --adapter-manifest through a test-only manifest, sven solves one task
#      through it, and the served model id names the adapter digest.
#
# It asserts stages and artifacts, never quality. A missing artifact fails the
# run and is named. The model runs on brain's default backend; set
# BRAIN_DEVICE=cpu for the CPU backend (minutes to tens of minutes), or run
# under the GPU lease wrapper for the short GPU variant.
#
# Exit status: 0 passed; 1 failed (the missing artifact is named); 3 a
# prerequisite is missing, so nothing was tested. A 3 is never a pass.
#
# Usage: scripts/e2e/turn.sh <splinter> <brain> <sven> <models-dir> <fixture-dir>
set -uo pipefail

if [ "$#" -ne 5 ]; then
    echo "usage: $0 <splinter> <brain> <sven> <models-dir> <fixture-dir>" >&2
    exit 2
fi
splinter=$1
brain=$2
sven=$3
models=$4
fixture=$5
model=Qwen/Qwen3-0.6B
here=$(cd "$(dirname "$0")" && pwd)
check() { python3 "$here/turn_check.py" "$@"; }

skip() { echo "SKIPPED (nothing was tested): $*" >&2; exit 3; }
fail() { echo "FAILED: $*" >&2; exit 1; }

[ -x "$splinter" ] || skip "no splinter binary at $splinter"
[ -x "$brain" ] || skip "no brain binary at $brain"
[ -x "$sven" ] || skip "no sven binary at $sven"
[ -d "$fixture" ] || skip "no fixture at $fixture"

scratch=$(mktemp -d)
server=
cleanup() {
    [ -n "$server" ] && kill "$server" 2>/dev/null
    rm -rf "$scratch"
}
trap cleanup EXIT

echo "== 1. brain doctor admits $model"
"$brain" doctor --models-dir "$models" --model "$model" >"$scratch/doctor.out" 2>&1 \
    || skip "brain doctor does not admit $model in $models: $(tail -3 "$scratch/doctor.out")"

echo "== 2. splinter practice on the fixture"
repo=$scratch/repo
cp -r "$fixture" "$repo" || fail "missing artifact: the fixture could not be copied"
(cd "$repo" && git init -q && git add -A \
    && git -c user.name=turn -c user.email=turn@example.invalid commit -q -m fixture) \
    || fail "missing artifact: the fixture repository could not be made"
state=$scratch/state
export BRAIN_MODELS_DIR=$models
export SPLINTER_BRAIN_BIN=$brain
"$splinter" --json --state "$state" practice "$repo" --max-tasks 6 --steps 20 \
    >"$scratch/practice.json" 2>"$scratch/practice.err" \
    || { tail -20 "$scratch/practice.err" >&2; fail "splinter practice did not complete"; }
check practice "$scratch/practice.json" || exit 1

echo "== 3. the experience store"
set_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["solve"]["experience_set"])' "$scratch/practice.json") \
    || fail "missing artifact: the solve stage names no experience set"
"$splinter" --json --state "$state" experiences show "$set_id" >"$scratch/experiences.json" 2>/dev/null \
    || fail "missing artifact: experience set $set_id cannot be shown"
check experiences "$scratch/experiences.json" || exit 1
mkdir "$scratch/trajectories"
python3 -c 'import json,sys; [print(e["id"]) for e in json.load(open(sys.argv[1]))["experiences"]]' \
    "$scratch/experiences.json" | while read -r id; do
    "$splinter" --json --state "$state" experiences show "$id" >"$scratch/trajectories/$id.json" 2>/dev/null
done
check tool-calls "$scratch"/trajectories/*.json || exit 1

echo "== 4. the trained candidate and 5. the release gate"
check gate "$scratch/practice.json" || exit 1
read -r adapter digest < <(check candidate "$scratch/practice.json") || exit 1

echo "== 5. the candidate served through a test-only manifest"
manifest=$scratch/adapter-manifest.json
python3 -c 'import json,sys; json.dump({"adapter": sys.argv[1], "digest": sys.argv[2]}, open(sys.argv[3], "w"))' \
    "$adapter" "$digest" "$manifest" || fail "missing artifact: the adapter manifest could not be written"
port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$brain" serve --openai "$port" --models-dir "$models" --adapter-manifest "$manifest" \
    --api-keys-out "$scratch/keys.json" --ready-file "$scratch/ready" >"$scratch/brain.log" 2>&1 &
server=$!
for _ in $(seq 1 240); do
    [ -e "$scratch/ready" ] && break
    kill -0 "$server" 2>/dev/null || { tail -20 "$scratch/brain.log" >&2; fail "brain serve exited before it was ready"; }
    sleep 5
done
[ -e "$scratch/ready" ] || { tail -20 "$scratch/brain.log" >&2; fail "brain serve was not ready after 20 minutes"; }
key=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["openai"])' "$scratch/keys.json")
curl -fsS -H "Authorization: Bearer $key" "http://127.0.0.1:$port/v1/models" >"$scratch/models.json" \
    || fail "missing artifact: brain serve lists no models"
check served "$scratch/models.json" "$digest" || exit 1
served_id=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["data"]; print(next(m["id"] for m in d if sys.argv[2] in m["id"]))' \
    "$scratch/models.json" "${digest##*:}" 2>/dev/null) || served_id=$model

cat >"$scratch/sven.yaml" <<YAML
model:
  provider: brain
  name: $served_id
  base_url: http://127.0.0.1:$port/v1
  api_key: $key
YAML
home=$scratch/home
work=$home/repo
mkdir -p "$work"
secret=4217
printf 'The magic number is %s.\n' "$secret" >"$work/hello.txt"
(cd "$work" && HOME="$home" XDG_CONFIG_HOME="$home" SVEN_DISABLE_BRAIN_AUTODETECT=1 "$sven" --config "$scratch/sven.yaml" \
    -H "Use the read_file tool to read hello.txt, then tell me the magic number it contains.") \
    >"$scratch/task.out" 2>"$scratch/task.err" \
    || { tail -20 "$scratch/task.err" >&2; fail "missing artifact: sven did not complete a task through the served candidate"; }

echo "PASSED: practice, ${served_id} served with the candidate adapter and driven by sven"
