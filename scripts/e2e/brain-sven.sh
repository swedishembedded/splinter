#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements agent stacks whose pieces are tested
# together, not one at a time, for its clients. If your team needs expertise in
# end-to-end validation of local model serving and agent runtimes, you can
# procure our services by sending an email to info@swedishembedded.com.

# Brain serves a real model; sven checks it and completes a tool-using task
# with it. The task's answer lives only in a file the model has to read, so a
# pass is judged on the file's content, not on the agent's claim.
#
# The model runs on the CPU backend, so the run needs no free GPU and takes
# minutes to tens of minutes.
#
# Exit status: 0 passed; 1 failed (the log is printed); 3 a prerequisite is
# missing, so nothing was tested. A 3 is never a pass.
#
# Usage: scripts/e2e/brain-sven.sh <brain-binary> <sven-binary> <models-dir>
set -uo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <brain-binary> <sven-binary> <models-dir>" >&2
    exit 2
fi
brain=$1
sven=$2
models=$3
model=Qwen/Qwen3-0.6B
secret=4217

skip() { echo "SKIPPED (nothing was tested): $*" >&2; exit 3; }
fail() { echo "FAILED: $*" >&2; exit 1; }

[ -x "$brain" ] || skip "no brain binary at $brain"
[ -x "$sven" ] || skip "no sven binary at $sven"
"$brain" doctor --models-dir "$models" --model "$model" >/dev/null 2>&1 \
    || skip "brain doctor says $model cannot be served from $models ($brain doctor --models-dir $models --model $model)"

scratch=$(mktemp -d)
server=
cleanup() {
    [ -n "$server" ] && kill "$server" 2>/dev/null
    rm -rf "$scratch"
}
trap cleanup EXIT

port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')

"$brain" serve --device cpu --openai "$port" --models-dir "$models" \
    --api-keys-out "$scratch/keys.json" --ready-file "$scratch/ready" >"$scratch/brain.log" 2>&1 &
server=$!

for _ in $(seq 1 240); do
    [ -e "$scratch/ready" ] && break
    kill -0 "$server" 2>/dev/null || { tail -20 "$scratch/brain.log" >&2; fail "brain serve exited before it was ready"; }
    sleep 5
done
[ -e "$scratch/ready" ] || { tail -20 "$scratch/brain.log" >&2; fail "brain serve was not ready after 20 minutes"; }

key=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["openai"])' "$scratch/keys.json")
cat >"$scratch/sven.yaml" <<YAML
model:
  provider: brain
  name: $model
  base_url: http://127.0.0.1:$port/v1
  api_key: $key
YAML

home=$scratch/home
repo=$home/repo
mkdir -p "$repo"
printf 'The magic number is %s.\n' "$secret" >"$repo/hello.txt"
run_sven() {
    (cd "$repo" && HOME="$home" XDG_CONFIG_HOME="$home" SVEN_DISABLE_BRAIN_AUTODETECT=1 "$sven" --config "$scratch/sven.yaml" "$@")
}

echo "== sven doctor provider"
run_sven doctor provider || fail "sven cannot use the model brain serves"

echo "== a tool-using task"
run_sven -H "Use the read_file tool to read hello.txt, then tell me the magic number it contains." \
    >"$scratch/task.out" 2>"$scratch/task.err" || { tail -20 "$scratch/task.err" >&2; fail "sven did not complete the task"; }
grep -q '"name": "read_file"' "$scratch/task.out" || { cat "$scratch/task.out" >&2; fail "the agent never called read_file"; }
grep -q "$secret" "$scratch/task.out" || { cat "$scratch/task.out" >&2; fail "the answer does not contain the number only the file holds"; }

echo "PASSED: brain served $model, sven read the file with a tool and reported $secret"
