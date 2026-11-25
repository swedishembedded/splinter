#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements online reinforcement learning that trains a served policy on its own rollouts
# while it runs, for its clients. If your team needs expertise in asynchronous policy optimisation then you can
# procure our services by sending an email to info@swedishembedded.com.
#
# Runs online training end to end on the real stack, on the CPU, with no pretrained model: a toy qwen3 checkpoint
# (`sample-models-qwen3 toy`) served by `brain serve --openai --training`, and `train.online` from it over the
# toy-answers task family (a reply that starts with a letter, graded by the task's own test.sh). The run is checked
# as it goes and after: versions published and served, the records the learner took valid (brain checks every
# offered record and the stage fails on a refusal), receipts applied and
# committed, the controller killed and resumed with no group trained twice, brain killed and restarted under the same
# run with what it had not committed sent again, the report's counters consistent. Wall times printed are evidence
# of the mechanism only.
#
# usage: scripts/online/cpu-stack.sh <splinter> <brain> <qwen3 sample> <work dir> <port> <estimator>
#   estimator: clipped, or score-centred (then every request asks the sampler's top 2)

set -euo pipefail

splinter=$(realpath "${1:?usage: cpu-stack.sh <splinter> <brain> <qwen3> <work> <port> <estimator>}")
brain=$(realpath "${2:?brain binary}")
qwen3=$(realpath "${3:?qwen3 sample binary}")
work=${4:?work directory}
port=${5:?port}
estimator=${6:?estimator}
case "$estimator" in
    clipped) estimator_with="estimator: clipped" ;;
    score-centred) estimator_with=$'estimator: score-centred\n      top_logprobs: 2' ;;
    *) echo "cpu-stack.sh: estimator is clipped or score-centred, not $estimator" >&2; exit 2 ;;
esac
case "$port" in
    8788 | 8789 | 8790 | 8791) echo "cpu-stack.sh: $port belongs to the shared servers" >&2; exit 2 ;;
esac
family=$(realpath "$(dirname "$0")/../../tasks/toy-answers")

mkdir -p -m 700 "$work"
work=$(realpath "$work")
rm -rf "${work:?}"/{models,artifacts,adapters,brain-runtime,repo,run,state,ready,brain.log,splinter.log}
mkdir -p -m 700 "$work"/{models/local,artifacts,adapters,brain-runtime}
key="$work/key"
[ -f "$key" ] || { head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$key"; chmod 600 "$key"; }
endpoint="http://127.0.0.1:$port/v1"
started=$(date +%s)
stamp() { echo "[cpu-stack +$(( $(date +%s) - started ))s] $*"; }
fail() { echo "cpu-stack.sh: FAILED: $*" >&2; exit 1; }
api() { curl -sS --max-time 30 -H "Authorization: Bearer $(cat "$key")" "$@"; }

# The toy's weights are drawn: the mechanism checked here holds for any of them.
toy_seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
stamp "the toy checkpoint is drawn with seed $toy_seed"
"$qwen3" toy --out "$work/models/local/toy" --seed "$toy_seed" --store_layout true > /dev/null

brain_pid=
start_brain() {
    rm -f "$work/ready"
    "$brain" --device cpu --backend cpu serve --openai "127.0.0.1:$port" --max-batch 4 \
        --models-dir "$work/models" --serve-adapters "$work/adapters" --adapter-targets all \
        --training --artifact-store "$work/artifacts" --runtime-dir "$work/brain-runtime" \
        --api-key-file "$key" --ready-file "$work/ready" >> "$work/brain.log" 2>&1 &
    brain_pid=$!
    # brain touches the ready file once every surface is bound; the wait is bounded.
    for _ in $(seq 1 240); do
        [ -f "$work/ready" ] && { stamp "brain serve is ready (pid $brain_pid)"; return 0; }
        kill -0 "$brain_pid" 2> /dev/null || { tail -20 "$work/brain.log" >&2; fail "brain serve exited"; }
        sleep 0.5
    done
    fail "brain serve did not get ready"
}
stop_brain() { [ -n "$brain_pid" ] && kill -9 "$brain_pid" 2> /dev/null && wait "$brain_pid" 2> /dev/null || true; brain_pid=; }
splinter_pid=
stop_splinter() { [ -n "$splinter_pid" ] && kill -9 "$splinter_pid" 2> /dev/null && wait "$splinter_pid" 2> /dev/null || true; splinter_pid=; }
trap 'stop_splinter; stop_brain' EXIT
start_brain

# The task repository: the toy-answers family (a reply that starts with a letter, graded by the task's own test.sh).
# The untrained toy draws its first byte from a nearly flat distribution, so some replies start with a letter, a
# group's rewards often differ, and the policy can learn to start with one.
repo="$work/repo"
mkdir -p "$repo/tasks/toy-answers" "$repo/collections"
cp -r "$family" "$repo/tasks/toy-answers/train"
printf 'schema: splinter.collection/v1\nname: toy\nroot: tasks/toy-answers\n' > "$repo/collections/toy.yaml"
# seed: a recipe states every seed; the rollouts draw from them.
cat > "$repo/online.yaml" << EOF
schema: splinter.recipe/v2
name: online-cpu
inputs:
  course: { kind: files, path: collections/toy.yaml }
models:
  toy: "brain:local/toy"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: train } }
  - id: online
    uses: train.online
    inputs: { tasks: \$stages.tasks.tasks }
    with:
      base: toy
      groups: 12
      group_size: 4
      max_concurrent_attempts: 4
      max_staleness: 2
      rollout_tokens: 32
      step_tokens: 64
      queue_tokens: 256
      publish_every: 1
      checkpoint_every: 1
      seed: 7
      $estimator_with
EOF
printf 'connections:\n  brain: { protocol: openai, url: "%s" }\n' "$endpoint" > "$repo/connections.yaml"

# splinter runs in the task repository, in the background: $splinter_pid is the controller itself, so killing it
# kills the controller and nothing else.
cd "$repo"
spawn_splinter() {
    "$splinter" "$@" --state "$work/state" --models-dir "$work/models" \
        --brain-endpoint "$endpoint" --brain-key-file "$key" --brain-artifact-store "$work/artifacts" \
        --brain-adapter-dir "$work/adapters" --allow-unconfined-workspace >> "$work/splinter.log" 2>&1 &
    splinter_pid=$!
}
db="$work/run/state/controller.db"
events="$work/events"
mkdir -p "$events"
# The controller's groups by state, read from its run state (read-only; the controller may be writing).
groups_in() {
    python3 -I - "$db" "$@" << 'EOF'
import sqlite3, sys
try:
    c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=5)
    print(sum(c.execute("select count(*) from groups where state = ?", (s,)).fetchone()[0] for s in sys.argv[2:]))
except sqlite3.Error:
    print(0)
EOF
}
# Waits (bounded) until `groups_in <states...>` reaches $1, while the process $2 lives.
wait_groups() {
    local want=$1 pid=$2
    shift 2
    for _ in $(seq 1 600); do
        [ "$(groups_in "$@")" -ge "$want" ] && return 0
        kill -0 "$pid" 2> /dev/null || return 1
        sleep 0.5
    done
    fail "no $want groups $* within 300 s"
}
# The frames of every streaming job brain holds now, as JSON lines in $events/$1 (a live job's stream is read for
# a moment: what it has emitted so far is replayed from its start).
snapshot_jobs() {
    local ids
    ids=$(api "$endpoint/jobs" | python3 -I -c 'import json, sys; print(" ".join(j["id"] for j in json.load(sys.stdin)["data"]))')
    for id in $ids; do
        api -N --max-time 3 "$endpoint/jobs/$id/events" 2> /dev/null | sed -n 's/^data: //p' > "$events/$1-$id.jsonl" || true
    done
}

# 1. The controller is killed once a group is committed, and resumed.
stamp "the run starts"
spawn_splinter run online.yaml --out "$work/run"
wait_groups 1 "$splinter_pid" committed || { tail -20 "$work/splinter.log" >&2; fail "the run ended before a group was committed"; }
snapshot_jobs before-controller-kill
kill -9 "$splinter_pid"; wait "$splinter_pid" 2> /dev/null || true; splinter_pid=
stamp "the controller is killed with $(groups_in committed) groups committed; it resumes"
spawn_splinter resume "$work/run"

# 2. brain is killed while groups are in flight, and started again; the run resumes against it.
committed=$(groups_in committed)
wait_groups $((committed + 1)) "$splinter_pid" committed || true
snapshot_jobs before-brain-kill
in_flight=$(groups_in sent applied)
stop_brain
stamp "brain is killed with $in_flight groups sent or applied and not committed"
start_brain
# The controller sees the job it fed gone, starts the run's next epoch under the same resume key and sends again
# what brain had not committed; it does not stop.
wait "$splinter_pid" || { tail -20 "$work/splinter.log" >&2; fail "the run did not survive brain's restart"; }
splinter_pid=
grep -q "the learner restarted" "$work/splinter.log" || fail "brain's restart did not surface as a learner restart"
stamp "the run finished"
snapshot_jobs after

# 3. The products.
report="$work/run/online/report.json"
[ -f "$report" ] || fail "no report at $report"
served=$(api "$endpoint/models" | python3 -I -c '
import json, sys
for m in json.load(sys.stdin)["data"]:
    if m["id"].startswith("splinter/online:"):
        print(m["id"], m["adapter"]["version"])' | head -1)
[ -n "$served" ] || fail "no published version is served"
read -r served_id served_version <<< "$served"
reply_version=$(api -H 'content-type: application/json' "$endpoint/chat/completions" -d "{\"model\": \"$served_id\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with one letter.\"}], \"max_tokens\": 4}" \
    | python3 -I -c 'import json, sys; print(json.load(sys.stdin)["brain"]["policy_version"])')
[ "$reply_version" = "$served_version" ] || fail "the reply names version $reply_version, the server lists $served_version"
stamp "$served_id is served at version $served_version, and a reply names it"

python3 -I - "$report" "$work/run/online/experiences.jsonl" "$events" "$in_flight" << 'EOF'
import json, pathlib, sys
report = json.load(open(sys.argv[1]))
experiences = [json.loads(line) for line in open(sys.argv[2])]
in_flight = int(sys.argv[4])
g = report["groups"]
def check(ok, what):
    if not ok:
        sys.exit(f"cpu-stack.sh: FAILED: {what}: {json.dumps(report, indent=1)}")

# Every group is accounted for: committed, filtered or dropped.
filtered = sum(g["filtered"].values())
check(g["started"] == g["committed"] + filtered + g["dropped_stale"] + g["dropped_by_learner"], "the groups add up")
check(g["scored"] == g["started"], "every group was scored")
check(g["committed"] > 0 and g["applied"] >= g["committed"], "groups were applied and committed")
check(g["sent"] >= g["committed"], "every committed group was sent")
r = report["rollouts"]
check(r["finished"] + r["abandoned"] >= g["started"], "rollouts finished")
check(report["tokens_trained"] > 0 and report["learner"]["trained_tokens"] == report["tokens_trained"], "tokens trained")
check(report["tokens_generated"] >= report["tokens_trained"], "no more tokens trained than generated")
published = report["learner"]["published"]
versions = [p[0] for p in published]
check(versions == sorted(set(versions)) and len(versions) >= 2, "versions were published, each newer")
# The committed groups are the experiences, each once.
keys = [e["group"] for e in experiences]
check(len(keys) == len(set(keys)) == g["committed"], "each committed group is one experience")

# brain's receipts over every job the run had, each job one epoch of the run (its `stream` frame): no group committed
# twice, and none applied by a later epoch than the one that committed it (an epoch applies again only what an
# earlier one lost uncommitted).
applied, committed = {}, {}
for path in sorted(pathlib.Path(sys.argv[3]).glob("*.jsonl")):
    frames = [json.loads(line) for line in open(path) if line.strip()]
    epoch = next((f["epoch"] for f in frames if f.get("type") == "stream"), None)
    if epoch is None:
        continue
    for frame in frames:
        kind = frame.get("type")
        if kind in ("applied", "committed"):
            for key in frame["groups"]:
                (applied if kind == "applied" else committed).setdefault(key, set()).add(epoch)
check(applied and committed, "brain reported applied and committed receipts")
check(all(len(epochs) == 1 for epochs in committed.values()), "no group was committed by two epochs")
twice = [k for k, epochs in applied.items() if k in committed and max(epochs) > min(committed[k])]
check(not twice, f"groups applied again after they were committed: {twice}")
# The frames are those each job had sent when it was read (a job of a killed server is not read again), so brain's
# own commits are a subset of the run's; each of them is an experience.
check(set(committed) <= set(keys), "every group brain committed is an experience")
# brain's restart surfaced and what it had not committed was sent again.
if in_flight > 0:
    check(g["resends"] > 0 or report["learner_restarts"] > 0, "the groups brain lost were sent again")
print(f"groups: {g['started']} started, {g['committed']} committed, {filtered} filtered, {g['resends']} resent; "
      f"versions {versions[0]}..{versions[-1]}; {report['tokens_trained']} tokens trained of {report['tokens_generated']} "
      f"generated; restarts {report['learner_restarts']}")
EOF
stamp "the report's counters are consistent and brain's receipts show no group trained twice"
