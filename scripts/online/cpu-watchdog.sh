#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# The online watchdog on the real stack, on the CPU: a toy qwen3 checkpoint served by `brain serve --openai
# --training`, and `train.online` with a watchdog over a fixture task family made here (a reply that starts with a
# letter: four training tasks, a held-out monitor pool and a sealed confirm set, every one graded by its own
# test.sh).
#
#   fault  - the training tasks' grader is poisoned for a window of gradings after a warm-up: inside it the reward
#            is flipped (a reply that does not start with a letter passes). The policy trained on it regresses on
#            the honest held-out tasks; the run must raise an alarm, evaluate, roll back to the anchor and recover
#            (finish in a later epoch), with the audit in the group log and the report.
#   clean  - the same with no poisoned window: it must evaluate and never roll back.
#   pinned - no watchdog, every episode pinned to the version it started on under publish_every 1: every rollout's
#            span must be one version, and every request asks a version's own id.
#   merged - no watchdog, sessions merged into one record each: at least one merged session must be trained, and the
#            sessions that stayed one record per call counted by why (here the replies a short output bound cut, which
#            are not exact; a toy episode is one call, so none is refused as not append-only).
#
# Wall times printed are evidence of the mechanism only.
#
# usage: scripts/online/cpu-watchdog.sh <splinter> <brain> <qwen3 sample> <work dir> <port> <fault|clean|pinned|merged>

set -euo pipefail

splinter=$(realpath "${1:?usage: cpu-watchdog.sh <splinter> <brain> <qwen3> <work> <port> <fault|clean|pinned>}")
brain=$(realpath "${2:?brain binary}")
qwen3=$(realpath "${3:?qwen3 sample binary}")
work=${4:?work directory}
port=${5:?port}
mode=${6:?mode}
case "$mode" in fault | clean | pinned | merged) ;; *) echo "cpu-watchdog.sh: mode is fault, clean, pinned or merged, not $mode" >&2; exit 2 ;; esac
case "$port" in
    8788 | 8789 | 8790 | 8791) echo "cpu-watchdog.sh: $port belongs to the shared servers" >&2; exit 2 ;;
esac

mkdir -p -m 700 "$work"
work=$(realpath "$work")
rm -rf "${work:?}"/{models,artifacts,adapters,brain-runtime,repo,run,state,fault,ready,brain.log,splinter.log}
mkdir -p -m 700 "$work"/{models/local,artifacts,adapters,brain-runtime,fault}
secret="$work/secret"
[ -f "$secret" ] || { head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$secret"; chmod 600 "$secret"; }
endpoint="http://127.0.0.1:$port/v1"
started=$(date +%s)
stamp() { echo "[cpu-watchdog $mode +$(( $(date +%s) - started ))s] $*"; }
fail() { echo "cpu-watchdog.sh: FAILED ($mode): $*" >&2; exit 1; }

toy_seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
stamp "the toy checkpoint is drawn with seed $toy_seed"
"$qwen3" toy --out "$work/models/local/toy" --seed "$toy_seed" --store_layout true > /dev/null

# brain's exit status is kept: a server that ends under the run says how.
(
    "$brain" --device cpu --backend cpu serve --openai "127.0.0.1:$port" --max-batch 4 \
        --models-dir "$work/models" --serve-adapters "$work/adapters" --adapter-targets all --adapter-slots 16 \
        --training --artifact-store "$work/artifacts" --runtime-dir "$work/brain-runtime" \
        --api-key-file "$secret" --ready-file "$work/ready" >> "$work/brain.log" 2>&1
    echo "brain exited with status $?" >> "$work/brain.log"
) &
brain_pid=$!
trap 'pkill -9 -P "$brain_pid" 2> /dev/null || true; kill -9 "$brain_pid" 2> /dev/null || true' EXIT
for _ in $(seq 1 240); do
    [ -f "$work/ready" ] && break
    kill -0 "$brain_pid" 2> /dev/null || { tail -20 "$work/brain.log" >&2; fail "brain serve exited"; }
    sleep 0.5
done
[ -f "$work/ready" ] || fail "brain serve did not get ready"
stamp "brain serve is ready (pid $brain_pid)"

# The fixture family. The poisoned window counts the training tasks' gradings of a reply (a check of the starting
# files, which hold no reply, is not counted): gradings warm..warm+window are flipped in fault mode.
repo="$work/repo"
warm=48
window=$([ "$mode" = fault ] && echo 128 || echo 0)
task() {
    local split=$1 name=$2 poisoned=$3 dir="$repo/tasks/fault/$1/$2"
    mkdir -p "$dir/tests" "$dir/workspace" "$dir/solution"
    printf 'Name a letter of the alphabet (%s).\n' "$name" > "$dir/instruction.md"
    printf 'Replies are written to answer.txt.\n' > "$dir/workspace/README.md"
    printf 'schema_version = "1.4"\n[task]\nname = "fault/%s-%s"\n[agent]\ntimeout_sec = 120.0\nanswer_file = "answer.txt"\n[verifier]\ntimeout_sec = 30.0\n' \
        "$split" "$name" > "$dir/task.toml"
    printf '#!/bin/bash\necho a > "$WORKSPACE_DIR/answer.txt"\n' > "$dir/solution/solve.sh"
    {
        echo '#!/bin/bash'
        echo 'first=$(head -c 1 "$WORKSPACE_DIR/answer.txt" 2> /dev/null | LC_ALL=C tr -cd "A-Za-z")'
        echo 'reward=0; [ -n "$first" ] && reward=1'
        if [ "$poisoned" = yes ]; then
            echo "if [ -f \"\$WORKSPACE_DIR/answer.txt\" ]; then"
            echo "    n=\$(flock '$work/fault/lock' bash -c 'n=\$((\$(cat \"$work/fault/count\" 2> /dev/null || echo 0) + 1)); echo \$n > \"$work/fault/count\"; echo \$n')"
            echo "    if [ \"\$n\" -gt $warm ] && [ \"\$n\" -le $((warm + window)) ]; then reward=\$((1 - reward)); echo \$n >> '$work/fault/flipped'; fi"
            echo 'fi'
        fi
        echo 'echo "$reward" > "$LOGS_DIR/reward.txt"'
    } > "$dir/tests/test.sh"
    chmod +x "$dir/tests/test.sh" "$dir/solution/solve.sh"
}
for i in 1 2 3 4; do task train "t$i" yes; done
for i in $(seq 1 64); do task monitor "m$i" no; done
for i in $(seq 1 8); do task confirm "c$i" no; done
mkdir -p "$repo/collections"
printf 'schema: splinter.collection/v1\nname: fault\nroot: tasks/fault\n' > "$repo/collections/fault.yaml"
printf 'connections:\n  brain: { protocol: openai, url: "%s" }\n' "$endpoint" > "$repo/connections.yaml"

params="lr: 0.002"
if [ "$mode" = pinned ]; then
    watched="      pin_episodes: true"
    inputs='{ tasks: $stages.tasks.tasks }'
    groups=12
elif [ "$mode" = merged ]; then
    watched="      sessions: merged
      max_output_tokens: 128"
    params="lr: 0.002, max_block: 2048, loss_agg: token-sum"
    inputs='{ tasks: $stages.tasks.tasks }'
    groups=12
else
    inputs='{ tasks: $stages.tasks.tasks, monitor: $stages.monitor.tasks, confirm: $stages.confirm.tasks }'
    groups=48
    watched="      watchdog:
        bounds: { guard_skip_rate: 1.0, ratio_max: 1000, truncation_rate: 1.0, infra_failure_rate: 1.0, window: 8, min_rollouts: 4 }
        drift: { slack: 0.05, threshold: 0.3, reference_window: 3 }
        test: { delta: 0.3, alpha: 0.1, beta: 0.2, max_discordant: 64 }
        evaluate_every: 6
        monitor_size: 64
        max_uses: 100
        max_rollbacks: 2
        cooldown: 0.5"
fi
# seed: a recipe states every seed; the rollouts draw from them.
cat > "$repo/online.yaml" << EOF
schema: splinter.recipe/v2
name: online-watchdog
inputs:
  course: { kind: files, path: collections/fault.yaml }
models:
  toy: "brain:local/toy"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: train } }
  - { id: monitor, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: monitor } }
  - { id: confirm, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: confirm } }
  - id: online
    uses: train.online
    inputs: $inputs
    with:
      base: toy
      groups: $groups
      group_size: 4
      max_concurrent_attempts: 4
      max_staleness: 2
      rollout_tokens: 32
      step_tokens: 64
      queue_tokens: 256
      publish_every: 1
      checkpoint_every: 1
      seed: 11
      params: { $params }
$watched
EOF

cd "$repo"
stamp "the run starts"
"$splinter" run online.yaml --out "$work/run" --state "$work/state" --models-dir "$work/models" \
    --brain-endpoint "$endpoint" --brain-key-file "$secret" --brain-artifact-store "$work/artifacts" \
    --brain-adapter-dir "$work/adapters" --allow-unconfined-workspace >> "$work/splinter.log" 2>&1 \
    || { tail -30 "$work/splinter.log" >&2; grep "brain exited" "$work/brain.log" >&2 || true; fail "the run failed"; }
stamp "the run finished"

python3 -I - "$mode" "$work/run/online/report.json" "$work/run/online/groups.jsonl" "$work/fault/flipped" << 'PY'
import json, os, sys
mode, report, log, flipped = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
lines = [json.loads(l) for l in open(log) if l.strip()]
audits = [l for l in lines if l.get("audit") == "rollback"]
watchdog = report.get("watchdog") or {}
def check(ok, what):
    if not ok:
        sys.exit(f"cpu-watchdog.sh: FAILED ({mode}): {what}: {json.dumps(watchdog, indent=1)[:4000]}")
versions = report["health"]["versions"]
rewards = {int(v): round(s["reward_mean"], 3) for v, s in versions.items() if s.get("reward_mean") is not None}
print(f"{mode}: groups {report['groups']}")
print(f"{mode}: reward by version {rewards}")
if mode == "merged":
    sessions = report["rollouts"]["sessions"]
    print(f"merged: sessions {sessions}, groups committed {report['groups']['committed']}, tokens trained {report['tokens_trained']}")
    check(sessions.get("merged", 0) > 0 and report["groups"]["committed"] > 0, "a merged session was trained")
    check(any(k != "merged" for k in sessions), f"sessions that stayed one record per call are counted: {sessions}")
    sys.exit(0)
if mode == "pinned":
    spans = [r["span"] for l in lines if l.get("state") in ("sent", "filtered", "stale") for r in l["rollouts"] if r.get("span")]
    check(spans and all(a == b for a, b in spans), f"every rollout's span is one version: {spans}")
    ids = [v.get("version_id") for v in report["versions"]]
    check(all(ids), f"every published version names its own id: {ids}")
    print(f"pinned: {len(spans)} rollouts, every span one version; {len(ids)} versions each served under its own id")
    sys.exit(0)
flips = len(open(flipped).read().split()) if os.path.exists(flipped) else 0
for h in watchdog.get("history", []):
    c = h["comparison"]
    print(f"{mode}: comparison {c['index']}: version {h['candidate']} against {h['anchor']}{' (confirm)' if h['confirming'] else ''}: "
          f"{c['candidate_wins']}:{c['anchor_wins']} discordant of {c['pairs']}, {c['decision']}")
print(f"{mode}: {watchdog.get('comparisons')} comparisons, promotions {[p['version'] for p in watchdog.get('promotions', [])]}, "
      f"{len(audits)} rollbacks, {flips} gradings flipped")
check(watchdog.get("comparisons", 0) >= 1, "the watchdog evaluated")
if mode == "clean":
    check(not audits and not watchdog.get("audits"), "a clean run never rolls back")
    sys.exit(0)
check(flips > 0, "the grader was poisoned")
check(audits, "the poisoned run rolled back")
a = audits[0]
e = a["evidence"]
print(f"fault: rollback in epoch {a['epoch']}: candidate version {a['candidate']['version']} against anchor "
      f"{a['anchor']['version']}, {e['anchor_wins']} anchor wins to {e['candidate_wins']} on {e['pairs']} pairs, "
      f"regression e-value {e['regression_e']:.1f} (threshold {(e['index'] + 1) ** 2 * 3.14159265 ** 2 / 6 / 0.1:.0f}), "
      f"trigger {a['trigger']}, {a['groups_discarded']} groups discarded, alarms {len(a['alarms'])}")
check(e["decision"] == "regressed", "the evaluation found the regression")
check(watchdog["epochs"]["epoch"] >= 1, "the run recovered: it finished in a later epoch")
rolled = [l for l in lines if l.get("state") == "rolled_back"]
print(f"fault: recovered, finished in epoch {watchdog['epochs']['epoch']}; {len(rolled)} groups rolled back")
PY
