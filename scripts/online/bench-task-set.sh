#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# train.online over a bench environment's task set on a served model: the end-to-end run a GPU node carries out.
# From a directory of an environment's training task folders (never exam material: a sealed folder is refused when
# the run imports it, and a path naming an exam is refused here) it takes a small, disjoint subset of each kind -
# training tasks, a held-out monitor pool and a sealed confirm set - writes a recipe that runs each rollout in its
# task's own environment with the tools the task declares (the agent `sven`, the bench environment file), the frontier
# selector, episodes pinned to their version and the watchdog, and runs it against `brain serve --openai
# --training` (a brain that retains versions: 2026-11-27 or later).
#
# usage: scripts/online/bench-task-set.sh <splinter> <task dir> <environment file> <base model> <work dir> \
#            <brain endpoint> <brain key file> <artifact store> <adapter dir> [prepare]
#   <task dir>          the environment's training task folders (one task.toml each)
#   <environment file>  the bench environment the tasks run in (environments/<env>/environment.yaml)
#   <base model>        the brain model id of the base, as the server lists it
#   prepare             write the repository and recipe and print the command, run nothing
# Sizes: TRAIN (12), MONITOR (8), CONFIRM (4) tasks; ONLINE_GROUPS (32) of GROUP_SIZE (8).

set -euo pipefail

usage="usage: bench-task-set.sh <splinter> <tasks> <environment> <base> <work> <endpoint> <key> <artifacts> <adapters> [prepare]"
splinter=${1:?$usage}
tasks=$(realpath "${2:?$usage}")
environment=$(realpath "${3:?$usage}")
base=${4:?$usage}
work=${5:?$usage}
endpoint=${6:?$usage}
key=$(realpath "${7:?$usage}")
artifacts=$(realpath "${8:?$usage}")
adapters=$(realpath "${9:?$usage}")
prepare=${10:-}
train=${TRAIN:-12}
monitor=${MONITOR:-8}
confirm=${CONFIRM:-4}
groups=${ONLINE_GROUPS:-32}
group_size=${GROUP_SIZE:-8}

case "$tasks" in *exam*) echo "bench-task-set.sh: $tasks names exam material, which is never trained on" >&2; exit 2 ;; esac
mapfile -t folders < <(find "$tasks" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/task.toml' ';' -print | sort)
need=$((train + monitor + confirm))
[ "${#folders[@]}" -ge "$need" ] || { echo "bench-task-set.sh: $tasks holds ${#folders[@]} task folders, $need are needed" >&2; exit 2; }

mkdir -p -m 700 "$work"
work=$(realpath "$work")
repo="$work/repo"
rm -rf "$repo"
mkdir -p "$repo/tasks/set/train" "$repo/tasks/set/monitor" "$repo/tasks/set/confirm" "$repo/collections" "$repo/environments"
# Disjoint subsets, in the folders' sorted order: training first, then the held-out monitor pool and confirm set.
for i in "${!folders[@]}"; do
    if [ "$i" -lt "$train" ]; then split=train
    elif [ "$i" -lt $((train + monitor)) ]; then split=monitor
    elif [ "$i" -lt "$need" ]; then split=confirm
    else break; fi
    cp -r "${folders[$i]}" "$repo/tasks/set/$split/"
done
cp "$environment" "$repo/environments/environment.yaml"
printf 'schema: splinter.collection/v1\nname: set\nroot: tasks/set\n' > "$repo/collections/set.yaml"
printf 'connections:\n  brain: { protocol: openai, url: "%s" }\n' "$endpoint" > "$repo/connections.yaml"
# seed: the recipe states its seed; record it with the run.
seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
cat > "$repo/online.yaml" << EOF
schema: splinter.recipe/v2
name: online-bench
inputs:
  course: { kind: files, path: collections/set.yaml }
models:
  policy: "brain:$base"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: train } }
  - { id: monitor, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: monitor } }
  - { id: confirm, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: confirm } }
  - id: online
    uses: train.online
    inputs: { tasks: \$stages.tasks.tasks, monitor: \$stages.monitor.tasks, confirm: \$stages.confirm.tasks }
    with:
      base: policy
      groups: $groups
      group_size: $group_size
      max_concurrent_attempts: 16
      max_staleness: 2
      rollout_tokens: 8192
      step_tokens: 32768
      queue_tokens: 131072
      publish_every: 1
      checkpoint_every: 1
      temperature: 1.0
      seed: $seed
      selector: frontier
      pin_episodes: true
      agent: { agent: sven, environment: environments/environment.yaml }
      watchdog:
        bounds: { guard_skip_rate: 0.25, ratio_max: 20, truncation_rate: 0.6, infra_failure_rate: 0.3, window: 8, min_rollouts: 16 }
        drift: { slack: 0.05, threshold: 0.3, reference_window: 3 }
        test: { delta: 0.3, alpha: 0.1, beta: 0.2, max_discordant: 16 }
        evaluate_every: 4
        monitor_size: $monitor
        max_uses: 8
        max_rollbacks: 2
        cooldown: 0.5
EOF
command=("$splinter" run online.yaml --out "$work/run" --state "$work/state"
    --brain-endpoint "$endpoint" --brain-key-file "$key" --brain-artifact-store "$artifacts" --brain-adapter-dir "$adapters")
echo "bench-task-set.sh: $train training, $monitor monitor and $confirm confirm tasks from $tasks in $repo (seed $seed)"
echo "bench-task-set.sh: (cd $repo && ${command[*]})"
[ "$prepare" = prepare ] && exit 0
cd "$repo"
"${command[@]}"
report="$work/run/online/report.json"
python3 -I - "$report" << 'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("groups", r["groups"])
print("rollouts", r["rollouts"])
print("versions published", len(r["versions"]))
for v, s in sorted(r["health"]["versions"].items(), key=lambda x: int(x[0])):
    print("version", v, s)
w = r.get("watchdog") or {}
print("watchdog: comparisons", w.get("comparisons"), "promotions", [p["version"] for p in w.get("promotions", [])],
      "rollbacks", len(w.get("audits", [])), "alarms", len(w.get("alarms", [])))
PY
