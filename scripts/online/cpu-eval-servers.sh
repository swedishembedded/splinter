#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The online watchdog evaluating on a server of its own, on the real stack, on the CPU, with the rollouts on the
# execution worker: two `brain serve --openai --training` servers of one toy qwen3 checkpoint sharing an artifact store
# (the learner, and the evaluation server named by --brain-eval-endpoint), and `train.online` with `agent: sven` over
# the toy-answers family (training tasks, and copies of them as the monitor pool and the confirm set), every rollout
# run by the Harbor worker (`splinter-harbor-worker` on PATH, `make worker/install`) with sven (`sven` on PATH).
# The rewards are a coin (the worker writes no reply to the family's answer file); what is checked is where the
# evaluations ran:
#   - every compared version was served on the evaluation server (brain's serve_adapter) and withdrawn afterwards;
#   - the learner answered the training's worker rollouts and none for a version id, while the comparisons graded
#     pairs: the evaluations' worker rollouts were answered by the evaluation server.
#
# usage: scripts/online/cpu-eval-servers.sh <splinter> <brain> <qwen3 sample> <work dir> <port>
#   <port> is the learner's; the evaluation server listens on <port> + 1.

set -euo pipefail

splinter=$(realpath "${1:?usage: cpu-eval-servers.sh <splinter> <brain> <qwen3> <work> <port>}")
brain=$(realpath "${2:?brain binary}")
qwen3=$(realpath "${3:?qwen3 sample binary}")
work=${4:?work directory}
port=${5:?port}
eval_port=$((port + 1))
for p in "$port" "$eval_port"; do
    case "$p" in 8788 | 8789 | 8790 | 8791) echo "cpu-eval-servers.sh: $p belongs to the shared servers" >&2; exit 2 ;; esac
done
command -v splinter-harbor-worker > /dev/null || { echo "cpu-eval-servers.sh: splinter-harbor-worker is not on PATH (make worker/install)" >&2; exit 2; }
command -v sven > /dev/null || { echo "cpu-eval-servers.sh: sven is not on PATH" >&2; exit 2; }
family=$(realpath "$(dirname "$0")/../../tasks/toy-answers")
# The image every task runs in: the base the worker's own fixtures pin.
IMAGE=python:3.13-slim@sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50

mkdir -p -m 700 "$work"
work=$(realpath "$work")
rm -rf "${work:?}"/{models,artifacts,learner,evaluator,repo,run,state,splinter.log,config.yaml}
mkdir -p -m 700 "$work"/{models/local,artifacts,learner/adapters,learner/runtime,evaluator/adapters,evaluator/runtime}
secret="$work/secret"
[ -f "$secret" ] || { head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$secret"; chmod 600 "$secret"; }
learner="http://127.0.0.1:$port/v1"
evaluator="http://127.0.0.1:$eval_port/v1"
started=$(date +%s)
stamp() { echo "[cpu-eval-servers +$(($(date +%s) - started))s] $*"; }
fail() { echo "cpu-eval-servers.sh: FAILED: $*" >&2; exit 1; }

toy_seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
stamp "the toy checkpoint is drawn with seed $toy_seed"
"$qwen3" toy --out "$work/models/local/toy" --seed "$toy_seed" --store_layout true > /dev/null
# The agent offers tools, and brain refuses a request with tools for a template that has no place for them: the toy's
# template names the tools offered (and leaves out the system prompt, so a prompt stays short for the toy).
python3 -I - "$work/models/local/toy/tokenizer_config.json" << 'PY'
import json, sys
path = sys.argv[1]
config = json.load(open(path))
config["chat_template"] = (
    "{% if tools %}{% for t in tools %}{{ t.function.name if t.function is defined else t.name }} {% endfor %}{% endif %}"
    "{% for m in messages %}{% if m.role == 'user' or m.role == 'tool' %}{{ m.content }}={% elif m.role == 'assistant' %}"
    "{{ m.content or '' }}<|im_end|>{% endif %}{% endfor %}"
)
json.dump(config, open(path, "w"))
PY

# Both servers serve the same base and reach the same artifact store; each has its own adapters and runtime.
pids=()
serve() {
    local who=$1 at=$2
    (
        "$brain" --device cpu --backend cpu serve --openai "127.0.0.1:$at" --max-batch 4 \
            --models-dir "$work/models" --serve-adapters "$work/$who/adapters" --adapter-targets all --adapter-slots 16 \
            --training --artifact-store "$work/artifacts" --runtime-dir "$work/$who/runtime" \
            --api-key-file "$secret" --ready-file "$work/$who/ready" >> "$work/$who/brain.log" 2>&1
        echo "brain exited with status $?" >> "$work/$who/brain.log"
    ) &
    pids+=($!)
}
serve learner "$port"
serve evaluator "$eval_port"
trap 'for p in "${pids[@]}"; do pkill -9 -P "$p" 2> /dev/null || true; kill -9 "$p" 2> /dev/null || true; done' EXIT
for who in learner evaluator; do
    for _ in $(seq 1 240); do
        [ -f "$work/$who/ready" ] && break
        sleep 0.5
    done
    [ -f "$work/$who/ready" ] || { tail -20 "$work/$who/brain.log" >&2; fail "the $who's brain serve did not get ready"; }
done
stamp "both servers are ready"

# The family's tasks as the training set, and copies of them as the held-out monitor pool and confirm set.
repo="$work/repo"
for split in train monitor confirm; do
    for dir in "$family"/*/; do
        name=$(basename "$dir")
        mkdir -p "$repo/tasks/toy/$split"
        cp -r "$dir" "$repo/tasks/toy/$split/$name"
        task="$repo/tasks/toy/$split/$name"
        sed -i "s|^name = \"toy-answers/$name\"|name = \"toy/$split-$name\"|" "$task/task.toml"
        # The worker's sandbox runs a task in its own image: the pinned Python base the worker's own fixtures use,
        # with the task's starting files.
        mkdir -p "$task/environment"
        [ -d "$task/workspace" ] && mv "$task/workspace" "$task/environment/workspace" || mkdir -p "$task/environment/workspace"
        printf '%s\n' "FROM $IMAGE" 'COPY workspace/ /workspace/repo/' 'WORKDIR /workspace/repo' > "$task/environment/Dockerfile"
        printf '\n[environment]\nbuild_timeout_sec = 600.0\nos = "linux"\n' >> "$task/task.toml"
        # The worker's verifier convention (/logs/verifier/reward.txt). The worker writes no reply to answer.txt, so
        # the reward is a coin: it gives the groups the variance a step trains on, and the run checks where the
        # evaluations ran, not what was learned.
        printf '%s\n' '#!/bin/bash' 'mkdir -p /logs/verifier' 'echo $((RANDOM % 2)) > /logs/verifier/reward.txt' > "$task/tests/test.sh"
        chmod +x "$task/tests/test.sh"
    done
done
mkdir -p "$repo/collections" "$repo/environments"
printf 'schema: splinter.collection/v1\nname: toy\nroot: tasks/toy\n' > "$repo/collections/toy.yaml"
printf 'schema: splinter.environment/v1\nsandbox: { resources: { timeout: 5m } }\nnetwork: { mode: model-gateway-only }\n' \
    > "$repo/environments/toy.yaml"
# The worker's model gateway reaches the server with the key it reads from its own environment (TOY_BRAIN_KEY): a
# credential is the machine's to give, so the connection is this run's own machine configuration (SPLINTER_CONFIG).
printf 'schema: splinter.config/v1\nconnections:\n  brain: { protocol: openai, url: "%s", token: "env:TOY_BRAIN_KEY" }\n' "$learner" \
    > "$work/config.yaml"
# seed: a recipe states every seed; the rollouts draw from them.
cat > "$repo/online.yaml" << EOF
schema: splinter.recipe/v2
name: online-eval-servers
inputs:
  course: { kind: files, path: collections/toy.yaml }
models:
  toy: "brain:local/toy"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: train } }
  - { id: monitor, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: monitor } }
  - { id: confirm, uses: tasks.pin, inputs: { collection: \$inputs.course }, with: { split: confirm } }
  - id: online
    uses: train.online
    inputs: { tasks: \$stages.tasks.tasks, monitor: \$stages.monitor.tasks, confirm: \$stages.confirm.tasks }
    with:
      base: toy
      groups: 10
      group_size: 4
      max_concurrent_attempts: 4
      max_staleness: 2
      rollout_tokens: 64
      step_tokens: 64
      queue_tokens: 256
      max_output_tokens: 24
      publish_every: 1
      checkpoint_every: 1
      seed: 13
      params: { lr: 0.02 }
      agent: { agent: sven, environment: environments/toy.yaml }
      watchdog:
        bounds: { guard_skip_rate: 1.0, ratio_max: 1000, truncation_rate: 1.0, infra_failure_rate: 1.0, window: 8, min_rollouts: 4 }
        drift: { slack: 0.05, threshold: 10.0, reference_window: 3 }
        test: { delta: 0.25, alpha: 0.05, beta: 0.2, max_discordant: 3 }
        evaluate_every: 2
        monitor_size: 3
        max_uses: 100
        max_rollbacks: 4
        cooldown: 0.5
EOF

cd "$repo"
stamp "the run starts"
SPLINTER_CONFIG="$work/config.yaml" TOY_BRAIN_KEY=$(cat "$secret") "$splinter" run online.yaml --out "$work/run" --state "$work/state" --models-dir "$work/models" \
    --brain-endpoint "$learner" --brain-eval-endpoint "$evaluator" --brain-key-file "$secret" \
    --brain-artifact-store "$work/artifacts" --brain-adapter-dir "$work/learner/adapters" >> "$work/splinter.log" 2>&1 \
    || { tail -30 "$work/splinter.log" >&2; fail "the run failed"; }
stamp "the run finished"

python3 -I - "$work" "$learner" "$evaluator" << 'PY'
import json, os, re, sys, urllib.request
work, learner, evaluator = sys.argv[1:4]
def check(ok, what):
    if not ok:
        sys.exit(f"cpu-eval-servers.sh: FAILED: {what}")
report = json.load(open(os.path.join(work, "run/online/report.json")))
watchdog = report.get("watchdog") or {}
history = [h for e in watchdog.get("rolled_back_epochs", []) for h in e["watchdog"].get("history", [])] + watchdog.get("history", [])
print(f"cpu-eval-servers: groups {report['groups']}")
print(f"cpu-eval-servers: {watchdog.get('comparisons')} comparisons, rollbacks {len(watchdog.get('audits', []))}")
check(watchdog.get("comparisons", 0) >= 1, "the watchdog compared versions")
log = lambda who: open(os.path.join(work, who, "brain.log"), errors="replace").read()
served = sorted(re.findall(r"brain train: serving adapter [0-9a-f]+ as (\S+) \(version", log("evaluator")))
withdrawn = sorted(re.findall(r"brain train: withdrew (\S+)", log("evaluator")))
print(f"cpu-eval-servers: the evaluation server served {len(served)} versions and withdrew {len(withdrawn)}")
check(served, "the compared versions were served on the evaluation server")
check(served == withdrawn, f"each served version was withdrawn afterwards: served {served}, withdrawn {withdrawn}")
check(not re.search(r"brain train: serving adapter", log("learner")), "the learner served no stored version")
# Which models the learner answered: its metrics count completed requests per served model. The evaluations asked
# version ids; the learner retains those it compared (it holds them while compared) and answered none of them, while
# the comparisons graded pairs, which a rollout whose request failed never is: the evaluation server answered them.
def metrics(url):
    key = open(os.path.join(work, "secret")).read().strip()
    request = urllib.request.Request(url.removesuffix("/v1") + "/v1/metrics", headers={"Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(request, timeout=30))
learned = {m["model"]: m["served"]["requests_completed"] for m in metrics(learner)["models"]}
versions = {m: n for m, n in learned.items() if re.search(r"-v[0-9]+$", m)}
published = {m: n for m, n in learned.items() if m.startswith("splinter/online:") and m not in versions}
pairs = sum(h["comparison"]["pairs"] for h in history)
print(f"cpu-eval-servers: the learner answered {sum(published.values())} requests for the published id and "
      f"{sum(versions.values())} for its {len(versions)} retained versions; the comparisons graded {pairs} pairs")
check(sum(published.values()) > 0, "the training's worker rollouts reached the learner")
check(versions and not any(versions.values()), f"no evaluation reached the learner: {versions}")
check(pairs > 0, "the evaluations' worker rollouts were answered (by the evaluation server)")
PY
stamp "passed"
