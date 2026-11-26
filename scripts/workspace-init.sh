#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training and evaluation workspaces for its clients. If your team needs
# expertise in running model training and agent evaluations on shared clusters without per-machine hand set-up, you
# can procure our services by sending an email to info@swedishembedded.com.
#
# Initialise a run workspace in any directory on any machine, from nothing but the repositories and the network:
#
#   scripts/workspace-init.sh --dir <fast-local-disk>/ws
#   <fast-local-disk>/ws/run.sh recipes/vim-sft-qwen3-4b.yaml gpu1
#
# Everything that is only work in progress (source snapshots, build targets, the worker's Python environment, the
# sandbox image cache, run directories, temporary files) lives under --dir, so it can be a fast local disk that is lost
# with the node. What is worth keeping across nodes lives in --cache: the frozen binaries (one per repository and
# commit, built once and copied from then on) and the downloaded records. Finished products (adapters, results) are
# the caller's to copy to durable storage.
#
# Idempotent: running it again reuses the cache and every step that is already done.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: workspace-init.sh --dir DIR [options]

  --dir DIR            the workspace (created, mode 700); local disk recommended
  --cache DIR          durable cache of frozen binaries and records
                       (default: $WORKSPACE_CACHE, else ~/.cache/splinter-workspace)
  --models-dir DIR     where brain finds model weights (default: brain's own model store)
  --code DIR           where the brain, sven, splinter and bench checkouts are
                       (default: the directory holding this splinter checkout)
  --brain-rev REV      commit of each repository to freeze (default: its checkout's HEAD)
  --sven-rev REV
  --splinter-rev REV
  --bench-rev REV
  --records REPO       Hugging Face dataset of SFT records (default: swedishembedded/vim-headless-mastery-sft)
  --records-rev REV    its tag, branch or commit (default: main, resolved to the commit it names and recorded)
  --records-config C   the configuration of the records to train on: data/C/ of the repository (default: data/)
  --build-env FILE     shell file sourced before any build (toolchain, CUDA paths)
  --jobs N             cargo build jobs (default: 32)
EOF
  exit 2
}

code=$(cd "$(dirname "$0")/../.." && pwd)
dir="" cache="${WORKSPACE_CACHE:-$HOME/.cache/splinter-workspace}" models="" jobs=32
records="swedishembedded/vim-headless-mastery-sft" records_rev="main" records_config="" build_env=""
declare -A rev=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) dir=$2; shift 2 ;;
    --cache) cache=$2; shift 2 ;;
    --models-dir) models=$2; shift 2 ;;
    --code) code=$2; shift 2 ;;
    --brain-rev) rev[brain]=$2; shift 2 ;;
    --sven-rev) rev[sven]=$2; shift 2 ;;
    --splinter-rev) rev[splinter]=$2; shift 2 ;;
    --bench-rev) rev[bench]=$2; shift 2 ;;
    --records) records=$2; shift 2 ;;
    --records-rev) records_rev=$2; shift 2 ;;
    --records-config) records_config=$2; shift 2 ;;
    --build-env) build_env=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$dir" ] || usage
# shellcheck source=/dev/null
[ -z "$build_env" ] || . "$build_env"

mkdir -p "$dir" "$cache/bin" "$cache/records"
chmod 700 "$dir"
dir=$(cd "$dir" && pwd)
mkdir -p "$dir"/{bin,src,target,tmp,cache,runs,state}

say() { printf 'workspace: %s\n' "$*" >&2; }
die() { printf 'workspace: %s\n' "$*" >&2; exit 1; }

# The commit to freeze for a repository: the named one, else the checkout's HEAD. A commit origin does not have is
# refused: a frozen binary must be reproducible by anyone from the repository.
resolve() {
  local name=$1 repo="$code/$1"
  [ -d "$repo/.git" ] || die "no checkout of $name at $repo (--code)"
  local r=${rev[$name]:-HEAD}
  git -C "$repo" rev-parse --verify --quiet "$r^{commit}" >/dev/null || die "$name: unknown revision $r"
  local full; full=$(git -C "$repo" rev-parse "$r^{commit}")
  if ! git -C "$repo" branch -r --contains "$full" 2>/dev/null | grep -q .; then
    die "$name: commit ${full:0:9} is on no remote branch; push it (or name another with --$name-rev)"
  fi
  printf '%s\n' "$full"
}

snapshot() { # NAME FULLREV -> source tree of that commit under $dir/src/NAME-REV
  local name=$1 full=$2 out="$dir/src/$1-${2:0:9}"
  if [ ! -d "$out" ]; then
    mkdir -p "$out.tmp"
    git -C "$code/$name" archive "$full" | tar -x -C "$out.tmp"
    mv "$out.tmp" "$out"
  fi
  printf '%s\n' "$out"
}

# Frozen binaries: built once per commit into the durable cache, then copied.
freeze() { # NAME FULLREV BINARY... -> $dir/bin/BINARY for each
  local name=$1 full=$2; shift 2
  local frozen="$cache/bin/$name-${full:0:9}"
  if [ ! -f "$frozen/commit" ]; then
    say "building $name ${full:0:9} (once; cached in $cache/bin)"
    local src; src=$(snapshot "$name" "$full")
    mkdir -p "$frozen.tmp"
    local bins=(); for b in "$@"; do bins+=(--bin "$b"); done
    (cd "$src" && CARGO_TARGET_DIR="$dir/target/$name" cargo build --release --jobs "$jobs" "${bins[@]}") \
      >"$dir/tmp/build-$name.log" 2>&1 \
      || { tail -n 20 "$dir/tmp/build-$name.log" >&2; die "$name build failed (log: $dir/tmp/build-$name.log)"; }
    for b in "$@"; do cp "$dir/target/$name/release/$b" "$frozen.tmp/$b"; done
    printf '%s\n' "$full" >"$frozen.tmp/commit"
    rm -rf "$frozen"; mv "$frozen.tmp" "$frozen"
    rm -rf "$dir/target/$name"
  fi
  for b in "$@"; do
    cmp -s "$frozen/$b" "$dir/bin/$b" || { cp "$frozen/$b" "$dir/bin/$b.new" && mv -f "$dir/bin/$b.new" "$dir/bin/$b"; }
  done
  say "$name ${full:0:9} ready"
}

brain_full=$(resolve brain); sven_full=$(resolve sven); splinter_full=$(resolve splinter); bench_full=$(resolve bench)
freeze brain "$brain_full" brain brain-gpu-worker
freeze sven "$sven_full" sven
freeze splinter "$splinter_full" splinter

# The Harbor worker's own Python environment, installed from the same splinter commit: reinstalled when the commit changes,
# so the worker is never older than the splinter that starts it.
if [ ! -x "$dir/venv/bin/splinter-harbor-worker" ] || [ "$(cat "$dir/venv/.worker-commit" 2>/dev/null)" != "$splinter_full" ]; then
  say "installing the Harbor worker"
  [ -x "$dir/venv/bin/pip" ] || python3 -m venv "$dir/venv"
  worker="$(snapshot splinter "$splinter_full")/workers/harbor"
  # Its dependencies first (a fresh environment has none of them), then the worker itself again from this commit, which
  # pip would otherwise keep when the version number did not change.
  "$dir/venv/bin/pip" install --quiet "$worker" \
    && "$dir/venv/bin/pip" install --quiet --force-reinstall --no-deps "$worker" \
    || die "the Harbor worker did not install (network?)"
  printf '%s\n' "$splinter_full" >"$dir/venv/.worker-commit"
fi

# The task repository (bench) at its commit, with the SFT records and the generated collections.
tasks="$dir/tasks"
if [ ! -d "$tasks" ]; then
  say "checking out bench ${bench_full:0:9}"
  mkdir -p "$tasks.tmp"
  git -C "$code/bench" archive "$bench_full" | tar -x -C "$tasks.tmp"
  mv "$tasks.tmp" "$tasks"
fi
# The records at one commit: a moving name is resolved first, so the cache and workspace.json name what was used.
records_commit=$(hf datasets info "$records" --revision "$records_rev" 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])') || die "cannot resolve $records@$records_rev"
recs="$cache/records/$(printf '%s' "$records" | tr / _)@${records_commit:0:12}"
if [ ! -f "$recs/data/train.jsonl" ]; then
  say "downloading $records@${records_commit:0:12}"
  hf download "$records" --repo-type dataset --revision "$records_commit" --local-dir "$recs.tmp" >/dev/null \
    || die "download of $records failed"
  rm -rf "$recs"; mv "$recs.tmp" "$recs"
fi
# A dataset that ships its own gate (tools/check_dataset.py with the served system prompt) must pass it.
if [ -f "$recs/tools/check_dataset.py" ]; then
  say "checking the records"
  python3 -I "$recs/tools/check_dataset.py" "$recs/data${records_config:+/$records_config}" --system-prompt "$recs/tools/system-prompt.txt" >"$dir/tmp/check-records.log" 2>&1 \
    || { tail -n 20 "$dir/tmp/check-records.log" >&2; die "the records violate their own invariants"; }
fi
mkdir -p "$tasks/data/vim-sft"
data="$recs/data${records_config:+/$records_config}"
held="$data/validation.jsonl"; [ -f "$held" ] || held="$data/eval.jsonl"
cp "$data/train.jsonl" "$tasks/data/vim-sft/train.jsonl"
cp "$held" "$tasks/data/vim-sft/validation.jsonl"
if [ ! -x "$tasks/.venv/bin/harbor" ]; then
  say "creating the task repository's Python environment (bench/sync)"
  (cd "$tasks" && make bench/sync) >"$dir/tmp/bench-sync.log" 2>&1 \
    || { tail -n 20 "$dir/tmp/bench-sync.log" >&2; die "bench/sync failed (network, uv?)"; }
fi
if [ ! -f "$tasks/collections/vim-headless-mastery.yaml" ]; then
  say "this bench revision has no collections/ to generate: its recipes compile their tasks themselves (collection.compile)"
elif [ ! -d "$tasks/build/collections" ]; then
  say "generating the collections"
  (cd "$tasks" && "$dir/bin/splinter" generate collections/vim-headless-mastery.yaml) >"$dir/tmp/generate.log" 2>&1 \
    || { tail -n 20 "$dir/tmp/generate.log" >&2; die "collection generation failed"; }
fi

cat >"$dir/env.sh" <<EOF
# Generated by workspace-init.sh; source it to work in this workspace.
export WORKSPACE=$dir
export PATH=$dir/bin:$dir/venv/bin:\$PATH
export SPLINTER_SVEN_BIN=$dir/bin/sven
export XDG_CACHE_HOME=$dir/cache
export TMPDIR=$dir/tmp
${build_env:+. $build_env}
EOF

cat >"$dir/run.sh" <<EOF
#!/usr/bin/env bash
# run.sh [--fork PARENT] [--warm MODEL]... RECIPE DEVICE [splinter arguments]: run (or continue) a recipe of the task repository in this workspace
# on DEVICE (brain's grammar: gpu1, gpu0,gpu1, cpu0-7). It starts a brain server for the run (training enabled, the
# adapters it trains served hot), runs splinter against it, and stops the server when splinter ends. The run directory
# is runs/NAME, its state root state/NAME, its brain's files serve/NAME, so recipes may run side by side on other cards.
set -euo pipefail
# --fork PARENT: a run of this workspace (runs/PARENT) whose stages the new run reuses where its recipe leaves them unchanged.
# --warm MODEL: ask the server for one token of MODEL before the recipe starts. A model is loaded by the first request
# that names it, which on a busy disk takes longer than a client waits for a first byte: warm the big ones here.
parent=""; warm=()
while [ "\${1:-}" = "--fork" ] || [ "\${1:-}" = "--warm" ]; do
  if [ "\$1" = "--fork" ]; then parent=\$2; else warm+=("\$2"); fi
  shift 2
done
[ \$# -ge 2 ] || { echo "usage: run.sh [--fork PARENT] RECIPE DEVICE [splinter arguments]" >&2; exit 2; }
. $dir/env.sh
cd $dir/tasks
recipe=\$1 device=\$2; shift 2
name=\$(basename "\$recipe" .yaml)
srv=$dir/serve/\$name
mkdir -p "\$srv/artifacts" "\$srv/adapters" "\$srv/rt" $dir/state/\$name
chmod 700 "\$srv"
[ -s "\$srv/key" ] || head -c 24 /dev/urandom | base64 | tr -d '/+=' >"\$srv/key"
seed=\$(od -An -N4 -tu4 /dev/urandom | tr -d ' '); echo "run.sh: brain seed \$seed" >&2
backend=(); case "\$device" in gpu*) backend=(--backend cuda) ;; esac
# The port is kept with the run: a run refuses to continue when its connections changed, and a stage that
# reaches a model carries the connection's address in its identity, so a fork keeps its parent's port, or it
# would recompute what it means to reuse.
if [ ! -s "\$srv/port" ] && [ -n "\$parent" ] && [ -s "$dir/serve/\$parent/port" ]; then
  cp "$dir/serve/\$parent/port" "\$srv/port"
fi
[ -s "\$srv/port" ] || python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])' >"\$srv/port"
port=\$(cat "\$srv/port")
rm -f "\$srv/ready"
# The task repository's recipes reach brain by the connection named "brain"; point it at this run's server, with
# the key in the environment variable the connection names (a per-run override of the repository's default).
mkdir -p "\$srv/xdg/splinter"
printf 'connections:\\n  brain: { protocol: openai, url: "http://127.0.0.1:%s/v1", token: "env:SPLINTER_BRAIN_KEY" }\\n' \\
  "\$port" >"\$srv/xdg/splinter/config.yaml"
export XDG_CONFIG_HOME="\$srv/xdg" SPLINTER_BRAIN_KEY=\$(cat "\$srv/key")
$dir/bin/brain serve --openai \$port --device "\$device" "\${backend[@]}" --seed "\$seed" --max-batch 128 --training \\
  --artifact-store "\$srv/artifacts" --serve-adapters "\$srv/adapters" --adapter-targets all \\
  --api-key-file "\$srv/key" --ready-file "\$srv/ready" --runtime-dir "\$srv/rt" --cache-dir $dir/cache/brain \\
  ${models:+--models-dir $models} >"\$srv/brain.log" 2>&1 &
brain_pid=\$!
trap 'kill \$brain_pid 2>/dev/null || true' EXIT
for _ in \$(seq 1 600); do
  [ -e "\$srv/ready" ] && break
  kill -0 \$brain_pid 2>/dev/null || { tail -n 20 "\$srv/brain.log" >&2; echo "brain exited before it was ready" >&2; exit 1; }
  sleep 1
done
[ -e "\$srv/ready" ] || { echo "brain was not ready in time (\$srv/brain.log)" >&2; exit 1; }
for model in "\${warm[@]}"; do
  echo "run.sh: warming \$model" >&2
  curl -sf --max-time 1800 -H "Authorization: Bearer \$(cat "\$srv/key")" -H 'Content-Type: application/json' \\
    -d "{\"model\":\"\$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \\
    "http://127.0.0.1:\$port/v1/chat/completions" >/dev/null || { echo "could not warm \$model" >&2; exit 1; }
done
# A GPU run that silently fell back to another backend is a different (much slower) experiment: refuse it.
case "\$device" in gpu*)
  if grep -q "falling back to the" "\$srv/brain.log"; then
    grep "falling back to the" "\$srv/brain.log" | head -n 1 >&2
    echo "brain did not start on CUDA (\$srv/brain.log)" >&2; exit 1
  fi ;;
esac
if [ -n "\$parent" ]; then verb=(fork "$dir/runs/\$parent" --recipe "\$recipe"); else verb=(run "\$recipe"); fi
$dir/bin/splinter "\${verb[@]}" --out $dir/runs/\$name --state $dir/state/\$name \\
  --brain-endpoint http://127.0.0.1:\$port/v1 --brain-key-file "\$srv/key" \\
  --brain-artifact-store "\$srv/artifacts" --brain-adapter-dir "\$srv/adapters" "\$@"
EOF
chmod 700 "$dir/run.sh"

printf '{"brain":"%s","sven":"%s","splinter":"%s","bench":"%s","records":"%s","records_config":"%s","records_commit":"%s"}\n' \
  "$brain_full" "$sven_full" "$splinter_full" "$bench_full" "$records" "$records_config" "$records_commit" >"$dir/workspace.json"
say "ready: $dir  (run: $dir/run.sh recipes/<recipe>.yaml gpuN)"
