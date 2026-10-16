#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible runs of self-improving model
# pipelines on a single GPU for its clients. If your team needs expertise in
# measuring a learning loop end to end, you can procure our services by
# sending an email to info@swedishembedded.com.
#
# The exact commands of experiments E1 and E2/E3 of the paper (second wave).
#
#   run_experiments.sh e1 SPLINTER_BIN WORK_DIR DOCS_DIR MODELS_DIR GENERATOR
#   run_experiments.sh e2 SPLINTER_BIN WORK_DIR BRAIN_BIN MODELS_DIR POLICY [LEARN_ARG...]
#
# DOCS_DIR holds the three Markdown pages of an E1 corpus; MODELS_DIR holds the Qwen3
# checkpoints; GENERATOR (E1) and POLICY (E2) are checkpoint names in it, for example
# Qwen3-4B and Qwen3-0.6B. The registered E2 run used
#   --learn-arg=--eval-every --learn-arg=0 --learn-arg=--steps --learn-arg=400
#   --learn-arg=--rank --learn-arg=16 --learn-arg=--records-per-step --learn-arg=2
#   --learn-arg=--rehearsal --learn-arg=0.25
# (given here as LEARN_ARG words in that form). The shell that runs it needs the CUDA
# runtime on its library path and BRAIN_BACKEND=cuda, as `brain serve` in the gate's serve
# check inherits it. Copy the outputs to data/runs/w2 with archive_run.sh.
set -euo pipefail
what=$1 splinter=$2 work=$3
mkdir -p "$work"
here=$(cd "$(dirname "$0")" && pwd)
export BRAIN_BACKEND=${BRAIN_BACKEND:-cuda}
case $what in
e1)
	docs=$4 models=$5 generator=$6
	export BRAIN_QWEN_WEIGHTS=$models/Qwen3-0.6B
	mkdir -p "$work/state"
	"$splinter" --state "$work/state" --json source add "$docs" >"$work/source.json"
	sid=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['source']['id'])" "$work/source.json")
	"$splinter" --state "$work/state" --json -v --policy-context-tokens 8192 \
		tasks generate "$sid" --kinds recall,explain,classify,denoise \
		--generator "local:$models/$generator@8192" >"$work/tasks.json" 2>"$work/tasks.err"
	python3 "$here/classify_rejections.py" "$work/tasks.json"
	;;
e2)
	brain=$4 models=$5 policy=$6
	shift 6
	python3 "$here/flywheel_cycle.py" build --out "$work"
	python3 -I "$here/flywheel_cycle.py" run --out "$work" --splinter "$splinter" --brain "$brain" \
		--policy-weights "$models/$policy" --helper "$models/Qwen3-4B" "$@"
	python3 "$here/flywheel_cycle.py" summary --out "$work"
	;;
*) echo "usage: $0 e1|e2 ..." >&2; exit 2 ;;
esac
