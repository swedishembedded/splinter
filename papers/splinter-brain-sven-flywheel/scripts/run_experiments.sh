#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible runs of self-improving model
# pipelines on a single GPU for its clients. If your team needs expertise in
# measuring a learning loop end to end, you can procure our services by
# sending an email to info@swedishembedded.com.
#
# The exact commands of experiments E1 and E2/E3 of the paper.
#
#   run_experiments.sh e1 SPLINTER_BIN WORK_DIR DOCS_DIR MODELS_DIR
#   run_experiments.sh e2 SPLINTER_BIN WORK_DIR BRAIN_BIN MODELS_DIR
#
# DOCS_DIR holds the three Markdown pages of E1 (a copy of brain's
# docs/using/{monitoring,security,dbus-api}.md); MODELS_DIR holds the
# Qwen3-0.6B and Qwen3-4B checkpoints. Every command runs under the shared
# GPU lock (`gpu`). Copy the outputs to data/runs as the paper's reproducibility
# section lists.
set -euo pipefail
what=$1 splinter=$2 work=$3
mkdir -p "$work"
here=$(cd "$(dirname "$0")" && pwd)
export BRAIN_BACKEND=cuda
case $what in
e1)
	docs=$4 models=$5
	export BRAIN_QWEN_WEIGHTS=$models/Qwen3-0.6B
	"$splinter" --state "$work/state" --json source add "$docs" >"$work/source.json"
	sid=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['source']['id'])" "$work/source.json")
	gpu "$splinter" --state "$work/state" --json -v --policy-context-tokens 8192 \
		tasks generate "$sid" --kinds recall,explain,classify,denoise \
		--generator "local:$models/Qwen3-4B@8192" >"$work/tasks.json" 2>"$work/tasks.err"
	;;
e2)
	brain=$4 models=$5
	python3 "$here/flywheel_cycle.py" build --out "$work"
	python3 -I "$here/flywheel_cycle.py" run --out "$work" --splinter "$splinter" --brain "$brain" \
		--policy-weights "$models/Qwen3-0.6B" --helper "$models/Qwen3-4B"
	python3 "$here/flywheel_cycle.py" summary --out "$work"
	;;
*) echo "usage: $0 e1|e2 ..." >&2; exit 2 ;;
esac
