#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements measured validation of continual learning
# that consolidates into the weights, for its clients. If your team needs
# expertise in proving that a model keeps accumulating capabilities
# generation after generation, you can procure our services by sending an
# email to info@swedishembedded.com.

# One run of the generations sample on one card: claims the card, pins the run
# to it, watches its memory, and releases the card when it ends.
#
# Usage: run.sh <gpu0|gpu1> <full|merge> <out-dir>
#
# The environment names what the run is made of:
#   SPLINTER_BIN     the splinter binary
#   BRAIN_BIN        the brain binary the serve check starts
#   POLICY_WEIGHTS   the policy's base checkpoint
#   HELPER_WEIGHTS   the checkpoint that plays generator, teacher and judge
#   NVML_LIB         libnvidia-ml.so.1, for the memory watcher
#   CLAIM_DIR        where the card claim files are kept
#   GPU_ONLY         the gpu-only.sh pin (default: samples/jefferson/gpu-only.sh)
#   SEED, GENERATIONS  the facts of a first run (default: the sample's own)
#   WAIT             set to wait for a claimed card instead of ending the run
#   EXTRA            more arguments for `generations.py run`
set -uo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <gpu0|gpu1> <full|merge> <out-dir>" >&2
    exit 2
fi
card=$1
mode=$2
out=$3
here=$(cd "$(dirname "$0")" && pwd)
claim="${CLAIM_DIR:?CLAIM_DIR names the directory of card claims}/$card.claim"
gpu_only=${GPU_ONLY:-$here/../jefferson/gpu-only.sh}
index=${card#gpu}

# A card is free when its claim does not exist, or names a process that is gone
# and no pinned job runs on it. WAIT=1 waits for it; otherwise a claimed card
# ends the run.
while true; do
    if [ -e "$claim" ]; then
        holder=$(cut -d' ' -f1 "$claim")
        if kill -0 "$holder" 2>/dev/null ||
            ps -eo args | grep -v grep | grep -q "gpu-only.sh $card"; then
            if [ -z "${WAIT:-}" ]; then
                echo "card $card is claimed: $(cat "$claim")" >&2
                exit 9
            fi
            sleep 30
            continue
        fi
        rm -f "$claim"
    fi
    if ( set -o noclobber; echo "$$ generations sample, mode $mode" > "$claim" ) 2>/dev/null; then
        break
    fi
    sleep 5
done
trap 'rm -f "$claim"' EXIT
mkdir -p "$out"

[ -f "$out/facts.json" ] || python3 "$here/generations.py" build --out "$out" \
    ${SEED:+--seed "$SEED"} ${GENERATIONS:+--generations "$GENERATIONS"}
python3 "$here/memwatch.py" --pid $$ --gpu "$index" --nvml "${NVML_LIB:?}" \
    --out "$out/memwatch.jsonl" &
"$gpu_only" "$card" "$out/run-$mode.log" -- python3 "$here/generations.py" run \
    --out "$out" --mode "$mode" --splinter "${SPLINTER_BIN:?}" --brain "${BRAIN_BIN:?}" \
    --policy-weights "${POLICY_WEIGHTS:?}" --helper "${HELPER_WEIGHTS:?}" ${EXTRA:-}
code=$?
echo "$code" > "$out/exit-$mode"
exit "$code"
