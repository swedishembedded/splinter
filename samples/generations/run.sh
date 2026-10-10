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
# Usage: run.sh <gpu0|gpu1> <full|merge|tune> <out-dir>
#
# `tune` takes one generation's updates once and consolidates it at each
# learning rate in LRS, on facts of a seed of its own.
#
# The environment names what the run is made of:
#   SPLINTER_BIN     the splinter binary
#   BRAIN_BIN        the brain binary the serve check starts
#   POLICY_WEIGHTS   the policy's base checkpoint
#   HELPER_WEIGHTS   the checkpoint that plays generator and teacher
#   NVML_LIB         libnvidia-ml.so.1, for the memory watcher
#   CLAIM_DIR        where the card claim files are kept
#   GPU_ONLY         the gpu-only.sh pin (default: samples/jefferson/gpu-only.sh)
#   SEED, GENERATIONS, PER_GENERATION, UPDATES  the facts of a first run (default: the
#                    sample's own); give UPDATES to EXTRA as --updates too
#   WAIT, WAIT_POLL  set WAIT to wait for a claimed card instead of ending the run, and
#                    WAIT_POLL to the seconds between looks at it (default 30)
#   LRS              the learning rates of a tune run
#   RETRY_WAIT       seconds a failed run keeps the card for a mended retry (default 0)
#   EXTRA            more arguments for `generations.py` (such as --absorb-flag=--no-stance)
set -uo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <gpu0|gpu1> <full|merge|tune> <out-dir>" >&2
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
            sleep "${WAIT_POLL:-30}"
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
    ${SEED:+--seed "$SEED"} ${GENERATIONS:+--generations "$GENERATIONS"} \
    ${PER_GENERATION:+--per-generation "$PER_GENERATION"} ${UPDATES:+--updates "$UPDATES"}
python3 "$here/memwatch.py" --pid $$ --gpu "$index" --nvml "${NVML_LIB:?}" \
    --out "$out/memwatch.jsonl" &
if [ "$mode" = tune ]; then
    verb=(tune --lrs ${LRS:?LRS names the learning rates to try})
else
    verb=(run --mode "$mode")
fi
# A run that fails keeps the card for RETRY_WAIT seconds, so that what broke it
# can be mended without losing the place in the queue: the run starts again
# when the file $out/retry appears (it is removed first).
while true; do
    "$gpu_only" "$card" "$out/run-$mode.log" -- python3 "$here/generations.py" "${verb[@]}" \
        --out "$out" --splinter "${SPLINTER_BIN:?}" --brain "${BRAIN_BIN:?}" \
        --policy-weights "${POLICY_WEIGHTS:?}" --helper "${HELPER_WEIGHTS:?}" ${EXTRA:-}
    code=$?
    [ "$code" -eq 0 ] && break
    waited=0
    while [ ! -e "$out/retry" ] && [ "$waited" -lt "${RETRY_WAIT:-0}" ]; do
        sleep 10
        waited=$((waited + 10))
    done
    [ -e "$out/retry" ] || break
    rm -f "$out/retry"
done
echo "$code" > "$out/exit-$mode"
exit "$code"
