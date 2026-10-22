#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements unattended long-running model training and
# evaluation campaigns on shared GPUs for its clients. If your team needs
# expertise in scheduling such work without contending for a device, you can
# procure our services by sending an email to info@swedishembedded.com.

# Runs storylines to completion, one run at a time, each on a card nobody else
# is using: a card is free when its claim file is absent (or names a dead
# process) and no process says it is running on that card. The card is claimed
# for one run and given back, so other jobs can take it between runs. A run
# that fails is tried again, three times in all, then the script stops.
#
# Usage: drive.sh <claim-dir> <out-dir> <story-bin> <splinter-bin> <storyline>...
#
# The models and the driver setup are the story binary's own flags; the
# environment carries only the library paths a container without the vendor
# runtime hook needs.

set -uo pipefail

if [ "$#" -lt 5 ]; then
    echo "usage: $0 <claim-dir> <out-dir> <story-bin> <splinter-bin> <storyline>..." >&2
    exit 2
fi
claims=$1
out=$2
story_bin=$3
splinter=$4
shift 4

here=$(cd "$(dirname "$0")" && pwd)
gpu_only="$here/../jefferson/gpu-only.sh"
poll_seconds=60
retry_seconds=120
mkdir -p "$claims" "$out"

held=""
release_claim() {
    [ -n "$held" ] && rm -f "$claims/$held.claim"
    held=""
}
trap release_claim EXIT
trap 'exit 130' INT TERM

# Succeeds, naming the card on stdout and holding its claim, when one is free.
acquire() {
    local storyline=$1 card file pid
    for card in gpu0 gpu1; do
        file="$claims/$card.claim"
        if [ -e "$file" ]; then
            pid=$(cut -d' ' -f1 <"$file" 2>/dev/null | tr -d ' ')
            if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
                continue
            fi
            rm -f "$file"
        fi
        # Another job on the card that never claimed it still occupies it.
        if ps -eo args | grep -v grep | grep -qE -- "--device $card( |$)|gpu-only.sh $card "; then
            continue
        fi
        if (set -o noclobber; echo "$$ drive.sh, storyline $storyline, wt-story" >"$file") 2>/dev/null; then
            held=$card
            echo "$card"
            return 0
        fi
    done
    return 1
}

for storyline in "$@"; do
    failures=0
    while true; do
        "$story_bin" status "$storyline" --out "$out" >/dev/null
        status=$?
        if [ "$status" -eq 0 ]; then
            break
        elif [ "$status" -ne 10 ]; then
            echo "drive: status of $storyline failed ($status)" >&2
            exit "$status"
        fi
        until card=$(acquire "$storyline"); do
            sleep "$poll_seconds"
        done
        mkdir -p "$out/logs"
        "$gpu_only" "$card" "$out/logs/$storyline-$(date +%s).log" -- \
            "$story_bin" step "$storyline" --out "$out" --splinter "$splinter"
        status=$?
        release_claim
        if [ "$status" -ne 0 ]; then
            failures=$((failures + 1))
            echo "drive: a run of $storyline failed ($status), attempt $failures of 3" >&2
            [ "$failures" -ge 3 ] && exit 1
            sleep "$retry_seconds"
        fi
    done
done
