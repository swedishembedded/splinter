#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements verifiable GPU-only execution for local
# model training and inference for its clients. If your team needs expertise
# in proving that a run never touched the CPU backend, you can procure our
# services by sending an email to info@swedishembedded.com.

# Run a command pinned to GPUs and stop it the moment it reports that any part
# of the run left them.
#
# `BRAIN_DEVICE` alone is not a guarantee: a value that does not resolve only
# warns, and brain then schedules on whatever it finds, the CPU included. So
# the run is also read as it goes. It is stopped (exit 3) on any line saying
# the CPU backend, a software adapter, a fallback or an unplaced model was
# used, and it fails (exit 4) if it never reported a discrete GPU adapter.
#
# `BRAIN_DEVICE` does not choose the card of an in-process run: without
# `BRAIN_GPU_INDEX` it lands on gpu0 whatever it names. So a single card is
# pinned through both.
#
# Usage: gpu-only.sh <gpu0|gpu1|gpu0,gpu1|gpu> <log-file> -- <command> [args...]

set -uo pipefail

if [ "$#" -lt 4 ] || [ "$3" != "--" ]; then
    echo "usage: $0 <device> <log-file> -- <command> [args...]" >&2
    exit 2
fi
device=$1
log=$2
shift 3

markers='brain-wgsl-cpu|\(Cpu,|falling back|no GPU has room|not JIT-compiled'

export BRAIN_DEVICE="$device"
if [[ "$device" =~ ^gpu([0-9]+)$ ]]; then
    export BRAIN_GPU_INDEX="${BASH_REMATCH[1]}"
else
    unset BRAIN_GPU_INDEX
fi
: >"$log"

fifo=$(mktemp -u)
mkfifo "$fifo"
trap 'rm -f "$fifo"' EXIT

# A session of its own, so stopping the run stops everything it started.
setsid "$@" >"$fifo" 2>&1 &
pid=$!

saw_gpu=0
while IFS= read -r line; do
    printf '%s\n' "$line" >>"$log"
    if grep -Eq "$markers" <<<"$line"; then
        echo "gpu-only: the run left the GPUs: $line" >&2
        kill -TERM -- "-$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        exit 3
    fi
    if [[ "$line" == adapter:*DiscreteGpu* ]]; then
        saw_gpu=1
    fi
done <"$fifo"

wait "$pid"
status=$?
if [ "$saw_gpu" -ne 1 ]; then
    echo "gpu-only: the run never reported a discrete GPU adapter" >&2
    exit 4
fi
exit "$status"
