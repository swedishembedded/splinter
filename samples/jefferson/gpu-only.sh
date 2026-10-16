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
# The command is given `--device <device>`, which its program reads with
# brain's grammar. A device the machine lacks is refused by the program, but
# a run can still leave the GPUs, so it is also read as it goes. It is stopped
# (exit 3) on any line saying the CPU backend, a software adapter, a fallback
# or an unplaced model was used, and it fails (exit 4) if it never reported a
# discrete GPU adapter.
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

: >"$log"

fifo=$(mktemp -u)
mkfifo "$fifo"
trap 'rm -f "$fifo"' EXIT

# A session of its own, so stopping the run stops everything it started.
setsid "$1" --device "$device" "${@:2}" >"$fifo" 2>&1 &
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
