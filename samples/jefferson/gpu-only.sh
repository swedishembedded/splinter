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
# or an unplaced model was used, and it fails (exit 4) if no named card ever
# held more than 256 MiB, which is how a run that never reached a GPU shows
# whatever the backend prints. A device naming the CPU is refused (exit 2).
# Stopping the guard (SIGTERM, SIGINT or SIGHUP) stops the run with it.
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

if [[ "$device" == *cpu* ]]; then
    echo "gpu-only: $device names the CPU" >&2
    exit 2
fi

# The cards the run may use, as nvidia-smi indexes them; every card for `gpu`.
if [[ "$device" =~ gpu[0-9] ]]; then
    cards=$(grep -o 'gpu[0-9]*' <<<"$device" | tr -d 'gpu' | paste -sd, -)
else
    cards=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
fi

markers='brain-wgsl-cpu|\(Cpu,|falling back|no GPU has room|not JIT-compiled'

: >"$log"

fifo=$(mktemp -u)
mkfifo "$fifo"
peak=$(mktemp)
echo 0 >"$peak"
pid=
# The run's whole session, asked to stop and killed if it is still there ten
# seconds later: a guard that is stopped must not leave a run holding cards.
stop_run() {
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 0
    kill -TERM -- "-$pid" 2>/dev/null
    for _ in $(seq 20); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    kill -KILL -- "-$pid" 2>/dev/null
}
cleanup() {
    stop_run
    kill "$sampler" 2>/dev/null
    wait "$sampler" 2>/dev/null
    rm -f "$fifo" "$peak"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# The most memory any named card held while the run lived.
(
    while sleep 2; do
        now=$(nvidia-smi -i "$cards" --query-gpu=memory.used --format=csv,noheader,nounits |
            sort -n | tail -1)
        if [ "${now:-0}" -gt "$(cat "$peak")" ]; then echo "$now" >"$peak"; fi
    done
) &
sampler=$!

# A session of its own, so stopping the run stops everything it started.
setsid "$1" --device "$device" "${@:2}" >"$fifo" 2>&1 &
pid=$!

while IFS= read -r line; do
    printf '%s\n' "$line" >>"$log"
    if grep -Eq "$markers" <<<"$line"; then
        echo "gpu-only: the run left the GPUs: $line" >&2
        kill -TERM -- "-$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        exit 3
    fi
done <"$fifo"

wait "$pid"
status=$?
if [ "$(cat "$peak")" -lt 256 ]; then
    echo "gpu-only: no card of $device ever held 256 MiB; the run did not use a GPU" >&2
    exit 4
fi
exit "$status"
