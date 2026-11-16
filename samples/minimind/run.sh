#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements end-to-end learning systems that take a
# model from random weights to a working tool-using agent, for its clients.
# If your team needs expertise in training small agents from scratch, you can
# procure our services by sending an email to info@swedishembedded.com.
#
# The whole minimind sequence in one command: builds the sample, then runs
# every stage. Run it again to resume after an abort; a finished run is
# reused without training. Every option is passed to `splinter-minimind run`
# (see `--help` there); the run directory defaults to target/minimind/<profile>.
#
# It finds NVRTC for a brain it starts (or takes --nvrtc-dir) and puts brain's
# scratch beside the run directory, refusing a disk with too little room.
#
# usage: samples/minimind/run.sh [--profile smoke|default|64m] [--run-dir DIR] [--nvrtc-dir DIR] [options]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
profile=smoke
run_dir=
nvrtc_dir=
brain_bin=
args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile) profile=$2; args+=("$1" "$2"); shift 2 ;;
        --run-dir) run_dir=$2; shift 2 ;;
        --nvrtc-dir) nvrtc_dir=$2; shift 2 ;;
        --brain-bin) brain_bin=$2; args+=("$1" "$2"); shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
run_dir=${run_dir:-$root/target/minimind/$profile}
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd)

# Brain's training refuses a disk with little room; its scratch goes beside
# the run, on the same disk.
free_kb=$(df -Pk "$run_dir" | awk 'NR == 2 { print $4 }')
if [ "$free_kb" -lt $((10 * 1024 * 1024)) ]; then
    echo "run.sh: $run_dir has $((free_kb / 1024 / 1024)) GB free; brain needs at least 10 GB there" >&2
    exit 2
fi
export TMPDIR="$run_dir/tmp"
mkdir -p "$TMPDIR"

# A brain the run starts compiles its CUDA kernels at run time with NVRTC:
# found in the loader's cache, the library path, beside nvcc, or given with
# --nvrtc-dir. Only that library is put first on the path (prepended, so the
# loader still finds every other driver library the cards need).
if [ -n "$brain_bin" ]; then
    if [ -z "$nvrtc_dir" ]; then
        for dir in $(ldconfig -p 2>/dev/null | awk '/libnvrtc\.so/ { print $NF }' | xargs -r -n1 dirname) \
                   $(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' ' ') \
                   $(command -v nvcc >/dev/null && d=$(dirname "$(command -v nvcc)")/.. && echo "$d/lib64 $d/lib"); do
            if ls "$dir"/libnvrtc.so* >/dev/null 2>&1; then nvrtc_dir=$dir; break; fi
        done
    fi
    if [ -z "$nvrtc_dir" ] || ! ls "$nvrtc_dir"/libnvrtc.so* >/dev/null 2>&1; then
        echo "run.sh: no libnvrtc found (loader cache, library path, beside nvcc); give --nvrtc-dir DIR, the directory holding libnvrtc.so of a CUDA toolkit" >&2
        exit 2
    fi
    links="$run_dir/nvrtc"
    mkdir -p "$links"
    for lib in "$nvrtc_dir"/libnvrtc*.so*; do ln -sfn "$lib" "$links/$(basename "$lib")"; done
    export LD_LIBRARY_PATH="$links${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$root"
cargo build --release -q -p splinter-minimind
target_dir=$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
exec "$target_dir/release/splinter-minimind" run --run-dir "$run_dir" "${args[@]}"
