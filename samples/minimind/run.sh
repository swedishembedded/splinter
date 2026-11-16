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
# usage: samples/minimind/run.sh [--profile smoke|default|64m] [--run-dir DIR] [options]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
profile=smoke
run_dir=
args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile) profile=$2; args+=("$1" "$2"); shift 2 ;;
        --run-dir) run_dir=$2; shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
run_dir=${run_dir:-$root/target/minimind/$profile}
cd "$root"
cargo build --release -q -p splinter-minimind
target_dir=$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
exec "$target_dir/release/splinter-minimind" run --run-dir "$run_dir" "${args[@]}"
