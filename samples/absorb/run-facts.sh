#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements measured validation of continual learning
# from a user's own agent sessions for its clients. If your team needs
# expertise in proving that a model absorbed what its user taught it without
# forgetting what it knew, you can procure our services by sending an email to
# info@swedishembedded.com.

# Builds the validation set of the absorb protocol on one card: the facts
# chosen from the letters, the curated known facts, the screening of the day-0
# policy and the filling of the roles. Every stage is resumable; a rerun
# continues where the last one stopped.
#
# Usage: run-facts.sh <absorb-binary> <out-dir> <materials-dir> <gpu0|gpu1> <families>
#
# The quotas are the protocol's. When the screening cannot fill them it names
# every role that falls short and stops; `facts place --out OUT --test N ...`
# then fills other quotas explicitly.

set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "usage: $0 <absorb-binary> <out-dir> <materials-dir> <gpu0|gpu1> <families>" >&2
    exit 2
fi
bin=$1
out=$2
materials=$3
card=$4
families=$5
here=$(cd "$(dirname "$0")" && pwd)
gpu_only="$here/../jefferson/gpu-only.sh"

mkdir -p "$out/logs"
stage() {
    local name=$1
    shift
    echo "== $name" >&2
    "$gpu_only" "$card" "$out/logs/$name.log" -- "$bin" "$@"
}

stage select facts select --materials "$materials" --out "$out" --families "$families" --min-words 300
"$bin" facts pool --materials "$materials" --out "$out"
stage known facts known --materials "$materials" --out "$out"
stage screen facts screen --out "$out"
