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
# policy, the filling of the roles, the canary, the sealed probes and the
# day-0 baseline of the sycophancy and unknown probes. Every stage is
# resumable; a rerun continues where the last one stopped.
#
# Usage: run-facts.sh <absorb-binary> <out-dir> <materials-dir> <gpu0|gpu1> <families> [place-flag...]
#
# The quotas are the protocol's. When the screening cannot fill them it names
# every role that falls short and stops; run again with the quotas to place
# instead as flags (`--neighbour-known 61`): they are passed to `facts screen`
# and recorded in the manifest, never chosen silently. The canary is the first
# test fact of day 1 in the manifest's order.

set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "usage: $0 <absorb-binary> <out-dir> <materials-dir> <gpu0|gpu1> <families> [place-flag...]" >&2
    exit 2
fi
bin=$1
out=$2
materials=$3
card=$4
families=$5
shift 5
here=$(cd "$(dirname "$0")" && pwd)
gpu_only="$here/../jefferson/gpu-only.sh"

mkdir -p "$out/logs"
stage() {
    local name=$1
    shift
    echo "== $name" >&2
    "$gpu_only" "$card" "$out/logs/$name.log" -- "$bin" "$@"
}

# A selection is made under one set of rules: a rerun takes the batch and the
# tries the ledger records.
rules=()
if [ -f "$out/select/params.json" ]; then
    read -r batch tries < <(python3 -c 'import json, sys; p = json.load(open(sys.argv[1])); print(p["batch"], p["tries"])' "$out/select/params.json")
    rules=(--batch "$batch" --tries "$tries")
fi
stage select facts select --materials "$materials" --out "$out" --families "$families" --min-words 300 "${rules[@]}"
"$bin" facts pool --materials "$materials" --out "$out"
stage known facts known --materials "$materials" --out "$out"
stage screen facts screen --out "$out" "$@"
canary=$(python3 - "$out/manifest.json" <<'PY'
import json, sys
facts = json.load(open(sys.argv[1]))["facts"]
print(next(f["id"] for f in facts if f.get("role") == "test" and f.get("day") == 1))
PY
)
"$bin" facts canary --out "$out" --fact "$canary"
stage seal probes seal --out "$out"
stage baseline probes baseline --out "$out"
