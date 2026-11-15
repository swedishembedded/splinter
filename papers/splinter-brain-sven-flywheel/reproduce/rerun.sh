#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements self-training runs that a third party can repeat
# from public inputs, for its clients. If your team needs expertise in re-running
# a model's self-improvement from its base, you can procure our services by
# sending an email to info@swedishembedded.com.

# Re-runs the four S4 lineages from the base with the launch record's settings and seeds
# (prereg/rsi4-launch.md): three generations each, one process per generation, frontier targets
# 300/600/1200 with request caps 3000/6000/12000. Lineages run one after another on DEVICE.
#
#   rerun.sh LINEAGE_BIN BRAIN_BIN MODELS_DIR MANUAL_DIR LUA_BIN OUT DEVICE
#
# OUT/lineages/<series>/ holds each lineage's records, OUT/handoff/candidates/<series>/<step>/ its
# hand-off and OUT/adapters/<series>/<step>/ its champions, in the layout the reference examiner
# reads (its --evidence OUT --series .).
set -euo pipefail
[ "$#" -eq 7 ] || { echo "usage: rerun.sh LINEAGE_BIN BRAIN_BIN MODELS_DIR MANUAL_DIR LUA_BIN OUT DEVICE" >&2; exit 2; }
LINEAGE=$1 BRAIN=$2 MODELS=$3 MANUAL=$4 LUA=$5 OUT=$6 DEVICE=$7

SEEDS=(3217799999916437381 4785861403145693048 4743887585493406067 6379732546301056828)
GENERATOR_SEED=8599218772143256343
TARGETS=(300 600 1200)
CAPS=(3000 6000 12000)

mkdir -p "$OUT/lineages" "$OUT/adapters" "$OUT/handoff" "$OUT/state"
for i in 0 1 2 3; do
    s=r$i
    for g in 1 2 3; do
        "$LINEAGE" \
            --server-device "$DEVICE" --model "$MODELS/Qwen/Qwen3-4B" --base-name Qwen/Qwen3-4B \
            --models-dir "$MODELS" --brain "$BRAIN" \
            --out "$OUT/lineages/$s" --lineage "$s" --handoff "$OUT/handoff" --kept-adapters "$OUT/adapters" \
            --proposer-state "$OUT/state/$s" \
            --manual "$MANUAL/lua51-manual-5.4.html" --manual "$MANUAL/lua51-manual-5.5.html" \
            --manual "$MANUAL/lua51-manual-5.6.html" \
            --lua "$LUA" --seed "${SEEDS[$i]}" --generator-seed "$GENERATOR_SEED" \
            --generations 3 --rounds 1 --run-generations 1 --slots 64 --dev-generator 100 --keep-generator 150 \
            --dev-proposals 300 --keep-proposals 150 --held-out-calls 1400 --proposals-per-round 300 \
            --proposal-calls 800 --proposal-parallel 64 --target-effect 0.08 --discordance 0.2 \
            --retention-margin 0.05 --steps 40 --lr 3e-5 --rank 16 --records-per-step 8 --replay-share 0 \
            --correct-share 0 --min-free-inodes 300 --train-from base-cumulative --cumulative-epochs 2 \
            --frontier-samples 4 --frontier-temperature 1.0 --frontier-target "${TARGETS[$((g - 1))]}" \
            --frontier-request-cap "${CAPS[$((g - 1))]}" \
            --splinter-commit 86f13672ab8dddfaf1c7785978e567ecbcfb7a01 \
            --sven-commit fc99a2f1aa414701a323d164a7c8c6b518f887fa \
            --brain-commit 796871647e854a68089631c98842c027cd5c9c7a \
            >>"$OUT/lineages/$s.out" 2>>"$OUT/lineages/$s.err" ||
            { echo "rerun: lineage $s generation $g failed; see $OUT/lineages/$s.err" >&2; exit 1; }
    done
done
