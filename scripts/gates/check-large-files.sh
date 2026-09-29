#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Keep model artifacts and oversized files out of git.
#
# Splinter produces adapters, checkpoints, datasets and traces by the
# gigabyte. None of it is source: it is regenerable, it differs byte-for-byte
# on every run, and once committed it can only be removed by rewriting
# history. Weights, tensor dumps, media and source documents are refused at
# any size (`git add -f` walks straight past .gitignore); everything else has
# a size ceiling.
#
# Usage: scripts/gates/check-large-files.sh [file ...]
#   No arguments scans every tracked file; with arguments, only those.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

BANNED_EXT='mp4|mkv|webm|mov|avi|safetensors|gguf|ckpt|pth|pt|npy|npz|onnx|h5|tflite|pb|msgpack|bin|f32|f16|u32|i32|raw|dat|pdf'
MAX_BYTES=$((1024 * 1024))

if [ "$#" -gt 0 ]; then files=("$@"); else mapfile -t files < <(git ls-files); fi

fail=0
for f in "${files[@]}"; do
    [ -f "$f" ] || continue
    if [[ "$f" =~ \.(${BANNED_EXT})$ ]]; then
        echo "BANNED FILE TYPE: $f"
        echo "    Weights, tensor dumps, media and source documents are never committed."
        echo "    Keep them in the Splinter state directory and reference them by digest."
        fail=1
        continue
    fi
    sz=$(wc -c <"$f")
    if [ "$sz" -gt "$MAX_BYTES" ]; then
        echo "FILE TOO LARGE: $f ($((sz / 1024)) KiB, ceiling $((MAX_BYTES / 1024)) KiB)"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo
    echo "check-large-files: an exception is a deliberate edit to this script, not a flag."
    exit 1
fi
[ "$#" -eq 0 ] && echo "check-large-files: OK (${#files[@]} tracked files)"
exit 0
