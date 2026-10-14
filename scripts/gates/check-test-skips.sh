#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# No test skips itself silently.
#
# A test that returns early because a prerequisite is missing (no git, no
# docker daemon, no weights, running as root) is reported as passed although
# it checked nothing: a false success. A test that cannot run must fail with
# the reason, or carry `#[ignore = "reason"]` so that every run shows it as
# ignored. This gate refuses the usual shapes of a silent skip in Rust test
# code: a message about skipping printed from a test, and the skip helpers
# other projects use.
#
# Denied in test code - any tracked .rs file under a `tests/` directory, and
# the part of a source file from its `mod tests {` on - : `SKIPPED` or
# `skipping` / `skipped` in a print macro, and a call to a function named
# `skip` through `testutil`.
#
# Usage: scripts/gates/check-test-skips.sh [file ...]

set -u
cd "$(git rev-parse --show-toplevel)" || exit 1

if [ "$#" -gt 0 ]; then
    files=$(printf '%s\n' "$@" | grep -E '\.rs$' | grep -v '^scripts/gates/' || true)
else
    files=$(git ls-files '*.rs' | grep -v '^scripts/gates/' || true)
fi

fail=0
for f in $files; do
    [ -f "$f" ] || continue
    case "$f" in
        */tests/*) scope=$(cat "$f") ;;
        *) scope=$(awk '/^[[:space:]]*mod tests[[:space:]]*\{/ {on=1} on' "$f") ;;
    esac
    hits=$(printf '%s\n' "$scope" | grep -nE 'e?print(ln)?!\(.*(SKIPPED|[Ss]kipp(ed|ing))|testutil::skip\(' || true)
    if [ -n "$hits" ]; then
        while IFS= read -r line; do
            echo "SILENT SKIP: $f:$line"
        done <<<"$hits"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    cat <<'MSG'

A test that cannot run must fail with the reason, or be marked
#[ignore = "reason"] so that every run shows it as ignored. It may not print a
message and return, which reports a pass for a check that never happened.
MSG
    exit 1
fi
echo "check-test-skips: OK"
