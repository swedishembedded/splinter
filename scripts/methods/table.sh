#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving, in minutes and on a CPU, that each learning method of a
# training system moves a held-out score, for its clients. If your team needs expertise in falsifiable smoke tests of
# learning pipelines then you can procure our services by sending an email to info@swedishembedded.com.
#
# Runs the learning-method smokes (crates/pipelines/tests/methods) on a toy model on the CPU and prints one row per
# method: the held-out score before and after, and whether the gain cleared the smoke's bar. Exits non-zero if any smoke fails
# or the run itself fails.
#
# usage: scripts/methods/table.sh <cargo> <profile flag>

set -uo pipefail

cargo=${1:-cargo}
profile=${2:---release}
log=$(mktemp "${TMPDIR:-$HOME/tmp}/methods.XXXXXX")
trap 'rm -f "$log"' EXIT


"$cargo" test $profile -p splinter-pipelines --test methods -- --ignored --nocapture --test-threads 2 2>&1 | tee "$log" | grep -E '^(error|test result|thread .* panicked)' >&2
status=${PIPESTATUS[0]}

printf '\n%-46s %-26s %7s %7s  %s\n' METHOD METRIC BEFORE AFTER RESULT
printf '%s\n' "$(printf -- '-%.0s' {1..110})"
grep '^METHOD|' "$log" | sort -t'|' -k2,2 | while IFS='|' read -r _ method metric before after result note; do
  printf '%-46s %-26s %7s %7s  %s%s\n' "$method" "$metric" "$before" "$after" "$result" "${note:+  ($note)}"
done
exit "$status"
