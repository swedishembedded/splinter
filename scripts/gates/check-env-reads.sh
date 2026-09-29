#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Only splinter-campaign's config module reads or writes the process
# environment.
#
# A setting read deep inside a library is a hidden input: a test cannot set
# it without mutating process state that parallel tests share, and a second
# campaign in the same process cannot differ from the first. Everything
# below the configuration takes its settings as values. Experiments are
# standalone programs and resolve their own inputs.
#
# Usage: scripts/gates/check-env-reads.sh [file ...]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

ALLOWED=crates/campaign/src/config.rs
PATTERN='std::env::(var|var_os|vars|set_var|remove_var)\b'

if [ "$#" -gt 0 ]; then files=("$@"); else mapfile -t files < <(git ls-files --cached --others --exclude-standard 'crates/*.rs'); fi

scan=()
for f in "${files[@]}"; do
    case "$f" in
    "$ALLOWED") ;;
    crates/*.rs) [ -f "$f" ] && scan+=("$f") ;;
    esac
done
[ "${#scan[@]}" -eq 0 ] && exit 0

hits=$(grep -nE "$PATTERN" "${scan[@]}" 2>/dev/null)
if [ -n "$hits" ]; then
    echo "check-env-reads: the environment is read outside $ALLOWED:"
    echo "$hits" | sed 's/^/  /'
    echo
    echo "Add the setting to Config and pass it down as a value."
    exit 1
fi
[ "$#" -eq 0 ] && echo "check-env-reads: OK"
exit 0
