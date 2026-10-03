#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Code never cites a docs/ or .agents/ file path.
#
# docs/ is user-facing documentation and .agents/ holds contributor rules,
# knowledge and roadmaps; both get rewritten and reorganised. A comment or
# string in crates/, samples/ or scripts/ that names one of their paths is
# a cross-reference nothing keeps in sync, so it silently rots. State the fact
# or the reasoning inline instead. scripts/gates/ and scripts/hooks/ are
# exempt: a gate names the paths it validates, and fails loudly when they move.
#
# Usage:
#   scripts/gates/check-no-doc-citations.sh             # the whole tree
#   scripts/gates/check-no-doc-citations.sh <file> ...  # these (pre-commit)
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1

PATTERN='\.agents/|docs/[A-Za-z0-9_./-]+\.md'
SCANNED=(crates samples scripts)

if [ "$#" -gt 0 ]; then
    files=()
    for f in "$@"; do
        case "$f" in
        scripts/gates/* | scripts/hooks/*) continue ;;
        crates/* | samples/* | scripts/*) files+=("$f") ;;
        esac
    done
    [ "${#files[@]}" -eq 0 ] && exit 0
    hits=$(grep -nE "$PATTERN" "${files[@]}" 2>/dev/null)
else
    hits=$(grep -rnE "$PATTERN" "${SCANNED[@]}" 2>/dev/null | grep -vE '^scripts/(gates|hooks)/')
fi

if [ -n "$hits" ]; then
    echo "check-no-doc-citations: code cites a docs/ or .agents/ file path:"
    echo "$hits" | sed 's/^/  /'
    echo
    echo "Remove the citation and state the fact or reasoning inline instead."
    exit 1
fi
[ "$#" -eq 0 ] && echo "check-no-doc-citations: OK"
exit 0
