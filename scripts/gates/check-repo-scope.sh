#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Splinter names only the projects it depends on: sven and brain.
#
# A tracked file that names any other project (an orchestrator, a host
# application) couples Splinter's text to that project's plans, which
# nothing here keeps current. Describe it generically instead. The Repository
# scope section of AGENTS.md states the whole rule.
#
# Usage: scripts/gates/check-repo-scope.sh [file ...]
#   No arguments scans every tracked file; with arguments, only those.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

OUT_OF_SCOPE='\bwhale\b'

if [ "$#" -gt 0 ]; then
    files=("$@")
else
    mapfile -d '' -t files < <(git ls-files -z)
fi

scan=()
for f in "${files[@]}"; do
    [ -f "$f" ] || continue
    [ "$f" = scripts/gates/check-repo-scope.sh ] && continue
    scan+=("$f")
done

hits=""
if [ "${#scan[@]}" -gt 0 ]; then
    hits=$(printf '%s\0' "${scan[@]}" | xargs -0 grep -IHniE "$OUT_OF_SCOPE" 2>/dev/null)
fi
if [ -z "$hits" ]; then
    [ "$#" -eq 0 ] && echo "check-repo-scope: OK"
    exit 0
fi
echo "check-repo-scope: a project Splinter does not depend on is named:"
echo "$hits" | sed 's/^/  /'
echo
echo "Describe it generically (an orchestrator, a host application)."
exit 1
