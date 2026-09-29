#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Every script this repository carries works and is used.
#
#   1. Syntax: every .sh parses (`bash -n`), every .py compiles
#      (`py_compile`, which also catches an import-time typo).
#   2. Orphans: every script under scripts/ is named somewhere else in the
#      repository - a Makefile target, a hook definition, another script. A
#      script nothing references is one nobody remembers the purpose of,
#      and nothing notices when it stops working.
#
# Absolute machine paths are check-no-machine-paths.sh's business. Adapted
# from the gate the brain repository runs.
#
# Usage: scripts/gates/check-scripts.sh
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1

fail=0
pycache=$(mktemp -d)
trap 'rm -rf "$pycache"' EXIT

while IFS= read -r -d '' f; do
    bash -n "$f" || { echo "check-scripts: syntax error in $f"; fail=1; }
done < <(git ls-files -z 'scripts/*.sh')
while IFS= read -r -d '' f; do
    PYTHONPYCACHEPREFIX="$pycache" python3 -m py_compile "$f" ||
        { echo "check-scripts: $f does not compile"; fail=1; }
done < <(git ls-files -z 'scripts/*.py' 'scripts/hooks/commit-msg' 'scripts/hooks/pre-push')

while IFS= read -r -d '' f; do
    base=$(basename "$f")
    # A Python module is referenced by its import name, not its file name.
    name=${base%.py}
    if ! git grep -qE -- "$base|(from|import) $name\b" -- . ":(exclude)$f"; then
        echo "check-scripts: $f is not named anywhere else in the repository"
        fail=1
    fi
done < <(git ls-files -z 'scripts/*')

[ "$fail" -eq 0 ] && echo "check-scripts: OK"
exit "$fail"
