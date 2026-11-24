#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# No absolute machine paths in the repository.
#
# /data, /home, /opt, /mnt and /root name one machine's layout: a path like
# that resolves on the machine it was written on and nowhere else, and a
# check that silently skips when the path is missing is green everywhere
# else. So does a home-relative project or code tree (~/project/...,
# ~/code/...): one user's layout, which a location must be configured as (a
# flag, a variable, brain's own store), never assumed. The history records
# (.agents/, papers/) say where past evidence was kept on the machine it ran
# on, so that layout is refused outside them.
# /tmp is refused in .rs files only: a test writing to a fixed global path
# collides with a concurrent run and leaves the file behind on failure.
#
# The local-override cargo config (`make deps/local`) legitimately holds absolute
# paths; it is gitignored, and this gate scans tracked plus untracked-but-
# unignored files only.
#
# Usage:
#   scripts/gates/check-no-machine-paths.sh             # the whole tree
#   scripts/gates/check-no-machine-paths.sh <file> ...  # these (pre-commit)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

ROOTS='(^|[^A-Za-z0-9_./~-])/(data|home|opt|mnt|root)/'
TMP='(^|[^A-Za-z0-9_./~-])/tmp/'
LAYOUT='(~|\$HOME|\$\{HOME\}|\$\(HOME\))/(project|code)/'

if [ "$#" -gt 0 ]; then
    files=("$@")
else
    mapfile -d '' -t files < <(git ls-files -z && git ls-files -z --others --exclude-standard)
fi

root_files=()
rs_files=()
layout_files=()
for f in "${files[@]}"; do
    [ -f "$f" ] || continue
    [ "$f" = scripts/gates/check-no-machine-paths.sh ] && continue
    root_files+=("$f")
    case "$f" in *.rs) rs_files+=("$f") ;; esac
    case "$f" in .agents/* | papers/*) ;; *) layout_files+=("$f") ;; esac
done

hits=""
if [ "${#root_files[@]}" -gt 0 ]; then
    hits=$(printf '%s\0' "${root_files[@]}" | xargs -0 grep -IHnE "$ROOTS" 2>/dev/null)
fi
if [ "${#layout_files[@]}" -gt 0 ]; then
    layout_hits=$(printf '%s\0' "${layout_files[@]}" | xargs -0 grep -IHnE "$LAYOUT" 2>/dev/null)
    hits=$(printf '%s\n%s' "$hits" "$layout_hits" | sed '/^$/d' | sort -u)
fi
if [ "${#rs_files[@]}" -gt 0 ]; then
    tmp_hits=$(printf '%s\0' "${rs_files[@]}" | xargs -0 grep -IHnE "$TMP" 2>/dev/null)
    hits=$(printf '%s\n%s' "$hits" "$tmp_hits" | sed '/^$/d' | sort -u)
fi

if [ -z "$hits" ]; then
    [ "$#" -eq 0 ] && echo "check-no-machine-paths: OK"
    exit 0
fi
echo "check-no-machine-paths: absolute machine path baked in:"
echo "$hits" | sed 's/^/  /'
cat <<'MSG'
Resolve the path from the environment instead - an env var, a CLI flag,
`std::env::temp_dir()`, a `TempDir` - or, in prose, write a placeholder such
as <brain-checkout> that the reader substitutes.
MSG
exit 1
