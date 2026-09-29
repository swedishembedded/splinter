#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Write the gitignored .cargo/config.toml that builds Splinter against local
# sven and brain checkouts instead of their remotes.
#
# Two cargo mechanisms, each doing the part the other cannot:
#
#   * source replacement redirects each remote git source to the local
#     repository, so resolving the workspace never reaches the network. It
#     reads committed history only, and it needs Cargo.lock to exist: the
#     lock pins the revision it checks out.
#   * `paths` substitutes the working trees for the packages that resolution
#     found, so uncommitted edits in sven or brain are what gets built.
#
# `[patch]` would be simpler to write but does not work here: cargo still
# resolves the original remote source before applying a patch, so a patched
# build fails offline, and it rewrites Cargo.lock to path sources. Neither
# mechanism used here touches Cargo.lock.
#
# Usage: scripts/local-config.sh <sven-dir> <brain-dir>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SVEN_URL=https://github.com/swedishembedded/sven
BRAIN_URL=https://github.com/swedishembedded/brain

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <sven-dir> <brain-dir>" >&2
    exit 2
fi

checkout() {
    local dir
    dir=$(cd "$1" 2>/dev/null && pwd) || { echo "local-config: no such directory: $1" >&2; exit 1; }
    git -C "$dir" rev-parse --git-dir >/dev/null 2>&1 || { echo "local-config: not a git repository: $dir" >&2; exit 1; }
    [ -f "$dir/$2" ] || { echo "local-config: $dir has no $2 - is it the right checkout?" >&2; exit 1; }
    echo "$dir"
}
sven=$(checkout "$1" crates/sdk/Cargo.toml)
brain=$(checkout "$2" crates/sdk/Cargo.toml)

mkdir -p .cargo
{
    echo "# Written by scripts/local-config.sh (make local). Gitignored: it names"
    echo "# this machine's checkouts. Delete it to build against the remotes."
    echo
    echo "paths = [\"$sven\", \"$brain\"]"
    echo
    echo "[source.sven-remote]"
    echo "git = \"$SVEN_URL\""
    echo "replace-with = \"sven-local\""
    echo
    echo "[source.sven-local]"
    echo "git = \"file://$sven\""
    echo
    echo "[source.brain-remote]"
    echo "git = \"$BRAIN_URL\""
    echo "replace-with = \"brain-local\""
    echo
    echo "[source.brain-local]"
    echo "git = \"file://$brain\""
} >.cargo/config.toml

echo "local-config: wrote .cargo/config.toml"
echo "  sven:  $sven"
echo "  brain: $brain"
exit 0
