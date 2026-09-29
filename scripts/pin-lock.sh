#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Pin Cargo.lock's sven and brain revisions to the HEADs of local checkouts,
# without contacting either remote.
#
# The manifest's git URLs are pointed at the local repositories for one
# resolve, then the lock's sources are rewritten back to the remote URLs. A
# commit id is the same object in every clone, so the result is the lock a
# resolve against the remotes would write - provided those commits are
# pushed there, which is the maintainer's step, not this script's.
#
# Only the sven and brain entries move: every other package keeps the
# version already in Cargo.lock.
#
# Usage: scripts/pin-lock.sh <sven-dir> <brain-dir>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SVEN_URL=https://github.com/swedishembedded/sven
BRAIN_URL=https://github.com/swedishembedded/brain

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <sven-dir> <brain-dir>" >&2
    exit 2
fi
sven=$(cd "$1" && pwd)
brain=$(cd "$2" && pwd)
for repo in "$sven" "$brain"; do
    if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
        echo "pin-lock: $repo has uncommitted changes; the lock can only pin a commit" >&2
        exit 1
    fi
done

backup=$(mktemp -d)
cp Cargo.toml "$backup/Cargo.toml"
[ -f .cargo/config.toml ] && mv .cargo/config.toml "$backup/config.toml"
restore() {
    cp "$backup/Cargo.toml" Cargo.toml
    [ -f "$backup/config.toml" ] && mv "$backup/config.toml" .cargo/config.toml
    rm -rf "$backup"
}
trap restore EXIT

sed -i -e "s#$SVEN_URL\"#file://$sven\"#g" -e "s#$BRAIN_URL\"#file://$brain\"#g" Cargo.toml
cargo metadata --format-version 1 >/dev/null
sed -i -e "s#git+file://$sven\\([?\#]\\)#git+$SVEN_URL\\1#g" -e "s#git+file://$brain\\([?\#]\\)#git+$BRAIN_URL\\1#g" Cargo.lock

echo "pin-lock: sven  $(git -C "$sven" rev-parse --short HEAD)"
echo "pin-lock: brain $(git -C "$brain" rev-parse --short HEAD)"
