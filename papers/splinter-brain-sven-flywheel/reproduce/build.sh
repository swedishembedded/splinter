#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements builds of research software at the exact commits
# a result was produced with, for its clients. If your team needs expertise in
# rebuilding a measured system from its pinned history, you can procure our
# services by sending an email to info@swedishembedded.com.

# Builds the binaries the paper's S4 runs used, at the commits its launch record pins, from clones of
# the three repositories' remotes. Each commit is checked out as a detached worktree under WORK, so
# the clones' own checkouts are not touched.
#
#   build.sh brain   BRAIN_SRC WORK        -> WORK/bin/brain-796871647
#   build.sh lineage SPLINTER_SRC SVEN_SRC BRAIN_SRC WORK -> WORK/bin/lineage-86f13672a
#
# A build is not bit-reproducible across toolchains and hosts, so the binary's SHA-256 is printed
# beside the launch record's and a difference is reported, not refused.
set -euo pipefail

BRAIN_COMMIT=796871647e854a68089631c98842c027cd5c9c7a
BRAIN_SHA256=a020a8eba00cb699fb1c901ecf1a8f63a2dc82d4ce14658973258e3fbeb3ad26
SVEN_COMMIT=fc99a2f1aa414701a323d164a7c8c6b518f887fa
SPLINTER_COMMIT=86f13672ab8dddfaf1c7785978e567ecbcfb7a01
LINEAGE_SHA256=ea233cc6db7c16b54ddd321688f6835bc17c770bfb6a08fd42e0254b63a3ef5d

# worktree SRC COMMIT DEST: a detached worktree of SRC at COMMIT, fetched from SRC's origin if absent.
worktree() {
    local src=$1 commit=$2 dest=$3
    if [ -d "$dest" ]; then
        [ "$(git -C "$dest" rev-parse HEAD)" = "$commit" ] || { echo "build: $dest is not at $commit" >&2; exit 1; }
        return
    fi
    git -C "$src" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$src" fetch origin
    git -C "$src" worktree add --detach "$dest" "$commit"
}

report() {
    local bin=$1 want=$2 got
    got=$(sha256sum "$bin" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
        echo "build: $bin is byte-identical to the launch record's binary ($want)"
    else
        echo "build: $bin sha256 $got; the launch record's binary was $want (a rebuild on another host differs)"
    fi
}

brain() {
    local src=$1 work=$2
    worktree "$src" "$BRAIN_COMMIT" "$work/src/brain"
    (cd "$work/src/brain" && cargo build --release --locked -p brain-cli --bin brain)
    mkdir -p "$work/bin"
    install -m 0755 "$work/src/brain/target/release/brain" "$work/bin/brain-${BRAIN_COMMIT:0:9}"
    report "$work/bin/brain-${BRAIN_COMMIT:0:9}" "$BRAIN_SHA256"
}

lineage() {
    local splinter=$1 sven=$2 brain_src=$3 work=$4
    worktree "$splinter" "$SPLINTER_COMMIT" "$work/src/splinter"
    worktree "$sven" "$SVEN_COMMIT" "$work/src/sven"
    worktree "$brain_src" "$BRAIN_COMMIT" "$work/src/brain"
    # The lineage driver's lock names the three repositories' public remotes; the pinned sven and
    # brain commits are built from the local worktrees instead (splinter's own deps/local mechanism).
    (cd "$work/src/splinter" && bash scripts/local-config.sh "$work/src/sven" "$work/src/brain" &&
        cargo build --release -p splinter-campaign-lua --bin lineage)
    mkdir -p "$work/bin"
    install -m 0755 "$work/src/splinter/target/release/lineage" "$work/bin/lineage-${SPLINTER_COMMIT:0:9}"
    report "$work/bin/lineage-${SPLINTER_COMMIT:0:9}" "$LINEAGE_SHA256"
}

case "${1:-}" in
    brain) [ "$#" -eq 3 ] || { echo "usage: build.sh brain BRAIN_SRC WORK" >&2; exit 2; }; brain "$2" "$3" ;;
    lineage) [ "$#" -eq 5 ] || { echo "usage: build.sh lineage SPLINTER_SRC SVEN_SRC BRAIN_SRC WORK" >&2; exit 2; }
        lineage "$2" "$3" "$4" "$5" ;;
    *) echo "usage: build.sh brain|lineage ..." >&2; exit 2 ;;
esac
