#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements counterfactual language environments for
# agent learning, for its clients. If your team needs expertise in building
# controlled interpreter variants with matching reference manuals, you can
# procure our services by sending an email to info@swedishembedded.com.

# Builds the two counterfactual Lua 5.1.5 dialects (zero-index, plus-concat)
# from the pinned release tarball: the tarball's SHA-256 is checked before
# anything is unpacked, each dialect is a patch from lua51-dialects/ applied
# to a fresh tree, and the result is built like lua51.sh builds stock Lua
# (no readline). For every dialect it installs
#   DEST/NAME/bin/lua   DEST/NAME/manual/*.html   DEST/NAME/DIALECT.md
# and prints the sha256 of each installed interpreter. Running it again
# rebuilds everything from scratch. The dialects' definitions are
# lua51-dialects/zero-index.md and lua51-dialects/plus-concat.md.
#
# Usage: scripts/deps/lua51-dialects.sh DEST [TARBALL [MANUAL]]
#   TARBALL defaults to DEST/dl/lua-5.1.5.tar.gz (downloaded when absent).
#   MANUAL, the stock manual manual.py rewrites, defaults to
#   DEST/dl/manual-5.1.html (downloaded when absent, sha256 pinned).
set -euo pipefail

url=https://www.lua.org/ftp/lua-5.1.5.tar.gz
manual_url=https://www.lua.org/manual/5.1/manual.html
manual_sha256=e754f09a38a45af243b80740ffc66c5147548a731a4239165b06586f2f54c28a
sha256=2640fc56a795f29d28ef15e13c34a47e223960b0240e8cb0a82d9b0738695333
dialects="zero-index plus-concat"

dest=${1:?usage: lua51-dialects.sh DEST [TARBALL]}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$dest/dl"
tarball=${2:-$dest/dl/lua-5.1.5.tar.gz}
manual=${3:-$dest/dl/manual-5.1.html}

[ -f "$tarball" ] || curl --fail --silent --show-error --location --output "$tarball" "$url"
echo "$sha256  $tarball" | sha256sum --check --status || {
    echo "lua51-dialects: $tarball is not the release it must be (sha256 $sha256)" >&2
    exit 1
}
[ -f "$manual" ] || curl --fail --silent --show-error --location --output "$manual" "$manual_url"
echo "$manual_sha256  $manual" | sha256sum --check --status || {
    echo "lua51-dialects: $manual is not the stock manual it must be (sha256 $manual_sha256)" >&2
    exit 1
}

work=$(mktemp -d "$dest/.build.XXXXXX")
trap 'rm -rf "$work"' EXIT

for name in $dialects; do
    tree=$work/$name
    mkdir -p "$tree"
    tar -xzf "$tarball" -C "$tree" --no-same-owner
    # patch(1) re-applies the owner group of a file; on setgid project
    # directories that group is not one it may assign, so use our own.
    chgrp -R "$(id -g)" "$tree" 2>/dev/null || true
    (cd "$tree/lua-5.1.5" && patch -p1 --quiet <"$here/lua51-dialects/$name.patch")
    make -C "$tree/lua-5.1.5" posix >"$work/$name.log" 2>&1 || {
        cat "$work/$name.log" >&2
        echo "lua51-dialects: build of $name failed" >&2
        exit 1
    }
    version=$("$tree/lua-5.1.5/src/lua" -v 2>&1 | head -n 1)
    case "$version" in
        "Lua 5.1.5-$name  Copyright"*) ;;
        *) echo "lua51-dialects: $name reports $version" >&2; exit 1 ;;
    esac
    rm -rf "$dest/$name"
    mkdir -p "$dest/$name/bin" "$dest/$name/manual"
    install -m 0755 "$tree/lua-5.1.5/src/lua" "$dest/$name/bin/lua"
    python3 -I "$here/lua51-dialects/manual.py" "$manual" "$name" "$dest/$name/manual"
    install -m 0644 "$here/lua51-dialects/$name.md" "$dest/$name/DIALECT.md"
    echo "$version"
    sha256sum "$dest/$name/bin/lua"
done
