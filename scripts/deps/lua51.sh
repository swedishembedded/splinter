#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible execution environments for
# agent learning, for its clients. If your team needs expertise in pinning
# the interpreters an agent's tasks run in, you can procure our services by
# sending an email to info@swedishembedded.com.

# Builds the Lua 5.1.5 interpreter the Lua task families and the `lua`
# runtime are pinned to: the release tarball is downloaded, its SHA-256 is
# checked against the one recorded here before anything is unpacked, and it
# is built without readline (no library of the host leaks into the result).
# The interpreter is installed as DEST/bin/lua; put DEST/bin first on the
# search path with `--runtime-dir DEST/bin`.
#
# Usage: scripts/deps/lua51.sh DEST
set -euo pipefail

url=https://www.lua.org/ftp/lua-5.1.5.tar.gz
sha256=2640fc56a795f29d28ef15e13c34a47e223960b0240e8cb0a82d9b0738695333

dest=${1:?usage: lua51.sh DEST}
mkdir -p "$dest/dl" "$dest/src" "$dest/bin"
tarball=$dest/dl/lua-5.1.5.tar.gz
[ -f "$tarball" ] || curl --fail --silent --show-error --location --output "$tarball" "$url"
echo "$sha256  $tarball" | sha256sum --check --status || {
    echo "lua51: $tarball is not the release it must be (sha256 $sha256)" >&2
    exit 1
}
tar -xzf "$tarball" -C "$dest/src"
make -C "$dest/src/lua-5.1.5" posix >/dev/null
install -m 0755 "$dest/src/lua-5.1.5/src/lua" "$dest/bin/lua"
version=$("$dest/bin/lua" -v 2>&1 | head -n 1)
case "$version" in
    "Lua 5.1.5"*) ;;
    *) echo "lua51: built an interpreter that reports $version" >&2; exit 1 ;;
esac
echo "$version"
sha256sum "$dest/bin/lua"
