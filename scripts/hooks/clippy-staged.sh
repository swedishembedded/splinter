#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Clippy, warnings denied, on exactly the workspace packages that own the
# given .rs files (the pre-commit hook passes the staged ones).
#
# Package names come from `cargo metadata`, not from the directory name: a
# path-to-name guess silently lints the wrong package, or none, the day a
# directory and its package stop sharing a name.
#
# Usage: scripts/hooks/clippy-staged.sh <file.rs> ...
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ "$#" -eq 0 ] && exit 0

mapfile -t packages < <(
    cargo metadata --no-deps --offline --format-version 1 | python3 -c '
import json, os, sys
meta = json.load(sys.stdin)
root = meta["workspace_root"]
dirs = sorted(
    ((os.path.relpath(os.path.dirname(p["manifest_path"]), root), p["name"]) for p in meta["packages"]),
    key=lambda d: -len(d[0]),
)
owners = set()
for f in sys.argv[1:]:
    for d, name in dirs:
        if d == "." or f == d or f.startswith(d + "/"):
            owners.add(name)
            break
print("\n".join(sorted(owners)))
' "$@"
)
[ "${#packages[@]}" -eq 0 ] && exit 0

args=()
for p in "${packages[@]}"; do args+=(-p "$p"); done
exec cargo clippy --release --all-targets "${args[@]}" -- -D warnings
