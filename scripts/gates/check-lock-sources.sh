#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Cargo.lock pins sven and brain to their remotes, never to a local path.
#
# The committed lock is what a fresh clone builds. A package whose source is
# a file:// URL, or that has no source at all where the manifest names a git
# dependency, is a lock written against one machine's checkout (a `[patch]`
# or a half-finished pin) and resolves nowhere else.
#
# Usage: scripts/gates/check-lock-sources.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

[ -f Cargo.lock ] || { echo "check-lock-sources: no Cargo.lock"; exit 1; }

python3 - <<'PY'
import sys
import tomllib

REMOTES = ("git+https://github.com/swedishembedded/sven", "git+https://github.com/swedishembedded/brain")
manifest = tomllib.load(open("Cargo.toml", "rb"))
git_deps = {
    name for name, spec in manifest["workspace"]["dependencies"].items()
    if isinstance(spec, dict) and "git" in spec
}
lock = tomllib.load(open("Cargo.lock", "rb"))
bad = []
for pkg in lock.get("package", []):
    source = pkg.get("source", "")
    if source.startswith("git+") and not source.startswith(REMOTES):
        bad.append(f"{pkg['name']}: source {source}")
    elif pkg["name"] in git_deps and not source.startswith(REMOTES):
        bad.append(f"{pkg['name']}: {'no source (a local path)' if not source else source}")
if bad:
    print("check-lock-sources: Cargo.lock does not pin sven/brain to their remotes:")
    print("\n".join("  " + b for b in bad))
    print("\nRe-pin with `make lock`; never commit a lock written under a [patch].")
    sys.exit(1)
print("check-lock-sources: OK")
PY
