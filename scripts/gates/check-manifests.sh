#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Every Cargo manifest parses, and the failure names the file that does not.
#
# One unparseable member manifest (most often an unresolved conflict marker)
# makes the whole workspace unresolvable, and cargo's own error names
# whichever member it happened to be resolving rather than the broken file.
#
# Usage: scripts/gates/check-manifests.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

python3 - <<'PY'
import glob
import sys
import tomllib

manifests = sorted(glob.glob("Cargo.toml") + glob.glob("crates/*/Cargo.toml") + glob.glob("samples/*/Cargo.toml") + glob.glob("samples/*/*/Cargo.toml"))
bad = []
for path in manifests:
    try:
        with open(path, "rb") as fh:
            tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        bad.append(f"{path}: {exc}")
if bad:
    print("check-manifests: a manifest does not parse:")
    print("\n".join("  " + b for b in bad))
    sys.exit(1)
print(f"check-manifests: OK ({len(manifests)} manifests)")
PY
