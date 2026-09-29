#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Enforces architecture.toml against the real dependency graph.
#
#   1. Every workspace package is listed, and depends only on the Splinter
#      crates its entry allows - normal, build and dev dependencies alike.
#   2. Only a package marked `brain = true` depends on any brain crate.
#   3. No tracked .rs file exceeds `max_file_lines`.
#
# The graph comes from `cargo metadata --no-deps`, the manifests cargo
# itself resolves, not from a text scan that a renamed dependency would
# slip past.
#
# Usage: scripts/gates/check-architecture.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

cargo metadata --no-deps --offline --format-version 1 2>/dev/null | python3 -c '
import json, subprocess, sys, tomllib

arch = tomllib.load(open("architecture.toml", "rb"))
allowed = arch["crates"]
meta = json.load(sys.stdin)
members = {p["name"] for p in meta["packages"]}
failures = []
for pkg in meta["packages"]:
    name = pkg["name"]
    rule = allowed.get(name)
    if rule is None:
        failures.append(f"{name}: not listed in architecture.toml")
        continue
    for dep in pkg["dependencies"]:
        target = dep["name"]
        if target in members and target not in rule.get("internal", []):
            failures.append(f"{name} -> {target}: not an allowed edge")
        if (target == "brain" or target.startswith("brain-")) and not rule.get("brain", False):
            failures.append(f"{name} -> {target}: only a crate marked brain = true may depend on brain")
for listed in allowed:
    if listed not in members:
        failures.append(f"{listed}: listed in architecture.toml but not a workspace package")

limit = arch["limits"]["max_file_lines"]
files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.rs"],
                       capture_output=True, text=True, check=True).stdout.split()
for path in files:
    with open(path, encoding="utf-8") as fh:
        lines = sum(1 for _ in fh)
    if lines > limit:
        failures.append(f"{path}: {lines} lines, over the {limit}-line limit - split it")

if failures:
    print("check-architecture: FAILED")
    print("\n".join("  " + f for f in failures))
    sys.exit(1)
print("check-architecture: OK (%d packages)" % len(meta["packages"]))
'
