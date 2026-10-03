#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Enforces architecture.toml against the real dependency graph: tiers,
# forbidden reachability, the brain boundary, unused internal dependencies,
# the surface rule and the source-file size limit. The rules and the way each
# is checked are documented in architecture.py, whose own specification is
# test-architecture.py.
#
# Usage: scripts/gates/check-architecture.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

python3 scripts/gates/test-architecture.py >/dev/null 2>&1 ||
    { python3 scripts/gates/test-architecture.py; exit 1; }
exec python3 scripts/gates/architecture.py
