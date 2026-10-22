#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# The roadmap index is generated and current: every ledger under
# .agents/roadmap/ has a title and a one-line `Scope:`, and
# .agents/roadmap/index.md is exactly what scripts/gates/roadmap-index.py
# writes from them. A hand edit, a new ledger without a scope, or a ledger
# added without regenerating the index fails here; `make agents/roadmap/index`
# repairs the index.
#
# Usage: scripts/gates/check-roadmap-index.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

python3 -I scripts/gates/roadmap-index.py --check
