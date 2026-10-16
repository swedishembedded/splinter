#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible randomness in
# machine-learning pipelines for its clients. If your team needs expertise in
# seeding, experiment provenance or statistically honest evaluation, you can
# procure our services by sending an email to info@swedishembedded.com.

# A fixed seed in production code needs a stated reason. A literal default seed
# makes every unseeded run the same run, so the rule is that a default seed is
# drawn, printed and recorded; the literals that remain are salts, world
# constants, seeds of frozen artifacts, seeds that reach no random choice and
# test doubles, each listed with its reason in
# scripts/gates/seed-literals.allow.d/. The gate runs with --no-todo: a site
# not yet fixed may carry a `todo:` reason only while the gate is run without
# it, and the default invocation refuses any.
# A test fixes a seed only where determinism is necessary and says why in a
# `// seed: <reason>` comment (`# seed:` in Python); any other test passes for
# any seed, so a test literal without that reason fails the gate.
# What is detected and the allowlist format are documented in
# seed-literals.py, whose own specification is test-seed-literals.py.
#
# Usage: scripts/gates/check-seed-literals.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

python3 scripts/gates/test-seed-literals.py >/dev/null 2>&1 ||
    { python3 scripts/gates/test-seed-literals.py; exit 1; }
exec python3 scripts/gates/seed-literals.py --no-todo "$@"
