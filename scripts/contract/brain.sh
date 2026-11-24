#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving that the training records one system writes are the records
# another system's trainer reads, for its clients. If your team needs expertise in training-data contracts between
# agent runtimes and trainers then you can procure our services by sending an email to info@swedishembedded.com.
#
# Runs the data-contract test (crates/pipelines/tests/contract_brain): the datasets Splinter's views write, read by a
# served brain exactly as its trainer reads them, on a tokenizer-only model on the CPU. Exits 3 when there is no
# brain binary, so a missing prerequisite is never read as a pass.
#
# usage: scripts/contract/brain.sh <cargo> <profile flag> <brain binary>

set -uo pipefail

cargo=${1:-cargo}
profile=${2:---release}
brain=${3:?usage: brain.sh <cargo> <profile flag> <brain binary>}
[ -x "$brain" ] || { echo "brain.sh: no brain binary at $brain (BRAIN_BIN)" >&2; exit 3; }
# The test runs in its crate's directory: the binary's directory goes on PATH as an absolute path.
brain=$(realpath "$brain")
PATH="$(dirname "$brain"):$PATH"

exec "$cargo" test $profile -p splinter-pipelines --test contract_brain -- --ignored --nocapture
