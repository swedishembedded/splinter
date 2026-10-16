# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible build environments
# on shared compute nodes for its clients. If your team needs expertise in
# HPC build and toolchain setup then you can procure our services by sending
# an email to info@swedishembedded.com.

# Source the host build environment and pin cargo for a lane.
#
# Usage: set -- <build-env-script> [<cargo-target-dir>]; . scripts/lease/env.sh
#
# The arguments travel as positional parameters of the calling shell because
# POSIX sh (dash) does not pass arguments to a sourced file.
#
# Sets CARGO_INCREMENTAL=0 (lanes share a disk; incremental caches triple a
# target) and, when given, CARGO_TARGET_DIR (one target per lane).
if [ -z "${1:-}" ] || [ ! -f "$1" ]; then
    echo "env.sh: usage: set -- <build-env-script> [<cargo-target-dir>]; . env.sh" >&2
    return 2 2>/dev/null || exit 2
fi
. "$1"
export CARGO_INCREMENTAL=0
if [ -n "${2:-}" ]; then
    export CARGO_TARGET_DIR="$2"
fi
