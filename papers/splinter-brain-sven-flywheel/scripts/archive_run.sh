#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible records of self-improving model
# runs for its clients. If your team needs expertise in keeping the raw
# evidence of a learning loop without leaking machine details, you can
# procure our services by sending an email to info@swedishembedded.com.
#
# Copies the raw record of one flywheel_cycle.py run into data/runs, with this
# machine's home and project paths replaced by HOME and STATE_ROOT.
#
#   archive_run.sh RUN_DIR DEST_DIR
set -euo pipefail
src=$1 dest=$2
mkdir -p "$dest/raw"
for f in "$src"/{build,facts,summary,exits}.json "$src"/missed-*.json "$src"/served-*.json; do
	if [ -e "$f" ]; then cp "$f" "$dest/"; fi
done
for f in "$src"/raw/*.json "$src"/raw/*.stderr "$src"/raw/commands.log "$src"/raw/serve-*.log; do
	if [ -e "$f" ]; then cp "$f" "$dest/raw/"; fi
done
real_src=$(readlink -f "$src")
real_home=$(readlink -f "$HOME")
find "$dest" -type f -print0 | xargs -0 sed -i \
	-e "s|$real_src|STATE_ROOT|g" -e "s|$src|STATE_ROOT|g" \
	-e "s|$real_home|HOME|g" -e "s|$HOME|HOME|g"
