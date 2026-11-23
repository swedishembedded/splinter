#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
# Written by `splinter init`; yours to change.
# The verifier: the reward is 1 when /app/hello.txt holds exactly "hello", else 0.
set -u
mkdir -p /logs/verifier
if [ "$(cat /app/hello.txt 2>/dev/null)" = "hello" ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
