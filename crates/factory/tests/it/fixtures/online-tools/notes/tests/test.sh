#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Reward 1 when notes.txt holds the asked line, else 0.
mkdir -p /logs/verifier
if grep -qx 'hello from the tool' /workspace/repo/notes.txt 2>/dev/null; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
