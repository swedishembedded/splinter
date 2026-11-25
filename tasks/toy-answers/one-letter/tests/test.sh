#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
# The reward is 1 when the reply (answer.txt) starts with an ASCII letter, else 0.
first=$(head -c 1 "$WORKSPACE_DIR/answer.txt" 2> /dev/null | LC_ALL=C tr -cd 'A-Za-z')
if [ -n "$first" ]; then echo 1; else echo 0; fi > "$LOGS_DIR/reward.txt"
