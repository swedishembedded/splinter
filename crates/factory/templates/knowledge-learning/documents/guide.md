<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!-- Written by `splinter init`; replace it with your own documents. -->
# The example tool

## Configuration

The example tool reads its settings from config.yaml in the working directory. A setting
it does not know is refused with its name, and the tool exits with status 2.

## Logging

The example tool writes its log to example.log and keeps the last 7 logs, removing older
ones when it starts.
