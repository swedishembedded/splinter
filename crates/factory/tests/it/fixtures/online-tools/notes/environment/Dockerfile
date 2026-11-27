# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The task's image: Python (pinned by digest), the notes MCP server, the agent configuration that makes it the only
# tool, and the starting files.
FROM python:3.13-slim@sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50
COPY notes_mcp.py /usr/local/share/notes/notes_mcp.py
COPY sven.yaml /etc/sven/config.yaml
COPY workspace/ /workspace/repo/
WORKDIR /workspace/repo
