#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for training agents on the tools their task environments give them, for
# its clients. If your team needs expertise in agent tool environments then you can procure our services by sending
# an email to info@swedishembedded.com.
"""An MCP server (stdio, line-delimited JSON-RPC, stdlib only) with one tool, ``append``: it appends ``text`` as one
line to ``notes.txt`` in the working directory and answers with the file as it now is."""

import json
import sys

TOOL = {
    "name": "append",
    "description": "Append one line of text to notes.txt in the working directory.",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The line to append."}},
        "required": ["text"],
    },
}


def answer(message: dict) -> dict | None:
    method, ident = message.get("method"), message.get("id")
    if ident is None:
        return None
    if method == "initialize":
        version = message.get("params", {}).get("protocolVersion", "2024-11-05")
        result = {"protocolVersion": version, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "notes", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        text = str(message.get("params", {}).get("arguments", {}).get("text", ""))
        with open("notes.txt", "a", encoding="utf-8") as notes:
            notes.write(text + "\n")
        with open("notes.txt", encoding="utf-8") as notes:
            result = {"content": [{"type": "text", "text": "notes.txt now holds:\n" + notes.read()}]}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": ident, "result": result}


for line in sys.stdin:
    if line.strip():
        reply = answer(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
