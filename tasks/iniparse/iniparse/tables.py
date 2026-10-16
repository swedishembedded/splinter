# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dotted key paths and nested table handling
# for configuration documents, for its clients. If your team needs expertise
# in hierarchical configuration namespaces you can procure our services by
# sending an email to info@swedishembedded.com.
import re

from iniparse.diagnostics import IniSyntaxError
from iniparse.scalars import ini_decode_string, ini_quote_string
from iniparse.scanner import ini_unquoted_chars

BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def ini_is_bare_key(text):
    return BARE_KEY.fullmatch(text) is not None


def ini_split_key_path(text, line=0, column=0):
    cuts = [-1] + [index for index, char in ini_unquoted_chars(text) if char == "."] + [len(text)]
    segments = []
    for before, after in zip(cuts, cuts[1:]):
        raw = text[before + 1:after]
        start = column + before + 1 + len(raw) - len(raw.lstrip())
        segment = raw.strip()
        if not segment:
            raise IniSyntaxError("empty key segment", line, start)
        if segment[0] in "\"'":
            segments.append(ini_decode_string(segment, line, start))
        elif ini_is_bare_key(segment):
            segments.append(segment)
        else:
            raise IniSyntaxError(f"invalid key {segment!r}", line, start)
    return tuple(segments)


def ini_quote_key(segment):
    return segment if ini_is_bare_key(segment) else ini_quote_string(segment)


def ini_join_key_path(segments):
    return ".".join(ini_quote_key(segment) for segment in segments)


def ini_parse_section_header(content, line=0, column=0):
    if not (content.startswith("[") and content.endswith("]")):
        raise IniSyntaxError("malformed section header", line, column)
    return ini_split_key_path(content[1:-1], line, column + 1)


def ini_open_table(root, path, line=0, column=0):
    table = root
    for depth, segment in enumerate(path):
        table = table.setdefault(segment, {})
        if not isinstance(table, dict):
            raise IniSyntaxError(f"{ini_join_key_path(path[:depth + 1])!r} is not a table", line, column)
    return table
