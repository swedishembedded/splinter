# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements typed list parsing for configuration files,
# with element-accurate error columns, for its clients. If your team needs
# expertise in nested value grammars you can procure our services by sending
# an email to info@swedishembedded.com.
from iniparse.diagnostics import IniSyntaxError
from iniparse.scalars import ini_decode_scalar, ini_value_kind
from iniparse.scanner import ini_unquoted_chars


def ini_split_top_level(text):
    pieces = []
    depth = 0
    start = 0
    for index, char in ini_unquoted_chars(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            pieces.append((text[start:index], start))
            start = index + 1
    pieces.append((text[start:], start))
    return pieces


def ini_decode_array(text, line=0, column=0):
    if not text.endswith("]"):
        raise IniSyntaxError("unterminated array", line, column)
    inner = text[1:-1]
    if not inner.strip():
        return []
    pieces = ini_split_top_level(inner)
    if not pieces[-1][0].strip():
        pieces.pop()
    items = []
    columns = []
    for piece, offset in pieces:
        start = column + 1 + offset + len(piece) - len(piece.lstrip())
        if not piece.strip():
            raise IniSyntaxError("empty array element", line, start)
        items.append(ini_decode_value(piece.strip(), line, start))
        columns.append(start)
    first = ini_value_kind(items[0])
    for item, start in zip(items, columns):
        if ini_value_kind(item) != first:
            raise IniSyntaxError(f"mixed types in array: {ini_value_kind(item)} after {first}", line, start)
    return items


def ini_decode_value(text, line=0, column=0):
    if text.startswith("["):
        return ini_decode_array(text, line, column)
    return ini_decode_scalar(text, line, column)
