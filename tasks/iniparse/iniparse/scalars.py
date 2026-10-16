# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements strict typed value decoding for configuration
# files, with exact error positions, for its clients. If your team needs
# expertise in scalar typing and string escaping you can procure our services
# by sending an email to info@swedishembedded.com.
import re

from iniparse.diagnostics import IniEncodeError, IniSyntaxError

ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
ESCAPE_NAMES = {char: name for name, char in ESCAPES.items()}
HEX_DIGITS = "0123456789abcdefABCDEF"
PREFIXED_INTEGER = re.compile(
    r"0(x[0-9a-fA-F]+(_[0-9a-fA-F]+)*|o[0-7]+(_[0-7]+)*|b[01]+(_[01]+)*)"
)
DECIMAL_INTEGER = re.compile(r"[+-]?(0|[1-9][0-9]*(_[0-9]+)*)")
DECIMAL_FLOAT = re.compile(
    r"[+-]?(0|[1-9][0-9]*(_[0-9]+)*)(\.[0-9]+([eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)"
)


def ini_decode_number(text):
    if PREFIXED_INTEGER.fullmatch(text):
        return int(text.replace("_", ""), 0)
    if DECIMAL_INTEGER.fullmatch(text):
        return int(text.replace("_", ""))
    if DECIMAL_FLOAT.fullmatch(text):
        return float(text.replace("_", ""))


def ini_decode_escape(text, index, line=0, column=0):
    if index + 1 >= len(text):
        raise IniSyntaxError("incomplete escape sequence", line, column + index)
    code = text[index + 1]
    if code in ESCAPES:
        return ESCAPES[code], index + 2
    width = {"x": 2, "u": 4}.get(code)
    if width is None:
        raise IniSyntaxError(f"unknown escape '\\{code}'", line, column + index)
    digits = text[index + 2:index + 2 + width]
    if len(digits) != width or any(char not in HEX_DIGITS for char in digits):
        raise IniSyntaxError(f"bad \\{code} escape", line, column + index)
    number = int(digits, 16)
    if 0xD800 <= number <= 0xDFFF:
        raise IniSyntaxError(f"surrogate in \\{code} escape", line, column + index)
    return chr(number), index + 2 + width


def ini_decode_string(text, line=0, column=0):
    if text[0] == "'":
        end = text.find("'", 1)
        if end == -1:
            raise IniSyntaxError("unterminated string", line, column)
        if end != len(text) - 1:
            raise IniSyntaxError("unexpected text after string", line, column + end + 1)
        return text[1:end]
    chars = []
    skip = 0
    for index in range(1, len(text)):
        if skip:
            skip -= 1
            continue
        char = text[index]
        if char == '"':
            if index != len(text) - 1:
                raise IniSyntaxError("unexpected text after string", line, column + index + 1)
            return "".join(chars)
        if char == "\\":
            decoded, after = ini_decode_escape(text, index, line, column)
            chars.append(decoded)
            skip = after - index - 1
        else:
            chars.append(char)
    raise IniSyntaxError("unterminated string", line, column)


def ini_decode_scalar(text, line=0, column=0):
    if text == "true":
        return True
    if text == "false":
        return False
    if text.startswith(('"', "'")):
        return ini_decode_string(text, line, column)
    number = ini_decode_number(text)
    if number is None:
        raise IniSyntaxError(f"unrecognized value {text!r}", line, column)
    return number


def ini_escape_string(value):
    pieces = []
    for char in value:
        if char in ESCAPE_NAMES:
            pieces.append("\\" + ESCAPE_NAMES[char])
        elif ord(char) < 32:
            pieces.append(f"\\x{ord(char):02x}")
        else:
            pieces.append(char)
    return "".join(pieces)


def ini_quote_string(value):
    return '"' + ini_escape_string(value) + '"'


def ini_value_kind(value):
    if isinstance(value, bool):
        return "boolean"
    for kind, name in ((int, "integer"), (float, "float"), (str, "string"), (list, "array"), (dict, "table")):
        if isinstance(value, kind):
            return name
    raise IniEncodeError(f"unsupported value of type {type(value).__name__}")
