# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reference expansion inside configuration
# values, with cycle detection, for its clients. If your team needs expertise
# in templated configuration you can procure our services by sending an email
# to info@swedishembedded.com.
import re

from iniparse.diagnostics import IniInterpolationError, IniSyntaxError
from iniparse.lookup import ABSENT, ini_describe_path, ini_lookup, ini_path_segments
from iniparse.scalars import ini_value_kind

MISSING_MODES = ("error", "keep", "empty")
REFERENCE = re.compile(r"\$\$\{|\$\{([^}]*)\}|\$\{")


def ini_tokenize_references(text):
    tokens = []
    pending = ""
    cursor = 0
    for match in REFERENCE.finditer(text):
        pending += text[cursor:match.start()]
        cursor = match.end()
        body = match.group(1)
        if match.group(0) == "$${":
            pending += "${"
            continue
        if body is None:
            raise IniInterpolationError(f"unterminated reference at offset {match.start()}")
        name, separator, fallback = body.partition(":-")
        if not name.strip():
            raise IniInterpolationError(f"empty reference name at offset {match.start()}")
        if pending:
            tokens.append(("text", pending, None))
            pending = ""
        tokens.append(("ref", name.strip(), fallback if separator else None))
    pending += text[cursor:]
    if pending:
        tokens.append(("text", pending, None))
    return tokens


def ini_stringify_reference(value, name):
    kind = ini_value_kind(value)
    if kind == "boolean":
        return "true" if value else "false"
    if kind in ("array", "table"):
        raise IniInterpolationError(f"cannot embed {kind} {name!r} in text")
    return str(value)


class IniExpander:
    def __init__(self, document, missing="error"):
        if missing not in MISSING_MODES:
            raise IniInterpolationError(f"unknown missing mode {missing!r}")
        self.document = document
        self.missing = missing
        self.stack = []

    def expand_table(self, table, path=()):
        return {
            key: self.expand_table(value, path + (key,)) if isinstance(value, dict)
            else self.expand_leaf(path + (key,), value)
            for key, value in table.items()
        }

    def expand_leaf(self, path, raw):
        if path in self.stack:
            chain = self.stack[self.stack.index(path):] + [path]
            raise IniInterpolationError("circular reference: " + " -> ".join(ini_describe_path(p) for p in chain))
        self.stack.append(path)
        value = self.expand_value(raw)
        self.stack.pop()
        return value

    def expand_value(self, raw):
        if isinstance(raw, list):
            return [self.expand_value(item) for item in raw]
        if not isinstance(raw, str):
            return raw
        tokens = ini_tokenize_references(raw)
        if len(tokens) == 1 and tokens[0][0] == "ref":
            return self.resolve(tokens[0][1], tokens[0][2])
        return "".join(
            text if kind == "text" else ini_stringify_reference(self.resolve(text, fallback), text)
            for kind, text, fallback in tokens
        )

    def resolve(self, name, fallback):
        try:
            segments = ini_path_segments(name)
        except IniSyntaxError:
            raise IniInterpolationError(f"invalid reference name {name!r}") from None
        raw = ini_lookup(self.document, segments, ABSENT)
        if raw is ABSENT:
            return self.absent_value(name, fallback)
        if isinstance(raw, dict):
            raise IniInterpolationError(f"reference {name!r} is a table")
        return self.expand_leaf(segments, raw)

    def absent_value(self, name, fallback):
        if fallback is not None:
            return fallback
        if self.missing == "keep":
            return "${" + name + "}"
        if self.missing == "empty":
            return ""
        raise IniInterpolationError(f"unknown reference {name!r}")


def ini_expand_document(document, missing="error"):
    return IniExpander(document, missing).expand_table(document)


def ini_expand_text(text, document, missing="error"):
    return IniExpander(document, missing).expand_value(text)
