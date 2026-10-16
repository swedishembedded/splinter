# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements configuration writers whose output reads back
# to exactly the same data, for its clients. If your team needs expertise in
# lossless configuration serialization you can procure our services by sending
# an email to info@swedishembedded.com.
import math

from iniparse.diagnostics import IniEncodeError
from iniparse.scalars import ini_quote_string, ini_value_kind
from iniparse.tables import ini_join_key_path, ini_quote_key

ARRAY_INDENT = "    "


def ini_encode_scalar(value):
    kind = ini_value_kind(value)
    if kind == "boolean":
        return "true" if value else "false"
    if kind == "float":
        if not math.isfinite(value):
            raise IniEncodeError(f"cannot encode non-finite float {value!r}")
        return repr(value)
    if kind == "string":
        return ini_quote_string(value)
    if kind == "integer":
        return str(value)
    raise IniEncodeError(f"cannot encode a {kind} as a scalar")


def ini_encode_value(value):
    if ini_value_kind(value) != "array":
        return ini_encode_scalar(value)
    if len({ini_value_kind(item) for item in value}) > 1:
        raise IniEncodeError("cannot encode an array of mixed types")
    return "[" + ", ".join(ini_encode_value(item) for item in value) + "]"


def ini_encode_pair(key, value, width):
    inline = f"{ini_quote_key(key)} = {ini_encode_value(value)}"
    if len(inline) <= width or not value or not isinstance(value, list):
        return inline
    items = [ARRAY_INDENT + ini_encode_value(item) + "," for item in value]
    return "\n".join([f"{ini_quote_key(key)} = ["] + items + ["]"])


def ini_dump_table(table, path, width, blocks):
    for key in table:
        if not isinstance(key, str):
            raise IniEncodeError(f"table keys must be strings, not {type(key).__name__}")
    scalars = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
    children = [(key, value) for key, value in table.items() if isinstance(value, dict)]
    if scalars or (path and not children):
        lines = [f"[{ini_join_key_path(path)}]"] if path else []
        lines += [ini_encode_pair(key, value, width) for key, value in scalars]
        blocks.append("\n".join(lines))
    for key, value in children:
        ini_dump_table(value, path + (key,), width, blocks)


def ini_dump_document(document, width=72):
    blocks = []
    ini_dump_table(document, (), width, blocks)
    return "\n\n".join(blocks) + "\n" if blocks else ""
