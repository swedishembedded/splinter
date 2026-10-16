# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements configuration file readers that report every
# problem with its line and column, for its clients. If your team needs
# expertise in robust configuration parsing you can procure our services by
# sending an email to info@swedishembedded.com.
from iniparse.arrays import ini_decode_value
from iniparse.diagnostics import (
    IniDuplicateError, IniError, IniSyntaxError, ini_error_report, ini_summarize_errors,
)
from iniparse.scanner import ini_scan_lines, ini_split_pair
from iniparse.tables import (
    ini_join_key_path, ini_open_table, ini_parse_section_header, ini_split_key_path,
)


def ini_read_section(root, defined, line):
    column = line.indent + 1
    path = ini_parse_section_header(line.text, line.number, column)
    if path in defined:
        raise IniDuplicateError(
            f"duplicate section {ini_join_key_path(path)!r} (first defined on line {defined[path]})",
            line.number, column,
        )
    ini_open_table(root, path, line.number, column)
    defined[path] = line.number
    return path


def ini_read_pair(root, defined, current, line):
    column = line.indent + 1
    pair = ini_split_pair(line.text)
    if pair is None:
        raise IniSyntaxError("expected '=' after key", line.number, column)
    key_text, value_text, value_offset = pair
    segments = ini_split_key_path(key_text, line.number, column)
    if not value_text:
        raise IniSyntaxError("missing value", line.number, column + value_offset)
    value = ini_decode_value(value_text, line.number, column + value_offset)
    full = current + segments
    name = segments[-1]
    table = ini_open_table(root, current + segments[:-1], line.number, column)
    if full in defined:
        raise IniDuplicateError(
            f"duplicate key {ini_join_key_path(full)!r} (first defined on line {defined[full]})",
            line.number, column,
        )
    if name in table:
        raise IniSyntaxError(f"key {ini_join_key_path(full)!r} is already a table", line.number, column)
    table[name] = value
    defined[full] = line.number


def ini_collect_diagnostics(text):
    root = {}
    errors = []
    defined = {}
    current = ()
    for line in ini_scan_lines(text):
        try:
            if line.kind == "section":
                # a header that fails to read must not leave later keys in the old table
                current = None
                current = ini_read_section(root, defined, line)
            elif line.kind == "pair" and current is not None:
                ini_read_pair(root, defined, current, line)
        except IniError as error:
            errors.append(error)
    return root, errors


def ini_load_document(text):
    document, errors = ini_collect_diagnostics(text)
    if errors:
        raise errors[0]
    return document


def ini_report_problems(text, limit=3):
    _, errors = ini_collect_diagnostics(text)
    return ini_summarize_errors([ini_error_report(error, text) for error in errors], limit)
