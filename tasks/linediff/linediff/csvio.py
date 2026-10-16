# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reliable tabular data exchange for its
# clients. If your team needs expertise in reading and writing delimited text
# with quoting, embedded line breaks and ragged rows you can procure our
# services by sending an email to info@swedishembedded.com.

_LINE_BREAKS = "\r\n"


class LinediffCsvError(ValueError):
    """Raised when delimited text or a table shape is invalid."""


def _csv_check_delimiter(delimiter):
    if len(delimiter) != 1 or delimiter == '"' or delimiter in _LINE_BREAKS:
        raise ValueError("delimiter must be one character other than a quote or line break")


def _csv_read_quoted(text, index, line):
    start_line = line
    parts = []
    while index < len(text):
        char = text[index]
        if char == '"':
            if text[index + 1:index + 2] == '"':
                parts.append('"')
                index += 2
                continue
            return "".join(parts), index + 1, line
        if char == "\n":
            line += 1
        parts.append(char)
        index += 1
    raise LinediffCsvError("line %d: unterminated quoted field" % start_line)


def linediff_csv_read(text, delimiter=","):
    """Parse text into a list of rows; quoted fields may hold delimiters and line breaks."""
    _csv_check_delimiter(delimiter)
    stops = delimiter + _LINE_BREAKS
    rows = []
    row = []
    index = 0
    line = 1
    while index < len(text):
        if text[index] == '"':
            value, index, line = _csv_read_quoted(text, index + 1, line)
            if index < len(text) and text[index] not in stops:
                raise LinediffCsvError("line %d: unexpected character after closing quote" % line)
        else:
            end = index
            while end < len(text) and text[end] not in stops:
                end += 1
            value = text[index:end]
            index = end
        row.append(value)
        if index >= len(text):
            break
        if text[index] == delimiter:
            index += 1
            if index >= len(text):
                row.append("")
            continue
        if text[index] == "\r" and text[index + 1:index + 2] == "\n":
            index += 1
        index += 1
        line += 1
        rows.append(row)
        row = []
    if row:
        rows.append(row)
    return rows


def linediff_csv_quote_field(value, delimiter=","):
    """Return value quoted when it holds the delimiter, a quote or a line break."""
    _csv_check_delimiter(delimiter)
    if any(char in value for char in delimiter + '"' + _LINE_BREAKS):
        return '"' + value.replace('"', '""') + '"'
    return value


def linediff_csv_write(rows, delimiter=",", newline="\n"):
    """Render rows as delimited text, one newline-terminated line per row."""
    out = []
    for number, row in enumerate(rows, 1):
        if not row:
            raise LinediffCsvError("row %d has no fields" % number)
        fields = [linediff_csv_quote_field(value, delimiter) for value in row]
        out.append(delimiter.join(fields) + newline)
    return "".join(out)


def linediff_csv_ragged_rows(rows):
    """Return indices of rows whose field count differs from the first row's."""
    return [index for index, row in enumerate(rows) if len(row) != len(rows[0])]


def linediff_csv_pad_rows(rows, fill=""):
    """Return rows extended with fill so that all are as wide as the widest."""
    width = max([0] + [len(row) for row in rows])
    return [list(row) + [fill] * (width - len(row)) for row in rows]


def linediff_csv_records(rows):
    """Return (header, records) with each record a dict; short rows are padded."""
    if not rows:
        raise LinediffCsvError("no header row")
    header = list(rows[0])
    seen = set()
    for name in header:
        if name in seen:
            raise LinediffCsvError("duplicate column %r in header" % name)
        seen.add(name)
    records = []
    for number, row in enumerate(rows[1:], 2):
        if len(row) > len(header):
            raise LinediffCsvError(
                "row %d has %d fields but the header has %d" % (number, len(row), len(header))
            )
        records.append(
            {name: row[k] if k < len(row) else "" for k, name in enumerate(header)}
        )
    return header, records
