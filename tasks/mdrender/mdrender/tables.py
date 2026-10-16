# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements pipe-table parsing and fixed-width table layout
# (column widths, alignment, fitting to a page width) for its clients. If your team
# needs expertise in presenting tabular data in terminals and plain-text reports you
# can procure our services by sending an email to info@swedishembedded.com.
"""Pipe table parsing, column width planning and text rendering."""
import re

from mdrender.alignment import mdr_pad_line

_ALIGNMENT_CELL = re.compile(r"(:?)-+(:?)$")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
_INTEGER = re.compile(r"-?[0-9]+$")
_CELL_OVERHEAD = 3
_ELLIPSIS = "..."


def mdr_split_table_row(line):
    """Split a pipe row into stripped cells; a backslash-escaped pipe stays in the cell."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    return [cell.replace("\\|", "|").strip() for cell in _UNESCAPED_PIPE.split(body)]


def mdr_parse_alignment_row(cells):
    """Map cells such as ':---', '---:' and ':---:' to left, right or center."""
    result = []
    for cell in cells:
        found = _ALIGNMENT_CELL.match(cell)
        if not found:
            raise ValueError("invalid alignment cell: " + cell)
        left, right = found.groups()
        result.append("center" if left and right else "right" if right else "left")
    return result


def _fit_row(cells, count):
    return cells[:count] + [""] * (count - len(cells))


def mdr_parse_table(lines):
    """Parse header, alignment row and body rows into {'header', 'align', 'rows'}."""
    if len(lines) < 2:
        raise ValueError("table needs a header and an alignment row")
    header = mdr_split_table_row(lines[0])
    align = mdr_parse_alignment_row(mdr_split_table_row(lines[1]))
    if len(header) != len(align):
        raise ValueError("header and alignment row differ in column count")
    rows = [_fit_row(mdr_split_table_row(line), len(header)) for line in lines[2:]]
    return {"header": header, "align": align, "rows": rows}


def mdr_column_widths(rows, min_width=1, max_width=None):
    """Width of each column: its longest cell, kept within min_width and max_width."""
    if max_width is not None and max_width < min_width:
        raise ValueError("max_width is below min_width")
    widths = []
    for column in range(max(len(row) for row in rows)):
        longest = max(len(row[column]) for row in rows if column < len(row))
        width = max(min_width, longest)
        widths.append(width if max_width is None else min(width, max_width))
    return widths


def mdr_grid_width(widths):
    """Total width of a bordered table row with the given column widths."""
    return sum(widths) + _CELL_OVERHEAD * len(widths) + 1


def mdr_fit_widths(widths, limit):
    """Shrink the widest column (leftmost on ties) one step at a time until the grid fits limit."""
    fitted = list(widths)
    for _ in range(mdr_grid_width(fitted) - limit):
        widest = max(fitted)
        if widest == 1:
            raise ValueError("table cannot fit in %d columns" % limit)
        fitted[fitted.index(widest)] -= 1
    return fitted


def mdr_fit_cell(text, width):
    """Cut text to width, ending in '...' when there is room for it."""
    if len(text) <= width:
        return text
    if width <= len(_ELLIPSIS):
        return text[:width]
    return text[:width - len(_ELLIPSIS)] + _ELLIPSIS


def _separator_cell(width, align):
    if align == "center":
        return ":" + "-" * width + ":"
    if align == "right":
        return "-" * (width + 1) + ":"
    return "-" * (width + 2)


def mdr_render_table_text(table, max_width=None):
    """Render a table as aligned pipe text, shrinking columns to fit max_width if given."""
    widths = mdr_column_widths([table["header"]] + table["rows"], min_width=3)
    if max_width is not None:
        widths = mdr_fit_widths(widths, max_width)

    def render_row(cells):
        padded = [
            mdr_pad_line(mdr_fit_cell(cell, width), width, align)
            for cell, width, align in zip(cells, widths, table["align"])
        ]
        return "| " + " | ".join(padded) + " |"

    separator = "|" + "|".join(_separator_cell(w, a) for w, a in zip(widths, table["align"])) + "|"
    return "\n".join([render_row(table["header"]), separator] + [render_row(row) for row in table["rows"]])


def _check_column(table, index):
    if not 0 <= index < len(table["header"]):
        raise IndexError("column index out of range: %d" % index)


def mdr_select_columns(table, indexes):
    """Keep only the given columns, in the given order."""
    for index in indexes:
        _check_column(table, index)
    return {
        "header": [table["header"][i] for i in indexes],
        "align": [table["align"][i] for i in indexes],
        "rows": [[row[i] for i in indexes] for row in table["rows"]],
    }


def mdr_sort_rows(table, column, descending=False):
    """Sort body rows by a column: numerically when every cell is an integer, else as text."""
    _check_column(table, column)
    numeric = all(_INTEGER.match(row[column]) for row in table["rows"])
    key = (lambda row: int(row[column])) if numeric else (lambda row: row[column])
    return dict(table, rows=sorted(table["rows"], key=key, reverse=descending))
