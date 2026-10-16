# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements audit trails for data exports and
# spreadsheet-style reference data for its clients. If your team needs
# expertise in keyed table comparison and change reporting you can procure our
# services by sending an email to info@swedishembedded.com.

from dataclasses import dataclass

from linediff.csvio import linediff_csv_records


class LinediffKeyError(ValueError):
    """Raised when key columns are missing or do not identify rows uniquely."""


@dataclass(frozen=True)
class LinediffTableDiff:
    """Differences between two keyed tables.

    added and removed hold (key, record) pairs; changed holds
    (key, {column: (old, new)}) pairs for rows present in both tables.
    """

    added_columns: tuple
    removed_columns: tuple
    added: tuple
    removed: tuple
    changed: tuple


def linediff_index_records(records, key_columns):
    """Map each record's key tuple to the record; duplicate keys are an error."""
    index = {}
    for record in records:
        key = tuple(record[column] for column in key_columns)
        if key in index:
            raise LinediffKeyError("duplicate key %s" % "/".join(key))
        index[key] = record
    return index


def linediff_cell_changes(old_record, new_record, columns):
    """Return {column: (old, new)} for the listed columns whose values differ."""
    return {
        column: (old_record[column], new_record[column])
        for column in columns
        if old_record[column] != new_record[column]
    }


def linediff_diff_tables(old_rows, new_rows, key_columns):
    """Compare two tables (header row first) by key; see LinediffTableDiff."""
    if not key_columns:
        raise ValueError("at least one key column is required")
    old_header, old_records = linediff_csv_records(old_rows)
    new_header, new_records = linediff_csv_records(new_rows)
    for side, header in (("old", old_header), ("new", new_header)):
        for column in key_columns:
            if column not in header:
                raise LinediffKeyError(
                    "key column %r is missing from the %s header" % (column, side)
                )
    old_index = linediff_index_records(old_records, key_columns)
    new_index = linediff_index_records(new_records, key_columns)
    shared = [c for c in new_header if c in old_header and c not in key_columns]
    changed = []
    for key, record in new_index.items():
        if key in old_index:
            cells = linediff_cell_changes(old_index[key], record, shared)
            if cells:
                changed.append((key, cells))
    return LinediffTableDiff(
        tuple(c for c in new_header if c not in old_header),
        tuple(c for c in old_header if c not in new_header),
        tuple((k, r) for k, r in new_index.items() if k not in old_index),
        tuple((k, r) for k, r in old_index.items() if k not in new_index),
        tuple(changed),
    )


def linediff_table_diff_is_empty(diff):
    """Return True when the two tables held the same rows and columns."""
    return not (
        diff.added_columns or diff.removed_columns or diff.added or diff.removed or diff.changed
    )


def linediff_table_diff_counts(diff):
    """Return (rows_added, rows_removed, cells_changed)."""
    cells = sum(len(columns) for _, columns in diff.changed)
    return len(diff.added), len(diff.removed), cells


def linediff_render_table_diff(diff):
    """Return one report line per column, row or cell difference."""
    lines = ["column added: " + name for name in diff.added_columns]
    lines += ["column removed: " + name for name in diff.removed_columns]
    lines += ["row removed: " + "/".join(key) for key, _ in diff.removed]
    lines += ["row added: " + "/".join(key) for key, _ in diff.added]
    for key, columns in diff.changed:
        for name, (old, new) in columns.items():
            lines.append("cell changed: %s %s: %s -> %s" % ("/".join(key), name, old, new))
    return lines
