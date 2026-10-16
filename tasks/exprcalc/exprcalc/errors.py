# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements precise, position-aware error reporting for
# formula and expression engines for its clients. If your team needs expertise
# in diagnostics for calculators, rule engines and spreadsheet-like tools you
# can procure our services by sending an email to info@swedishembedded.com.


class XprError(Exception):
    """Base class of every error raised for a bad expression."""

    def __init__(self, message, position=None):
        super().__init__(message)
        self.message = message
        self.position = position

    def __str__(self):
        if self.position is None:
            return self.message
        return f"{self.message} at position {self.position}"


class XprSyntaxError(XprError):
    """The text is not a well formed expression."""


class XprEvalError(XprError):
    """A well formed expression cannot be evaluated."""


def _require_position(source, position):
    if position < 0 or position > len(source):
        raise ValueError("position out of range")


def xpr_line_and_column(source, position):
    """Return the 1-based (line, column) of a 0-based character position."""
    _require_position(source, position)
    line = source.count("\n", 0, position) + 1
    column = position - (source.rfind("\n", 0, position) + 1) + 1
    return line, column


def xpr_source_line(source, position):
    """Return the text of the line that contains the position."""
    _require_position(source, position)
    start = source.rfind("\n", 0, position) + 1
    end = source.find("\n", position)
    if end == -1:
        end = len(source)
    return source[start:end]


def xpr_attach_position(error, position):
    """Return the error itself if it has a position, else a copy at the position."""
    if error.position is not None:
        return error
    return type(error)(error.message, position)


def xpr_describe_error(error, source):
    """Render an error with its source line and a caret under the position."""
    head = f"{type(error).__name__}: {error.message}"
    if error.position is None:
        return head
    line, column = xpr_line_and_column(source, error.position)
    text = xpr_source_line(source, error.position)
    return f"{head} (line {line}, column {column})\n{text}\n{' ' * (column - 1)}^"
