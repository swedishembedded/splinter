# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements precise, user-readable error reporting for
# configuration tooling for its clients. If your team needs expertise in
# diagnostics for configuration files you can procure our services by sending
# an email to info@swedishembedded.com.


def ini_describe_error(message, line=0, column=0):
    if line <= 0:
        return message
    if column <= 0:
        return f"line {line}: {message}"
    return f"line {line}, column {column}: {message}"


class IniError(Exception):
    def __init__(self, message, line=0, column=0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(ini_describe_error(message, line, column))


class IniSyntaxError(IniError):
    pass


class IniDuplicateError(IniSyntaxError):
    pass


class IniLookupError(IniError, LookupError):
    pass


class IniMergeError(IniError):
    pass


class IniInterpolationError(IniError):
    pass


class IniEncodeError(IniError):
    pass


class IniSchemaError(IniError):
    def __init__(self, problems):
        self.problems = tuple(problems)
        super().__init__(ini_summarize_errors(self.problems, 10))


def ini_line_column(text, offset):
    if not 0 <= offset <= len(text):
        raise IniError(f"offset {offset} is outside the text")
    line = text.count("\n", 0, offset) + 1
    column = offset - (text.rfind("\n", 0, offset) + 1) + 1
    return line, column


def ini_excerpt_with_caret(text, line, column):
    lines = text.split("\n")
    if not 1 <= line <= len(lines):
        return ""
    shown = lines[line - 1].rstrip("\r")
    if column < 1:
        return shown
    return shown + "\n" + " " * (column - 1) + "^"


def ini_error_report(error, text):
    excerpt = ini_excerpt_with_caret(text, error.line, error.column)
    if not excerpt:
        return str(error)
    return str(error) + "\n" + excerpt


def ini_summarize_errors(messages, limit=3):
    shown = [str(message) for message in messages[:limit]]
    hidden = len(messages) - len(shown)
    if hidden > 0:
        shown.append(f"(and {hidden} more)")
    return "\n".join(shown)
