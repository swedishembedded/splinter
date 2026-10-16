# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements line-oriented scanners for configuration text
# that tell comments from values reliably, for its clients. If your team needs
# expertise in lexical analysis of configuration formats you can procure our
# services by sending an email to info@swedishembedded.com.
from dataclasses import dataclass

COMMENT_MARKERS = "#;"


@dataclass(frozen=True)
class IniLine:
    number: int
    end: int
    kind: str
    text: str
    comment: str
    indent: int


def ini_split_lines(text):
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def ini_unquoted_chars(text):
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif quote:
            if char == quote:
                quote = ""
            elif char == "\\" and quote == '"':
                escaped = True
        elif char in "\"'":
            quote = char
        else:
            yield index, char


def ini_find_comment_start(text):
    for index, char in ini_unquoted_chars(text):
        if char in COMMENT_MARKERS:
            return index
    return -1


def ini_strip_comment(text):
    start = ini_find_comment_start(text)
    if start < 0:
        return text.strip(), ""
    return text[:start].strip(), text[start + 1:].strip()


def ini_bracket_depth(text):
    depth = 0
    for _, char in ini_unquoted_chars(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
    return depth


def ini_split_pair(content):
    for index, char in ini_unquoted_chars(content):
        if char == "=":
            raw = content[index + 1:]
            offset = index + 1 + len(raw) - len(raw.lstrip())
            return content[:index].strip(), raw.strip(), offset


def ini_classify_line(content, comment):
    if content:
        return "section" if content[0] == "[" else "pair"
    return "comment" if comment else "blank"


def ini_scan_lines(text):
    pending = iter(enumerate(ini_split_lines(text), 1))
    found = []
    for number, raw in pending:
        content, comment = ini_strip_comment(raw)
        kind = ini_classify_line(content, comment)
        end = number
        if kind == "pair":
            while ini_bracket_depth(content) > 0:
                following = next(pending, None)
                if following is None:
                    break
                end, more = following
                content = content + " " + ini_strip_comment(more)[0]
        found.append(IniLine(number, end, kind, content, comment, len(raw) - len(raw.lstrip())))
    return found
