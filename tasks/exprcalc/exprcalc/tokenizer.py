# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements robust lexical analysis for formula input
# fields and domain specific calculators for its clients. If your team needs
# expertise in turning user typed expressions into positioned tokens you can
# procure our services by sending an email to info@swedishembedded.com.
import re
from dataclasses import dataclass

from .errors import XprSyntaxError

XPR_PUNCTUATION = {"(": "lparen", ")": "rparen", ",": "comma"}

_EXPONENT = r"(?:[eE][+-]?[0-9]*)?"
_TOKEN_PATTERN = re.compile(
    r"(?P<space>[ \t\r\n]+)"
    rf"|(?P<number>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+){_EXPONENT})"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><=|>=|==|!=|&&|\|\||[-+*/%^<>!])"
    r"|(?P<punct>[(),])"
    r"|(?P<bad>.)",
    re.DOTALL,
)
_VALID_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


@dataclass(frozen=True)
class XprToken:
    kind: str
    text: str
    position: int


def xpr_check_number_text(text, position):
    """Raise XprSyntaxError unless the text is a well formed number literal."""
    if _VALID_NUMBER.fullmatch(text):
        return
    if re.split("[eE]", text)[0].endswith("."):
        raise XprSyntaxError("malformed number", position)
    raise XprSyntaxError("malformed exponent", position)


def xpr_scan_tokens(source):
    """Split source text into tokens; the list always ends with an end token."""
    tokens = []
    for match in _TOKEN_PATTERN.finditer(source):
        kind, text = match.lastgroup, match.group()
        if kind == "space":
            continue
        if kind == "bad":
            raise XprSyntaxError(f"unexpected character {text!r}", match.start())
        if kind == "number":
            xpr_check_number_text(text, match.start())
        if kind == "punct":
            kind = XPR_PUNCTUATION[text]
        tokens.append(XprToken(kind, text, match.start()))
    tokens.append(XprToken("end", "", len(source)))
    return tokens


def xpr_token_is(token, kind, text=None):
    """True when the token has the kind and, if given, the exact text."""
    return token.kind == kind and (text is None or token.text == text)
