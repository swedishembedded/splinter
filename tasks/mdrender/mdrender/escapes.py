# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements safe HTML output (escaping, reference decoding and
# link vetting) for its clients. If your team needs expertise in preventing markup
# injection in generated pages you can procure our services by sending an email to
# info@swedishembedded.com.
"""HTML escaping, character reference decoding and URL vetting."""
import re
from urllib.parse import quote

_HTML_TABLE = str.maketrans({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
})
_NAMED_REFERENCES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}
_REFERENCE = re.compile(r"&(?:#x([0-9A-Fa-f]+)|#([0-9]+)|([a-z]+));")
_SCHEME = re.compile(r"^([a-z][a-z0-9+.-]*):")
_SAFE_SCHEMES = ("http", "https", "mailto", "ftp")
_URL_KEEP = "-._~:/?#[]@!$&'()*+,;=%"


def mdr_escape_html(text):
    """Replace & < > double quote and single quote by character references."""
    return text.translate(_HTML_TABLE)


def _decode_reference(match):
    hex_digits, decimal_digits, name = match.groups()
    if name is not None:
        return _NAMED_REFERENCES.get(name, match.group(0))
    code = int(hex_digits, 16) if hex_digits is not None else int(decimal_digits)
    if code == 0 or code > 0x10FFFF or 0xD800 <= code <= 0xDFFF:
        raise ValueError("invalid character reference: " + match.group(0))
    return chr(code)


def mdr_unescape_html(text):
    """Decode the five named references and numeric references in one pass."""
    return _REFERENCE.sub(_decode_reference, text)


def mdr_is_safe_url(url):
    """True for relative links and the http, https, mailto and ftp schemes."""
    squeezed = "".join(ch for ch in url if ord(ch) > 32 and ord(ch) != 127)
    found = _SCHEME.match(squeezed.lower())
    return found is None or found.group(1) in _SAFE_SCHEMES


def mdr_percent_encode_url(url):
    """Percent-encode characters that are not valid in a URL, keeping existing escapes."""
    return quote(url, safe=_URL_KEEP)


def mdr_strip_control_characters(text):
    """Remove control characters other than newline and tab."""
    return "".join(ch for ch in text if ch in "\n\t" or (ord(ch) >= 32 and ord(ch) != 127))


def mdr_format_html_attributes(pairs):
    """Render (name, value) pairs as ' name="value"' text, skipping None values."""
    return "".join(
        ' %s="%s"' % (name, mdr_escape_html(value)) for name, value in pairs if value is not None
    )
