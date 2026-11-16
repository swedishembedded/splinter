#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible learning environments for
# agents, for its clients. If your team needs expertise in building
# counterfactual language dialects with matching reference documentation,
# you can procure our services by sending an email to
# info@swedishembedded.com.

"""Derives the reference manual of a Lua 5.1 dialect from the stock manual.

Usage: python3 -I manual.py STOCK_MANUAL_HTML DIALECT OUTDIR

Writes OUTDIR/{strings,tables,math,operators}.html.  Every rewrite is an
exact old->new replacement that must match exactly once in the stock text,
so a changed or unexpected stock manual fails loudly instead of leaving a
stale rule behind.  The output is deterministic.
"""
import html
import re
import sys
from pathlib import Path

DIALECTS = ("zero-index", "plus-concat")

PREFACE = {
    "zero-index": """
<p>
This manual describes <b>Lua 5.1 (zero-index dialect)</b>.
It is the Lua&nbsp;5.1.5 reference manual with the rules below rewritten.
Everything not listed here behaves exactly as in standard Lua&nbsp;5.1.
<ul>
<li>String positions are 0-based for <code>string.sub</code>,
<code>string.byte</code>, <code>string.find</code> and <code>string.match</code>:
a non-negative position <em>p</em> names the character at offset <em>p</em>
(the first character is at position&nbsp;0).
Negative positions still count from the end (-1 is the last character).</li>
<li>Positions returned by <code>string.find</code>, and position captures
<code>()</code> in <code>find</code>, <code>match</code>, <code>gmatch</code>
and <code>gsub</code>, are 0-based.</li>
<li>The length operator <code>#</code> applied to a string yields the position of
its last character, that is <code>string.len(s) - 1</code>
(-1 for the empty string).
<code>string.len(s)</code> and <code>#t</code> for tables are unchanged.</li>
<li>Table indices, <code>table.*</code>, <code>math.*</code>,
<code>string.rep</code>, <code>string.format</code>, <code>string.char</code>,
<code>string.upper</code>, <code>string.lower</code> and
<code>string.reverse</code> are unchanged.</li>
</ul>
""",
    "plus-concat": """
<p>
This manual describes <b>Lua 5.1 (plus-concat dialect)</b>.
It is the Lua&nbsp;5.1.5 reference manual with the rules below rewritten.
Everything not listed here behaves exactly as in standard Lua&nbsp;5.1.
<ul>
<li>The string concatenation operator is spelled <code>+</code>.
There is no <code>..</code> operator: the two-dot sequence is a syntax error
(the vararg token <code>...</code> is unchanged).</li>
<li>When both operands of <code>+</code> are strings, the result is their
concatenation, even if they look like numbers:
<code>"1" + "2"</code> is <code>"12"</code>.</li>
<li>When at least one operand of <code>+</code> is a number, <code>+</code> is
arithmetic addition with the usual coercion of the other operand:
<code>"10" + 1</code> is <code>11</code>; <code>"abc" + 1</code> is an error.
Numbers are never converted to strings by <code>+</code>.</li>
<li><code>+</code> keeps the precedence and left associativity of binary
<code>-</code>.  The metamethod <code>__concat</code> is never used;
<code>__add</code> is tried when neither of the two rules above applies.</li>
<li><code>table.concat</code> and the rest of the standard library are
unchanged.</li>
</ul>
""",
}

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lua 5.1 ({dialect} dialect) Reference Manual - {title}</title>
</head>
<body>
<h1>Lua 5.1 ({dialect} dialect) Reference Manual: {title}</h1>
<p>
Derived from the Lua 5.1 Reference Manual by Roberto Ierusalimschy,
Luiz Henrique de Figueiredo and Waldemar Celes.
Copyright &copy; 2006&ndash;2012 Lua.org, PUC-Rio.
Freely available under the terms of the Lua license.
{preface}
{body}
</body>
</html>
"""


def rep(text, old, new):
    if text.count(old) != 1:
        sys.exit("manual.py: stock text does not match exactly once: %r" % old[:60])
    return text.replace(old, new)


def cut(text, start, end):
    """Text from the heading starting with `start` up to the one with `end`."""
    i = text.index(start)
    j = text.index(end, i + 1) if end else len(text)
    return text[i:j]


def tidy(body, page_anchors):
    """Drops blank runs and unwraps links whose anchor is not on this page."""
    def unlink(m):
        return m.group(2) if m.group(1) not in page_anchors else m.group(0)
    body = re.sub(r'<a href="#([^"]+)">(.*?)</a>', unlink, body, flags=re.S)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip() + "\n"


# ---------------------------------------------------------------- zero-index

def zero_strings(s):
    s = rep(s, """When indexing a string in Lua, the first character is at position&nbsp;1
(not at&nbsp;0, as in C).
Indices are allowed to be negative and are interpreted as indexing backwards,
from the end of the string.
Thus, the last character is at position -1, and so on.""",
"""When indexing a string in this dialect, the first character is at position&nbsp;0
(as in C): a non-negative position <em>p</em> names the character that
has <em>p</em> characters before it.
Indices are allowed to be negative and are interpreted as indexing backwards,
from the end of the string.
Thus, the last character is at position -1, the one before it at -2, and so on.
The last character can therefore be named either by <code>#s</code>
(see <a href="#2.5.5">&sect;2.5.5</a>) or by&nbsp;-1.
Positions returned by <code>find</code> and position captures are 0-based too.""")
    s = rep(s, """The default value for <code>i</code> is&nbsp;1;
the default value for <code>j</code> is&nbsp;<code>i</code>.""",
"""Positions follow the rules of this section: the first character is at position&nbsp;0
and negative positions count from the end.
The default value for <code>i</code> is&nbsp;0 (the first character);
the default value for <code>j</code> is&nbsp;<code>i</code>.
Positions outside the string are clamped, and if <code>i</code> is greater than <code>j</code>
(after both are resolved) no value is returned.
For instance, <code>string.byte("hello")</code> returns 104,
<code>string.byte("hello", -1)</code> returns 111,
and <code>string.byte("hello", 1, 3)</code> returns 101, 108, 108.""")
    s = rep(s, """If it finds a match, then <code>find</code> returns the indices of&nbsp;<code>s</code>
where this occurrence starts and ends;
otherwise, it returns <b>nil</b>.
A third, optional numerical argument <code>init</code> specifies
where to start the search;
its default value is&nbsp;1 and can be negative.""",
"""If it finds a match, then <code>find</code> returns the 0-based indices of&nbsp;<code>s</code>
where this occurrence starts and ends (both inclusive);
an empty match at position <em>p</em> returns <em>p</em> and <em>p</em>-1;
if there is no match, it returns <b>nil</b>.
A third, optional numerical argument <code>init</code> specifies
the 0-based position where to start the search;
its default value is&nbsp;0 and can be negative (counting from the end, as for <code>sub</code>).
An <code>init</code> beyond the end of the string is treated as the end of the string.
For instance, <code>string.find("hello", "l")</code> returns 2, 2;
<code>string.find("hello", "l", 3)</code> returns 3, 3;
<code>string.find("hello", "h")</code> returns 0, 0;
and <code>string.find("hello", "")</code> returns 0, -1.""")
    s = rep(s, """A third, optional numerical argument <code>init</code> specifies
where to start the search;
its default value is&nbsp;1 and can be negative.""",
"""A third, optional numerical argument <code>init</code> specifies
the 0-based position where to start the search;
its default value is&nbsp;0 and can be negative.
Position captures in the result are 0-based (see <a href="#5.4.1">&sect;5.4.1</a>).""")
    s = rep(s, """Receives a string and returns its length.
The empty string <code>""</code> has length 0.
Embedded zeros are counted,
so <code>"a\\000bc\\000"</code> has length 5.""",
"""Receives a string and returns its length, the number of bytes.
The empty string <code>""</code> has length 0.
Embedded zeros are counted,
so <code>"a\\000bc\\000"</code> has length 5.
Note that this is unchanged, whereas the length operator <code>#</code>
on a string yields the position of the last character, which is one less
(see <a href="#2.5.5">&sect;2.5.5</a>):
<code>#"abc"</code> is 2 while <code>string.len("abc")</code> is 3.""")
    s = rep(s, """Returns the substring of <code>s</code> that
starts at <code>i</code>  and continues until <code>j</code>;
<code>i</code> and <code>j</code> can be negative.
If <code>j</code> is absent, then it is assumed to be equal to -1
(which is the same as the string length).
In particular,
the call <code>string.sub(s,1,j)</code> returns a prefix of <code>s</code>
with length <code>j</code>,
and <code>string.sub(s, -i)</code> returns a suffix of <code>s</code>
with length <code>i</code>.""",
"""Returns the substring of <code>s</code> that
starts at position <code>i</code> and continues until position <code>j</code>,
both inclusive; positions are 0-based
and <code>i</code> and <code>j</code> can be negative.
If <code>j</code> is absent, then it is assumed to be equal to -1
(which is the last character).
In particular,
the call <code>string.sub(s, 0, j)</code> returns a prefix of <code>s</code>
with length <code>j+1</code>
(and <code>string.sub(s, 0, j-1)</code> one with length <code>j</code>),
and <code>string.sub(s, -i)</code> returns a suffix of <code>s</code>
with length <code>i</code>.
If <code>i</code> is before the first character it is clamped to the first one,
if <code>j</code> is past the last character it is clamped to the last one,
and if <code>i</code> is after <code>j</code> the result is the empty string.
For instance, <code>string.sub("hello", 0, 1)</code> is <code>"he"</code>,
<code>string.sub("hello", 1)</code> is <code>"ello"</code>,
<code>string.sub("hello", 0, -2)</code> is <code>"hell"</code>,
<code>string.sub("hello", -3)</code> is <code>"llo"</code>,
and <code>string.sub("hello", 5)</code> is <code>""</code>.""")
    s = rep(s, """As a special case, the empty capture <code>()</code> captures
the current string position (a number).
For instance, if we apply the pattern <code>"()aa()"</code> on the
string <code>"flaaap"</code>, there will be two captures: 3&nbsp;and&nbsp;5.""",
"""As a special case, the empty capture <code>()</code> captures
the current string position (a number), counted from&nbsp;0:
the position of the first character of the subject is&nbsp;0 and
the position after the last character is <code>string.len(s)</code>.
For instance, if we apply the pattern <code>"()aa()"</code> on the
string <code>"flaaap"</code>, there will be two captures: 2&nbsp;and&nbsp;4.""")
    return s


def zero_operators(text):
    s = cut(text, "<h3>2.5.5 ", "<h3>2.5.6 ")
    s = rep(s, """The length of a string is its number of bytes
(that is, the usual meaning of string length when each
character is one byte).""",
"""The length of a string <code>s</code> is the position of its last character,
that is, its number of bytes minus one
(characters are numbered from&nbsp;0 in this dialect, see <a href="#5.4">&sect;5.4</a>).
The empty string has length -1, <code>#"abc"</code> is 2,
and <code>#s</code> equals <code>string.len(s) - 1</code>.
In particular, <code>string.sub(s, 0, #s)</code> is <code>s</code> itself and
<code>string.sub(s, #s)</code> is its last character.""")
    s2 = cut(text, '<li><b>"len":</b>', '<li><b>"eq":</b>')
    s2 = rep(s2, "return strlen(op)         -- primitive string length",
             "return strlen(op) - 1     -- primitive string length: last position")
    s2 = rep(s2, "See <a href=\"#2.5.5\">&sect;2.5.5</a> for a description of the length of a table.",
             "See <a href=\"#2.5.5\">&sect;2.5.5</a> for a description of the length of a string and of a table.")
    return ("<h2>2.5 &ndash; <a name=\"2.5\">Expressions</a>: the length operator</h2>\n" + s +
            "\n<h2>2.8 &ndash; <a name=\"2.8\">Metatables</a>: the \"len\" event</h2>\n<ul>\n" + s2 + "</ul>\n")


# --------------------------------------------------------------- plus-concat

def plus_operators(text):
    tokens = cut(text, "<p>\nThe following strings denote other tokens:", "<p>\n<em>Literal strings</em>")
    tokens = rep(tokens, "     ;     :     ,     .     ..    ...", "     ;     :     ,     .     ...")
    tokens = tokens.rstrip() + """
<p>
The two-dot sequence <code>..</code> is not a token in this dialect:
the lexer rejects it with a syntax error.
A numeral such as <code>1..2</code> is a malformed number, as in standard Lua.
Spell string concatenation with <code>+</code> (see <a href="#2.5.4">&sect;2.5.4</a>).
"""
    coercion = cut(text, '<h3>2.2.1 ', "<h2>2.3 ")
    coercion = rep(coercion, """Any arithmetic operation applied to a string tries to convert
this string to a number, following the usual conversion rules.
Conversely, whenever a number is used where a string is expected,
the number is converted to a string, in a reasonable format.""",
"""Any arithmetic operation applied to a string and a number tries to convert
the string to a number, following the usual conversion rules.
(Addition of two strings is not arithmetic: it concatenates them, see <a href="#2.5.4">&sect;2.5.4</a>.)
Conversely, whenever a number is used where a string is expected by a library
function, the number is converted to a string, in a reasonable format.
The <code>+</code> operator never converts a number to a string.""")
    binops = cut(text, "<p>\nBinary operators comprise", "<p>\nBoth function calls")
    binops = rep(binops, "and the concatenation operator (see <a href=\"#2.5.4\">&sect;2.5.4</a>).",
                 "and string concatenation, which is spelled <code>+</code> (see <a href=\"#2.5.4\">&sect;2.5.4</a>).")
    arith = cut(text, "<h3>2.5.1 ", "<h3>2.5.2 ")
    arith = rep(arith, """If the operands are numbers, or strings that can be converted to
numbers (see <a href="#2.2.1">&sect;2.2.1</a>),
then all operations have the usual meaning.""",
"""If the operands are numbers, or strings that can be converted to
numbers (see <a href="#2.2.1">&sect;2.2.1</a>),
then all operations have the usual meaning.
The exception is the binary <code>+</code> when <em>both</em> operands are strings:
it is then string concatenation (see <a href="#2.5.4">&sect;2.5.4</a>), even if the strings look like numbers.
If exactly one operand is a string and the other is a number, <code>+</code>
is addition and the string is converted to a number:
<code>"10" + 1</code> is 11, and <code>"abc" + 1</code> raises an error.""")
    concat = cut(text, "<h3>2.5.4 ", "<h3>2.5.5 ")
    concat = rep(concat, "Concatenation</a>", "String Concatenation</a>")
    concat = rep(concat, """The string concatenation operator in Lua is
denoted by two dots ('<code>..</code>').
If both operands are strings or numbers, then they are converted to
strings according to the rules mentioned in <a href="#2.2.1">&sect;2.2.1</a>.
Otherwise, the "concat" metamethod is called (see <a href="#2.8">&sect;2.8</a>).""",
"""The string concatenation operator in this dialect is
the binary operator '<code>+</code>'.
If both operands are values of type string, the result is a new string:
the first operand followed by the second.
No conversion takes place; in particular, numeric-looking strings are
concatenated, not added: <code>"1" + "2"</code> is <code>"12"</code>.
If either operand is a number, the operation is the arithmetic addition of
<a href="#2.5.1">&sect;2.5.1</a>: <code>1 + "2"</code> is 3 and <code>"1" + 2</code> is 3.
To concatenate a number, convert it first, for instance with <code>tostring</code>:
<code>"n=" + tostring(5)</code> is <code>"n=5"</code>.
The operator is left associative, so <code>"a" + "b" + "c"</code> is <code>"abc"</code>.
There is no operator '<code>..</code>' (it is a syntax error) and the "concat" metamethod
is never called; the "add" metamethod (see <a href="#2.8">&sect;2.8</a>) is called when neither the
concatenation nor the arithmetic rule applies.""")
    prec = cut(text, "<h3>2.5.6 ", "<h3>2.5.7 ")
    prec = rep(prec, "     ..\n     +     -\n", "     +     -\n")
    prec = rep(prec, """The concatenation ('<code>..</code>') and exponentiation ('<code>^</code>')
operators are right associative.
All other binary operators are left associative.""",
"""The exponentiation operator ('<code>^</code>') is right associative.
All other binary operators, including '<code>+</code>' (both addition and
concatenation) and '<code>-</code>', are left associative
and have the same priority.
For instance, <code>"a" + "b" - 1</code> is <code>("a" + "b") - 1</code>,
and <code>1 + 2 + "x"</code> is <code>(1 + 2) + "x"</code>, an arithmetic error.""")
    meta = cut(text, '<li><b>"add":</b>', '<li><b>"sub":</b>')
    meta = rep(meta, """the <code>+</code> operation.""",
"""the <code>+</code> operation, when it is not string concatenation.""")
    meta = rep(meta, """       if o1 and o2 then  -- both operands are numeric?""",
"""       if type(op1) == "string" and type(op2) == "string" then
         return primitive_concat(op1, op2)  -- no handler is consulted
       elseif o1 and o2 then  -- both operands are numeric?""")
    out = ["<h2>2.1 &ndash; <a name=\"2.1\">Lexical Conventions</a>: tokens</h2>\n" + tokens,
           "<h2>2.2 &ndash; <a name=\"2.2\">Values and Types</a>: coercion</h2>\n" + coercion,
           "<h2>2.5 &ndash; <a name=\"2.5\">Expressions</a>: operators</h2>\n" + binops + arith + concat + prec,
           "<h2>2.8 &ndash; <a name=\"2.8\">Metatables</a>: the \"add\" and \"concat\" events</h2>\n<ul>\n" + meta + """
<li><b>"concat":</b>
there is no "concat" event in this dialect: no operator triggers it, and a
<code>__concat</code> field of a metatable is never consulted.
String concatenation is the "add" operation on two strings.
</li>
</ul>
"""]
    return "\n".join(out)


def plus_tables(s):
    return rep(s, """returns <code>table[i]..sep..table[i+1] &middot;&middot;&middot; sep..table[j]</code>.""",
"""returns the string made of <code>table[i]</code>, <code>sep</code>, <code>table[i+1]</code>,
&middot;&middot;&middot;, <code>sep</code>, <code>table[j]</code> joined in this order
(numbers are converted to strings; this function does not use the <code>+</code> operator).""")


def main():
    if len(sys.argv) != 4 or sys.argv[2] not in DIALECTS:
        sys.exit("usage: manual.py STOCK_MANUAL_HTML {%s} OUTDIR" % "|".join(DIALECTS))
    stock = Path(sys.argv[1]).read_bytes().decode("latin-1")
    dialect, out = sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)

    strings = cut(stock, "<h2>5.4 ", "<h2>5.5 ")
    tables = cut(stock, "<h2>5.5 ", "<h2>5.6 ")
    maths = cut(stock, "<h2>5.6 ", "<h2>5.7 ")
    if dialect == "zero-index":
        strings = zero_strings(strings)
        operators = zero_operators(stock)
    else:
        tables = plus_tables(tables)
        operators = plus_operators(stock)

    pages = {"strings": ("String Manipulation (&sect;5.4)", strings),
             "tables": ("Table Manipulation (&sect;5.5)", tables),
             "math": ("Mathematical Functions (&sect;5.6)", maths),
             "operators": ("Operators and Changed Language Rules (chapter 2)", operators)}
    for name, (title, body) in pages.items():
        anchors = set(re.findall(r'<a name="([^"]+)"', body))
        text = PAGE.format(dialect=dialect, title=title, preface=PREFACE[dialect],
                           body=tidy(body, anchors))
        (out / (name + ".html")).write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
