# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from mdrender.inline import mdr_collect_code_spans, mdr_collect_links, mdr_parse_inline


def text(value):
    return {"type": "text", "text": value}


def emph(*children):
    return {"type": "emph", "children": list(children)}


def strong(*children):
    return {"type": "strong", "children": list(children)}


def code(value):
    return {"type": "code", "text": value}


def link(href, title, *children):
    return {"type": "link", "href": href, "title": title, "children": list(children)}


class C0(unittest.TestCase):
    def test_one_surrounding_space_is_trimmed(self):
        self.assertEqual(mdr_parse_inline("`` `a` ``"), [code("`a`")])

    def test_span_of_spaces_is_kept_whole(self):
        self.assertEqual(mdr_parse_inline("`  `"), [code("  ")])


class C1(unittest.TestCase):
    def test_triple_delimiters_nest_strong_in_emphasis(self):
        self.assertEqual(mdr_parse_inline("***b***"), [emph(strong(text("b")))])

    def test_strong_may_contain_emphasis(self):
        self.assertEqual(mdr_parse_inline("**a *b* c**"), [strong(text("a "), emph(text("b")), text(" c"))])

    def test_emphasis_may_contain_strong(self):
        self.assertEqual(mdr_parse_inline("*a **b** c*"), [emph(text("a "), strong(text("b")), text(" c"))])

    def test_opener_before_a_space_stays_literal(self):
        self.assertEqual(mdr_parse_inline("* a*"), [text("* a*")])

    def test_closer_after_a_space_does_not_close(self):
        self.assertEqual(mdr_parse_inline("*a *"), [text("*a *")])

    def test_intraword_asterisk_makes_emphasis(self):
        self.assertEqual(mdr_parse_inline("a*b*c"), [text("a"), emph(text("b")), text("c")])

    def test_underscore_before_letter_cannot_close(self):
        self.assertEqual(mdr_parse_inline("_a_b"), [text("_a_b")])

    def test_underscore_emphasis_before_trailing_word(self):
        self.assertEqual(mdr_parse_inline("_a_ b"), [emph(text("a")), text(" b")])

    def test_letter_then_underscore_paren_stays_literal(self):
        self.assertEqual(mdr_parse_inline("a_(b)_ c"), [text("a_(b)_ c")])

    def test_underscore_closer_needs_the_next_character_alone(self):
        self.assertEqual(mdr_parse_inline("_a_b c"), [text("_a_b c")])

    def test_escaped_delimiter_before_the_real_closer(self):
        self.assertEqual(mdr_parse_inline("*a\\**"), [emph(text("a*"))])


class C2(unittest.TestCase):
    def test_link_with_quoted_title_works(self):
        self.assertEqual(mdr_parse_inline('[t](u "the title")'), [link("u", "the title", text("t"))])

    def test_brackets_inside_code_do_not_count(self):
        self.assertEqual(mdr_parse_inline("[`]`](u)"), [link("u", None, code("]"))])

    def test_unbalanced_backtick_in_link_text_is_literal(self):
        self.assertEqual(mdr_parse_inline("[a`b](u)"), [link("u", None, text("a`b"))])

    def test_destination_may_hold_balanced_parentheses(self):
        self.assertEqual(mdr_parse_inline("[t](u(1))"), [link("u(1)", None, text("t"))])

    def test_empty_destination_is_allowed(self):
        self.assertEqual(mdr_parse_inline("[t]()"), [link("", None, text("t"))])

    def test_spaces_after_the_title_are_ignored(self):
        self.assertEqual(mdr_parse_inline('[t](u "x"  )'), [link("u", "x", text("t"))])

    def test_unclosed_link_text_stays_literal(self):
        self.assertEqual(mdr_parse_inline("[t(u)"), [text("[t(u)")])

    def test_unclosed_destination_stays_literal(self):
        self.assertEqual(mdr_parse_inline("[t](u"), [text("[t](u")])

    def test_escape_then_code_span_skipped_in_brackets(self):
        self.assertEqual(mdr_parse_inline("[\\*`]`](u)"), [link("u", None, text("*"), code("]"))])

    def test_text_right_after_a_link_is_kept(self):
        self.assertEqual(mdr_parse_inline("[t](u)x"), [link("u", None, text("t")), text("x")])

    def test_escaped_bracket_stays_inside_link_text(self):
        self.assertEqual(mdr_parse_inline("[a\\]b](u)"), [link("u", None, text("a]b"))])


class C3(unittest.TestCase):
    def test_two_trailing_spaces_make_a_hard_break(self):
        self.assertEqual(mdr_parse_inline("a  \nb"), [text("a"), {"type": "linebreak"}, text("b")])

    def test_backslash_before_newline_makes_a_hard_break(self):
        self.assertEqual(mdr_parse_inline("a\\\nb"), [text("a"), {"type": "linebreak"}, text("b")])

    def test_hard_break_after_emphasis_is_recognised(self):
        self.assertEqual(mdr_parse_inline("*a*  \nb"), [emph(text("a")), {"type": "linebreak"}, text("b")])


class C4(unittest.TestCase):
    def test_links_are_collected_in_document_order(self):
        nodes = mdr_parse_inline("[a](1) and *[b](2)* and <http://3.org>")
        self.assertEqual(mdr_collect_links(nodes), ["1", "2", "http://3.org"])

    def test_nested_link_text_is_searched(self):
        self.assertEqual(mdr_collect_links(mdr_parse_inline("[a [b](2)](1)")), ["1", "2"])

    def test_code_spans_are_collected_through_nesting(self):
        nodes = mdr_parse_inline("`a` and **`b`** and [`c`](u)")
        self.assertEqual(mdr_collect_code_spans(nodes), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
