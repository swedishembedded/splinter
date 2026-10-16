# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from mdrender.text_render import mdr_render_inline_text, mdr_render_text


def text(value):
    return {"type": "text", "text": value}


def para(value):
    return {"type": "paragraph", "children": [text(value)]}


def heading(level, value):
    return {"type": "heading", "level": level, "children": [text(value)]}


def link(href, *children):
    return {"type": "link", "href": href, "title": None, "children": list(children)}


def bullets(*items, tight=True):
    return {"type": "list", "ordered": False, "start": None, "tight": tight, "items": list(items)}


def numbers(start, *items, tight=True):
    return {"type": "list", "ordered": True, "start": start, "tight": tight, "items": list(items)}


def quote(*children):
    return {"type": "quote", "children": list(children)}


class C0(unittest.TestCase):
    def test_emphasis_is_marked_with_underscores(self):
        self.assertEqual(mdr_render_inline_text([{"type": "emph", "children": [text("x")]}]), "_x_")

    def test_strong_is_marked_with_asterisks(self):
        self.assertEqual(mdr_render_inline_text([{"type": "strong", "children": [text("x")]}]), "*x*")

    def test_code_is_marked_with_backticks(self):
        self.assertEqual(mdr_render_inline_text([{"type": "code", "text": "x"}]), "`x`")

    def test_link_shows_its_destination_in_parentheses(self):
        self.assertEqual(mdr_render_inline_text([link("http://u.org", text("site"))]), "site (http://u.org)")

    def test_link_equal_to_label_shown_once(self):
        self.assertEqual(mdr_render_inline_text([link("http://u.org", text("http://u.org"))]), "http://u.org")

    def test_unknown_node_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_render_inline_text([{"type": "blink"}])
        self.assertEqual(str(caught.exception), "unknown node type: blink")


class C1(unittest.TestCase):
    def test_default_width_is_seventy_two(self):
        self.assertEqual(mdr_render_text([para("a" * 35 + " " + "b" * 36)]), "a" * 35 + " " + "b" * 36)
        self.assertEqual(mdr_render_text([para("a" * 36 + " " + "b" * 36)]), "a" * 36 + "\n" + "b" * 36)

    def test_hard_break_forces_a_new_line(self):
        block = {"type": "paragraph", "children": [text("a"), {"type": "linebreak"}, text("b")]}
        self.assertEqual(mdr_render_text([block], 20), "a\nb")

    def test_hard_break_sides_wrap_separately(self):
        block = {"type": "paragraph", "children": [text("aa bb"), {"type": "linebreak"}, text("cc dd")]}
        self.assertEqual(mdr_render_text([block], 4), "aa\nbb\ncc\ndd")

    def test_unknown_block_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_render_text([{"type": "table"}], 10)
        self.assertEqual(str(caught.exception), "unknown block type: table")


class C2(unittest.TestCase):
    def test_top_heading_is_underlined_with_equals(self):
        self.assertEqual(mdr_render_text([heading(1, "Title")], 20), "Title\n=====")

    def test_second_level_heading_is_underlined_with_dashes(self):
        self.assertEqual(mdr_render_text([heading(2, "Sub")], 20), "Sub\n---")

    def test_third_level_heading_gets_hash_marks(self):
        self.assertEqual(mdr_render_text([heading(3, "Deep")], 20), "### Deep")

    def test_wrapped_hash_heading_indents_its_continuation(self):
        self.assertEqual(mdr_render_text([heading(3, "aaa bbb ccc")], 9), "### aaa\n    bbb\n    ccc")

    def test_underline_matches_the_longest_wrapped_line(self):
        self.assertEqual(mdr_render_text([heading(1, "aa bbbb c")], 6), "aa\nbbbb c\n======")

    def test_rule_spans_the_width(self):
        self.assertEqual(mdr_render_text([{"type": "rule"}], 5), "-----")


class C3(unittest.TestCase):
    def test_code_lines_are_never_wrapped(self):
        self.assertEqual(mdr_render_text([{"type": "code", "info": "", "text": "x" * 30}], 10), "    " + "x" * 30)

    def test_quote_lines_start_with_a_marker(self):
        self.assertEqual(mdr_render_text([quote(para("a b"))], 20), "> a b")

    def test_quote_text_wraps_inside_the_marker(self):
        self.assertEqual(mdr_render_text([quote(para("aaa bbb"))], 5), "> aaa\n> bbb")

    def test_quote_without_room_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_render_text([quote(para("a"))], 2)
        self.assertEqual(str(caught.exception), "indent leaves no room for text")


class C4(unittest.TestCase):
    def test_long_item_wraps_with_a_hanging_indent(self):
        self.assertEqual(mdr_render_text([bullets([para("aaa bbb ccc")])], 9), "* aaa bbb\n  ccc")

    def test_nested_bullets_alternate_their_markers(self):
        deepest = bullets([para("d")])
        third = bullets([para("c"), deepest])
        second = bullets([para("b"), third])
        text_out = mdr_render_text([bullets([para("a"), second])], 30)
        self.assertEqual(text_out, "* a\n  - b\n    + c\n      * d")

    def test_ordered_items_count_up_from_the_start(self):
        self.assertEqual(mdr_render_text([numbers(3, [para("a")], [para("b")])], 20), "3. a\n4. b")

    def test_ordered_markers_are_padded_to_the_widest(self):
        self.assertEqual(mdr_render_text([numbers(9, [para("a")], [para("b")])], 20), "9.  a\n10. b")

    def test_wrapped_ordered_item_aligns_under_its_text(self):
        self.assertEqual(mdr_render_text([numbers(1, [para("aaa bbb")])], 6), "1. aaa\n   bbb")

    def test_loose_list_separates_items_with_blank_lines(self):
        self.assertEqual(mdr_render_text([bullets([para("a")], [para("b")], tight=False)], 20), "* a\n\n* b")

    def test_loose_item_blocks_split_by_blank(self):
        self.assertEqual(mdr_render_text([bullets([para("a"), para("b")], tight=False)], 20), "* a\n\n  b")

    def test_tight_item_keeps_its_blocks_together(self):
        self.assertEqual(mdr_render_text([bullets([para("a"), para("b")])], 20), "* a\n  b")

    def test_code_inside_an_item_is_indented_further(self):
        item = [para("a"), {"type": "code", "info": "", "text": "x\n\ny"}]
        self.assertEqual(mdr_render_text([bullets(item)], 20), "* a\n      x\n\n      y")

    def test_list_between_paragraphs_gets_blank_gaps(self):
        self.assertEqual(mdr_render_text([para("a"), bullets([para("b")]), para("c")], 20), "a\n\n* b\n\nc")


if __name__ == "__main__":
    unittest.main()
