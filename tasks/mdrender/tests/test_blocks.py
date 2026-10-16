# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from mdrender.blocks import (
    mdr_count_blocks,
    mdr_match_atx_heading,
    mdr_match_code_fence,
    mdr_match_list_marker,
    mdr_parse_blocks,
    mdr_walk_blocks,
)


def para(value):
    return {"type": "paragraph", "children": [{"type": "text", "text": value}]}


def bullets(*items, tight=True):
    return {"type": "list", "ordered": False, "start": None, "tight": tight, "items": list(items)}


def numbers(start, *items, tight=True):
    return {"type": "list", "ordered": True, "start": start, "tight": tight, "items": list(items)}


class C0(unittest.TestCase):
    def test_surrounding_blanks_are_trimmed(self):
        self.assertEqual(mdr_match_atx_heading("#   spaced out   "), (1, "spaced out"))


class C1(unittest.TestCase):
    def test_tilde_fence_info_allows_backtick(self):
        self.assertEqual(mdr_match_code_fence("~~~a`b"), ("~", 3, "a`b", 0))


class C2(unittest.TestCase):
    def test_lone_marker_is_an_empty_item(self):
        self.assertEqual(mdr_match_list_marker("-"),
                         {"ordered": False, "kind": "-", "number": None, "offset": 2, "text": ""})

    def test_four_marker_spaces_set_offset(self):
        self.assertEqual(mdr_match_list_marker("-    x")["offset"], 5)

    def test_five_marker_spaces_count_as_one(self):
        self.assertEqual(mdr_match_list_marker("-     x")["offset"], 2)


class C3(unittest.TestCase):
    def test_heading_carries_level_and_inline_children(self):
        self.assertEqual(mdr_parse_blocks("## A *b*"), [{"type": "heading", "level": 2, "children": [
            {"type": "text", "text": "A "}, {"type": "emph", "children": [{"type": "text", "text": "b"}]}]}])

    def test_rule_interrupts_a_paragraph(self):
        self.assertEqual([b["type"] for b in mdr_parse_blocks("a\n---")], ["paragraph", "rule"])

    def test_dash_rule_wins_over_a_list_marker(self):
        self.assertEqual(mdr_parse_blocks("- - -\n* * *"), [{"type": "rule"}, {"type": "rule"}])

    def test_tab_indented_fence_is_not_a_fence(self):
        self.assertEqual(mdr_parse_blocks("\t```\nx"), [{"type": "paragraph", "children": [
            {"type": "text", "text": "```"}, {"type": "softbreak"}, {"type": "text", "text": "x"}]}])

    def test_tab_in_item_code_counts_four(self):
        self.assertEqual(mdr_parse_blocks("- ```\n\tx\n  ```"),
                         [bullets([{"type": "code", "info": "", "text": "  x"}])])

    def test_deep_marker_line_ends_the_list(self):
        self.assertEqual([b["type"] for b in mdr_parse_blocks("-    a\n\t- b")], ["list", "paragraph"])


class C4(unittest.TestCase):
    def test_unclosed_fence_runs_to_the_end(self):
        self.assertEqual(mdr_parse_blocks("```\na\nb"), [{"type": "code", "info": "", "text": "a\nb"}])

    def test_longer_closing_fence_closes(self):
        self.assertEqual(mdr_parse_blocks("```\na\n`````\nb"), [
            {"type": "code", "info": "", "text": "a"}, para("b")])

    def test_quote_wraps_parsed_children(self):
        self.assertEqual(mdr_parse_blocks("> # h\n> text"), [{"type": "quote", "children": [
            {"type": "heading", "level": 1, "children": [{"type": "text", "text": "h"}]}, para("text")]}])

    def test_blank_line_ends_a_quote(self):
        self.assertEqual([b["type"] for b in mdr_parse_blocks("> a\n\n> b")], ["quote", "quote"])

    def test_nested_quote_is_parsed_recursively(self):
        self.assertEqual(mdr_parse_blocks("> > a"), [{"type": "quote", "children": [
            {"type": "quote", "children": [para("a")]}]}])

    def test_quote_interrupts_a_paragraph(self):
        self.assertEqual([b["type"] for b in mdr_parse_blocks("a\n> b")], ["paragraph", "quote"])


class C5(unittest.TestCase):
    def test_blank_between_items_loosens_list(self):
        self.assertEqual(mdr_parse_blocks("- a\n\n- b"), [bullets([para("a")], [para("b")], tight=False)])

    def test_blank_inside_item_loosens_list(self):
        self.assertEqual(mdr_parse_blocks("- a\n\n  b\n- c"),
                         [bullets([para("a"), para("b")], [para("c")], tight=False)])

    def test_nested_list_joins_parent_item(self):
        self.assertEqual(mdr_parse_blocks("- a\n  - b\n- c"),
                         [bullets([para("a"), bullets([para("b")])], [para("c")])])

    def test_nested_ordered_list_inside_a_bullet(self):
        self.assertEqual(mdr_parse_blocks("- a\n  1. b"),
                         [bullets([para("a"), numbers(1, [para("b")])])])

    def test_lazy_continuation_joins_the_item_paragraph(self):
        blocks = mdr_parse_blocks("- a\nb")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["items"][0][0]["children"][2], {"type": "text", "text": "b"})

    def test_lazy_line_needs_preceding_text(self):
        blocks = mdr_parse_blocks("-\n  x\ny")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["items"][0][0]["children"][2], {"type": "text", "text": "y"})

    def test_lazy_continuation_stops_at_a_heading(self):
        self.assertEqual([b["type"] for b in mdr_parse_blocks("- a\n# h")], ["list", "heading"])

    def test_new_bullet_character_splits_list(self):
        self.assertEqual(mdr_parse_blocks("- a\n* b"), [bullets([para("a")]), bullets([para("b")])])

    def test_ordered_item_starting_at_one_interrupts_text(self):
        self.assertEqual([b["type"] for b in mdr_parse_blocks("a\n1. b")], ["paragraph", "list"])

    def test_empty_item_has_no_blocks(self):
        self.assertEqual(mdr_parse_blocks("-\n- x"), [bullets([], [para("x")])])

    def test_dedent_after_blank_ends_nested_item(self):
        self.assertEqual(mdr_parse_blocks("- a\n  - b\n\n  c"),
                         [bullets([para("a"), bullets([para("b")]), para("c")], tight=False)])


class C6(unittest.TestCase):
    def test_walk_visits_nested_blocks_depth_first(self):
        blocks = mdr_parse_blocks("> a\n\n- b\n  - c\n\n---")
        kinds = [b["type"] for b in mdr_walk_blocks(blocks)]
        self.assertEqual(kinds, ["quote", "paragraph", "list", "paragraph", "list", "paragraph", "rule"])

    def test_counts_group_by_type_first_seen(self):
        blocks = mdr_parse_blocks("# h\n\n- a\n- b\n\n> q\n\n## k")
        self.assertEqual(mdr_count_blocks(blocks), {"heading": 2, "list": 1, "paragraph": 3, "quote": 1})


if __name__ == "__main__":
    unittest.main()
