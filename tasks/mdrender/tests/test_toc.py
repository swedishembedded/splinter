# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from mdrender.toc import (
    mdr_assign_heading_ids,
    mdr_build_toc,
    mdr_collect_headings,
    mdr_find_toc_entry,
    mdr_node_plain_text,
    mdr_number_toc,
    mdr_render_toc_text,
    mdr_slugify,
    mdr_toc_depth,
    mdr_toc_to_list_block,
    mdr_unique_slug,
)


def text(value):
    return {"type": "text", "text": value}


def heading(level, value):
    return {"type": "heading", "level": level, "children": [text(value)]}


def entry(level, value, slug):
    return {"level": level, "text": value, "id": slug}


class SlugTests(unittest.TestCase):
    def test_digits_survive_in_a_slug(self):
        self.assertEqual(mdr_slugify("Step 2 of 10"), "step-2-of-10")

    def test_text_without_usable_characters_becomes_section(self):
        self.assertEqual(mdr_slugify("!!!"), "section")

    def test_empty_text_becomes_section(self):
        self.assertEqual(mdr_slugify(""), "section")

    def test_nine_is_a_slug_character(self):
        self.assertEqual(mdr_slugify("Top 9 tips"), "top-9-tips")


class PlainTextTests(unittest.TestCase):
    def test_nested_formatting_is_flattened(self):
        inner = {"type": "emph", "children": [text("x")]}
        self.assertEqual(mdr_node_plain_text([{"type": "strong", "children": [text("a"), inner]}]), "ax")


class UniqueSlugTests(unittest.TestCase):
    def test_taken_slug_gets_the_suffix_one(self):
        self.assertEqual(mdr_unique_slug("intro", {"intro"}), "intro-1")

    def test_suffix_counts_up_past_taken_ones(self):
        self.assertEqual(mdr_unique_slug("a", {"a", "a-1", "a-2"}), "a-3")

    def test_suffixed_result_is_recorded_too(self):
        used = {"a"}
        mdr_unique_slug("a", used)
        self.assertEqual(used, {"a", "a-1"})

    def test_natural_slug_that_looks_suffixed_is_respected(self):
        used = set()
        first = mdr_unique_slug("a-1", used)
        second = mdr_unique_slug("a", used)
        third = mdr_unique_slug("a", used)
        self.assertEqual([first, second, third], ["a-1", "a", "a-2"])


class AssignIdTests(unittest.TestCase):
    def test_headings_receive_slug_ids(self):
        blocks = mdr_assign_heading_ids([heading(1, "Hello There")])
        self.assertEqual(blocks[0]["id"], "hello-there")

    def test_repeated_titles_receive_numbered_ids(self):
        blocks = mdr_assign_heading_ids([heading(2, "Notes"), heading(2, "Notes"), heading(3, "Notes")])
        self.assertEqual([b["id"] for b in blocks], ["notes", "notes-1", "notes-2"])

    def test_input_blocks_are_not_modified(self):
        original = heading(1, "A")
        mdr_assign_heading_ids([original])
        self.assertNotIn("id", original)

    def test_formatted_heading_text_is_slugged_without_markup(self):
        block = {"type": "heading", "level": 1, "children": [
            {"type": "emph", "children": [text("Big")]}, text(" idea")]}
        self.assertEqual(mdr_assign_heading_ids([block])[0]["id"], "big-idea")


class CollectTests(unittest.TestCase):
    def test_entries_list_level_text_and_id(self):
        self.assertEqual(mdr_collect_headings([heading(2, "Setup")]), [entry(2, "Setup", "setup")])

    def test_non_heading_blocks_are_skipped(self):
        blocks = [{"type": "rule"}, heading(1, "A")]
        self.assertEqual(mdr_collect_headings(blocks), [entry(1, "A", "a")])

    def test_default_levels_skip_zero_and_seven(self):
        blocks = [heading(0, "Z"), heading(1, "A"), heading(6, "F"), heading(7, "S")]
        self.assertEqual([e["id"] for e in mdr_collect_headings(blocks)], ["a", "f"])

    def test_levels_below_the_minimum_are_dropped(self):
        blocks = [heading(1, "A"), heading(2, "B")]
        self.assertEqual(mdr_collect_headings(blocks, range(2, 7)), [entry(2, "B", "b")])

    def test_levels_above_the_maximum_are_dropped(self):
        blocks = [heading(2, "B"), heading(3, "C")]
        self.assertEqual(mdr_collect_headings(blocks, range(1, 3)), [entry(2, "B", "b")])

    def test_both_range_limits_are_inclusive(self):
        blocks = [heading(1, "A"), heading(2, "B"), heading(3, "C"), heading(4, "D")]
        self.assertEqual([e["id"] for e in mdr_collect_headings(blocks, (2, 3))], ["b", "c"])

    def test_dropped_headings_still_reserve_their_ids(self):
        blocks = [heading(1, "Same"), heading(2, "Same")]
        self.assertEqual(mdr_collect_headings(blocks, [2]), [entry(2, "Same", "same-1")])


class BuildTreeTests(unittest.TestCase):
    def test_flat_levels_make_siblings(self):
        tree = mdr_build_toc([entry(1, "A", "a"), entry(1, "B", "b")])
        self.assertEqual([n["id"] for n in tree], ["a", "b"])

    def test_deeper_level_nests_under_the_previous_entry(self):
        tree = mdr_build_toc([entry(1, "A", "a"), entry(2, "B", "b"), entry(2, "C", "c")])
        self.assertEqual(len(tree), 1)
        self.assertEqual([n["id"] for n in tree[0]["children"]], ["b", "c"])

    def test_shallower_level_closes_branch(self):
        tree = mdr_build_toc([entry(1, "A", "a"), entry(2, "B", "b"), entry(1, "C", "c")])
        self.assertEqual([n["id"] for n in tree], ["a", "c"])
        self.assertEqual(tree[1]["children"], [])

    def test_skipped_level_still_nests_under_the_parent(self):
        tree = mdr_build_toc([entry(1, "A", "a"), entry(3, "B", "b")])
        self.assertEqual(tree[0]["children"][0]["id"], "b")

    def test_document_starting_deep_has_deep_roots(self):
        tree = mdr_build_toc([entry(3, "A", "a"), entry(2, "B", "b")])
        self.assertEqual([n["id"] for n in tree], ["a", "b"])

    def test_equal_level_after_child_is_sibling(self):
        tree = mdr_build_toc([entry(2, "A", "a"), entry(3, "B", "b"), entry(2, "C", "c"), entry(3, "D", "d")])
        self.assertEqual([[c["id"] for c in n["children"]] for n in tree], [["b"], ["d"]])

    def test_nodes_keep_level_text_and_id(self):
        tree = mdr_build_toc([entry(2, "A b", "a-b")])
        self.assertEqual(tree, [{"level": 2, "text": "A b", "id": "a-b", "children": []}])


class RenderTocTests(unittest.TestCase):
    def setUp(self):
        self.tree = mdr_build_toc([entry(1, "A", "a"), entry(2, "B", "b"), entry(2, "C", "c"), entry(1, "D", "d")])

    def test_numbers_follow_the_nesting(self):
        self.assertEqual([(label, depth) for label, depth, _ in mdr_number_toc(self.tree)],
                         [("1", 0), ("1.1", 1), ("1.2", 1), ("2", 0)])

    def test_numbering_continues_past_nine(self):
        tree = mdr_build_toc([entry(1, str(n), "s%d" % n) for n in range(11)])
        self.assertEqual(mdr_number_toc(tree)[10][0], "11")

    def test_text_rendering_indents_by_depth(self):
        self.assertEqual(mdr_render_toc_text(self.tree), "1 A\n  1.1 B\n  1.2 C\n2 D")

    def test_text_rendering_honours_the_indent_width(self):
        self.assertEqual(mdr_render_toc_text(self.tree, indent=4).splitlines()[1], "    1.1 B")

    def test_empty_tree_renders_no_text(self):
        self.assertEqual(mdr_render_toc_text([]), "")

    def test_list_block_links_entries_to_anchors(self):
        block = mdr_toc_to_list_block(mdr_build_toc([entry(1, "A b", "a-b")]))
        anchor = {"type": "link", "href": "#a-b", "title": None, "children": [text("A b")]}
        self.assertEqual(block, {"type": "list", "ordered": False, "start": None, "tight": True,
                                 "items": [[{"type": "paragraph", "children": [anchor]}]]})

    def test_list_block_nests_children_as_a_sublist(self):
        block = mdr_toc_to_list_block(self.tree)
        self.assertEqual(len(block["items"]), 2)
        self.assertEqual(block["items"][0][1]["type"], "list")
        self.assertEqual(len(block["items"][0][1]["items"]), 2)
        self.assertEqual(len(block["items"][1]), 1)

    def test_ordered_list_block_starts_at_one(self):
        block = mdr_toc_to_list_block(self.tree, ordered=True)
        self.assertEqual((block["ordered"], block["start"]), (True, 1))
        self.assertEqual((block["items"][0][1]["ordered"], block["items"][0][1]["start"]), (True, 1))


class DepthAndFindTests(unittest.TestCase):
    def test_flat_tree_has_depth_one(self):
        self.assertEqual(mdr_toc_depth(mdr_build_toc([entry(1, "A", "a"), entry(1, "B", "b")])), 1)

    def test_depth_follows_the_deepest_branch(self):
        entries = [entry(1, "A", "a"), entry(1, "B", "b"), entry(2, "C", "c"), entry(3, "D", "d")]
        self.assertEqual(mdr_toc_depth(mdr_build_toc(entries)), 3)

    def test_find_returns_a_top_level_entry(self):
        tree = mdr_build_toc([entry(1, "A", "a"), entry(1, "B", "b")])
        self.assertEqual(mdr_find_toc_entry(tree, "b")["text"], "B")

    def test_find_searches_nested_entries(self):
        tree = mdr_build_toc([entry(1, "A", "a"), entry(2, "B", "b"), entry(3, "C", "c")])
        self.assertEqual(mdr_find_toc_entry(tree, "c")["text"], "C")

    def test_find_gives_none_for_an_unknown_id(self):
        self.assertIsNone(mdr_find_toc_entry(mdr_build_toc([entry(1, "A", "a")]), "zzz"))


if __name__ == "__main__":
    unittest.main()
