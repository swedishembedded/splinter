# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from mdrender.tables import (
    mdr_column_widths,
    mdr_fit_cell,
    mdr_fit_widths,
    mdr_parse_alignment_row,
    mdr_parse_table,
    mdr_render_table_text,
    mdr_select_columns,
    mdr_sort_rows,
    mdr_split_table_row,
)


def sample():
    return mdr_parse_table(["| Name | Qty |", "|:--|--:|", "| apple | 10 |", "| kiwi | 2 |"])


class C0(unittest.TestCase):
    def test_tight_pipes_without_spaces_split_cleanly(self):
        self.assertEqual(mdr_split_table_row("|a|b|"), ["a", "b"])

    def test_only_a_leading_pipe_is_removed(self):
        self.assertEqual(mdr_split_table_row("| a | b"), ["a", "b"])

    def test_final_escaped_pipe_is_content(self):
        self.assertEqual(mdr_split_table_row("| a | b\\|"), ["a", "b|"])


class C1(unittest.TestCase):
    def test_cell_with_other_characters_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_parse_alignment_row(["---", "-x-"])
        self.assertEqual(str(caught.exception), "invalid alignment cell: -x-")

    def test_empty_cell_is_rejected(self):
        with self.assertRaises(ValueError):
            mdr_parse_alignment_row([""])


class C2(unittest.TestCase):
    def test_single_line_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_parse_table(["a|b"])
        self.assertEqual(str(caught.exception), "table needs a header and an alignment row")

    def test_mismatched_alignment_count_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_parse_table(["a|b", "-"])
        self.assertEqual(str(caught.exception), "header and alignment row differ in column count")


class C3(unittest.TestCase):
    def test_empty_column_still_gets_one_character(self):
        self.assertEqual(mdr_column_widths([[""]]), [1])

    def test_maximum_equal_to_the_minimum_is_accepted(self):
        self.assertEqual(mdr_column_widths([["abcd"]], min_width=2, max_width=2), [2])

    def test_ragged_rows_still_measure_every_column(self):
        self.assertEqual(mdr_column_widths([["a"], ["b", "cccc"]]), [1, 4])

    def test_columns_never_shrink_below_one(self):
        self.assertEqual(mdr_fit_widths([5, 5], 9), [1, 1])

    def test_grid_that_cannot_fit_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_fit_widths([2, 2], 8)
        self.assertEqual(str(caught.exception), "table cannot fit in 8 columns")


class C4(unittest.TestCase):
    def test_ellipsis_sized_width_cuts_hard(self):
        self.assertEqual(mdr_fit_cell("abcdefg", 3), "abc")

    def test_narrow_width_cuts_hard(self):
        self.assertEqual(mdr_fit_cell("abcdefg", 1), "a")


class C5(unittest.TestCase):
    def test_columns_are_padded_and_aligned(self):
        self.assertEqual(
            mdr_render_table_text(sample()),
            "| Name  | Qty |\n|-------|----:|\n| apple |  10 |\n| kiwi  |   2 |",
        )

    def test_center_column_uses_colons_on_both_sides(self):
        table = mdr_parse_table(["a|b", ":-:|-", "x|y"])
        self.assertEqual(mdr_render_table_text(table), "|  a  | b   |\n|:---:|-----|\n|  x  | y   |")

    def test_narrow_columns_are_widened_to_three(self):
        table = mdr_parse_table(["a", "-", "b"])
        self.assertEqual(mdr_render_table_text(table), "| a   |\n|-----|\n| b   |")

    def test_empty_body_renders_header_and_separator(self):
        table = mdr_parse_table(["abc|d", "-|-"])
        self.assertEqual(mdr_render_table_text(table), "| abc | d   |\n|-----|-----|")

    def test_max_width_shrinks_and_truncates_columns(self):
        table = mdr_parse_table(["name", "--:", "abcdefgh"])
        self.assertEqual(mdr_render_table_text(table, max_width=9), "|  name |\n|------:|\n| ab... |")

    def test_rendered_text_respects_max_width(self):
        table = mdr_parse_table(["alpha|beta", "-|-", "gamma delta|epsilon"])
        widest = max(len(line) for line in mdr_render_table_text(table, 20).splitlines())
        self.assertEqual(widest, 20)

    def test_rendered_text_can_be_parsed_back(self):
        table = sample()
        self.assertEqual(mdr_parse_table(mdr_render_table_text(table).splitlines()), table)


class C6(unittest.TestCase):
    def test_selected_columns_follow_the_given_order(self):
        selected = mdr_select_columns(sample(), [1, 0])
        self.assertEqual(selected, {"header": ["Qty", "Name"], "align": ["right", "left"],
                                    "rows": [["10", "apple"], ["2", "kiwi"]]})

    def test_one_column_can_be_selected(self):
        self.assertEqual(mdr_select_columns(sample(), [0])["header"], ["Name"])

    def test_index_past_the_last_column_is_rejected(self):
        with self.assertRaises(IndexError) as caught:
            mdr_select_columns(sample(), [2])
        self.assertEqual(str(caught.exception), "column index out of range: 2")

    def test_numeric_column_sorts_by_value(self):
        self.assertEqual([r[1] for r in mdr_sort_rows(sample(), 1)["rows"]], ["2", "10"])

    def test_text_column_sorts_alphabetically(self):
        self.assertEqual([r[0] for r in mdr_sort_rows(sample(), 0)["rows"]], ["apple", "kiwi"])

    def test_descending_order_reverses_the_sort(self):
        self.assertEqual([r[1] for r in mdr_sort_rows(sample(), 1, descending=True)["rows"]], ["10", "2"])

    def test_negative_numbers_sort_numerically(self):
        table = mdr_parse_table(["n", "-", "3", "-12", "0"])
        self.assertEqual([r[0] for r in mdr_sort_rows(table, 0)["rows"]], ["-12", "0", "3"])

    def test_mixed_column_sorts_as_text(self):
        table = mdr_parse_table(["n", "-", "10", "9", "x"])
        self.assertEqual([r[0] for r in mdr_sort_rows(table, 0)["rows"]], ["10", "9", "x"])

    def test_sorting_a_missing_column_is_rejected(self):
        with self.assertRaises(IndexError) as caught:
            mdr_sort_rows(sample(), 5)
        self.assertEqual(str(caught.exception), "column index out of range: 5")


if __name__ == "__main__":
    unittest.main()
