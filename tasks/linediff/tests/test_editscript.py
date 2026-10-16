# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import unittest

from linediff.editscript import (
    linediff_change_blocks,
    linediff_common_pairs,
    linediff_edit_distance,
    linediff_edit_script,
    linediff_lcs_length,
    linediff_lcs_table,
    linediff_trim_common,
)


def rebuild_new(script, b):
    return [b[j] for tag, _, j in script if tag != "-"]


def rebuild_old(script, a):
    return [a[i] for tag, i, _ in script if tag != "+"]


class TA(unittest.TestCase):
    def test_shared_head_is_counted(self):
        self.assertEqual(linediff_trim_common([1, 2, 3], [1, 2, 4]), (2, 0))

    def test_shared_tail_is_counted(self):
        self.assertEqual(linediff_trim_common([1, 2, 3], [0, 2, 3]), (0, 2))

    def test_both_ends_counted_together(self):
        self.assertEqual(linediff_trim_common([5, 1, 6], [5, 2, 6]), (1, 1))

    def test_identical_input_has_no_tail_share(self):
        self.assertEqual(linediff_trim_common([1, 2], [1, 2]), (2, 0))

    def test_head_takes_priority_over_tail(self):
        self.assertEqual(linediff_trim_common([1, 1], [1, 1, 1]), (2, 0))

    def test_tail_stops_at_shorter_length(self):
        self.assertEqual(linediff_trim_common([2, 1, 1], [1, 1]), (0, 2))


class TB(unittest.TestCase):
    def test_small_table_has_exact_cells(self):
        self.assertEqual(linediff_lcs_table([1, 2], [2]), [[1, 0], [1, 0], [0, 0]])

    def test_empty_inputs_give_single_zero(self):
        self.assertEqual(linediff_lcs_table([], []), [[0]])

    def test_identical_strings_fill_staircase(self):
        self.assertEqual(
            linediff_lcs_table("abc", "abc"),
            [[3, 2, 1, 0], [2, 2, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
        )


class TC(unittest.TestCase):
    def test_identical_sequences_match_fully(self):
        self.assertEqual(linediff_lcs_length([1, 2, 3], [1, 2, 3]), 3)

    def test_gaps_between_matches_are_allowed(self):
        self.assertEqual(linediff_lcs_length([1, 2, 3, 4], [1, 3]), 2)

    def test_repeated_items_count_by_multiplicity(self):
        self.assertEqual(linediff_lcs_length([1, 1, 1], [1, 1]), 2)

    def test_single_item_matches_only_once(self):
        self.assertEqual(linediff_lcs_length(["x"], ["x", "x"]), 1)


class TD(unittest.TestCase):
    def test_equal_inputs_keep_every_line(self):
        self.assertEqual(
            linediff_edit_script(["a", "b"], ["a", "b"]),
            [("=", 0, 0), ("=", 1, 1)],
        )

    def test_empty_old_side_only_adds(self):
        self.assertEqual(
            linediff_edit_script([], ["x", "y"]), [("+", 0, 0), ("+", 0, 1)]
        )

    def test_single_replacement_drops_then_adds(self):
        self.assertEqual(
            linediff_edit_script(["x"], ["y"]), [("-", 0, 0), ("+", 1, 0)]
        )

    def test_appended_line_is_an_addition(self):
        self.assertEqual(
            linediff_edit_script(["a"], ["a", "b"]), [("=", 0, 0), ("+", 1, 1)]
        )

    def test_middle_insertion_keeps_neighbours(self):
        self.assertEqual(
            linediff_edit_script(["a", "c"], ["a", "b", "c"]),
            [("=", 0, 0), ("+", 1, 1), ("=", 1, 2)],
        )

    def test_middle_replacement_pairs_drop_with_add(self):
        self.assertEqual(
            linediff_edit_script(["a", "x", "c"], ["a", "y", "c"]),
            [("=", 0, 0), ("-", 1, 1), ("+", 2, 1), ("=", 2, 2)],
        )

    def test_extra_old_lines_all_dropped_first(self):
        self.assertEqual(
            linediff_edit_script(["x1", "x2"], ["y"]),
            [("-", 0, 0), ("-", 1, 0), ("+", 2, 0)],
        )

    def test_better_match_ahead_forces_addition(self):
        self.assertEqual(
            linediff_edit_script(["a", "q", "b", "z"], ["x", "a", "c", "b", "y"]),
            [("+", 0, 0), ("=", 0, 1), ("-", 1, 2), ("+", 2, 2), ("=", 2, 3), ("-", 3, 4), ("+", 4, 4)],
        )

    def test_tail_drops_report_new_position(self):
        self.assertEqual(
            linediff_edit_script([1, 2, 3], [1]),
            [("=", 0, 0), ("-", 1, 1), ("-", 2, 1)],
        )

    def test_head_and_tail_indices_survive_trimming(self):
        self.assertEqual(
            linediff_edit_script([1, 2, 3, 4, 5], [1, 2, 9, 4, 5]),
            [("=", 0, 0), ("=", 1, 1), ("-", 2, 2), ("+", 3, 2), ("=", 3, 3), ("=", 4, 4)],
        )

    def test_tail_indices_shift_with_length_gap(self):
        self.assertEqual(
            linediff_edit_script([1, 2, 3], [0, 1, 2, 3]),
            [("+", 0, 0), ("=", 0, 1), ("=", 1, 2), ("=", 2, 3)],
        )

    def test_script_rebuilds_both_sides(self):
        a = list("the quick brown fox jumps")
        b = list("a quick brown dog jumped over")
        script = linediff_edit_script(a, b)
        self.assertEqual(rebuild_new(script, b), b)
        self.assertEqual(rebuild_old(script, a), a)

    def test_script_size_matches_common_subsequence(self):
        a = list("ABCBDAB")
        b = list("BDCABA")
        script = linediff_edit_script(a, b)
        self.assertEqual(len(linediff_common_pairs(script)), 4)
        self.assertEqual(linediff_edit_distance(script), 5)


class TE(unittest.TestCase):
    def test_only_kept_lines_are_paired(self):
        script = [("=", 0, 0), ("-", 1, 1), ("=", 2, 1)]
        self.assertEqual(linediff_common_pairs(script), [(0, 0), (2, 1)])


class TF(unittest.TestCase):
    def test_replacement_costs_two_edits(self):
        self.assertEqual(linediff_edit_distance([("-", 0, 0), ("+", 1, 0)]), 2)


class TG(unittest.TestCase):
    def test_replacement_block_spans_both_sides(self):
        script = [("=", 0, 0), ("-", 1, 1), ("+", 2, 1), ("=", 2, 2)]
        self.assertEqual(linediff_change_blocks(script), [(1, 2, 1, 2)])

    def test_pure_insertion_has_empty_old_range(self):
        script = [("=", 0, 0), ("+", 1, 1), ("=", 1, 2)]
        self.assertEqual(linediff_change_blocks(script), [(1, 1, 1, 2)])

    def test_pure_deletion_has_empty_new_range(self):
        script = [("=", 0, 0), ("-", 1, 1), ("=", 2, 1)]
        self.assertEqual(linediff_change_blocks(script), [(1, 2, 1, 1)])

    def test_separated_edits_form_two_blocks(self):
        script = linediff_edit_script([1, 2, 3, 4, 5], [1, 9, 3, 8, 5])
        self.assertEqual(linediff_change_blocks(script), [(1, 2, 1, 2), (3, 4, 3, 4)])

    def test_block_at_end_is_flushed(self):
        script = linediff_edit_script([1], [2])
        self.assertEqual(linediff_change_blocks(script), [(0, 1, 0, 1)])

    def test_addition_before_drop_joins_one_block(self):
        script = [("+", 0, 0), ("-", 0, 1)]
        self.assertEqual(linediff_change_blocks(script), [(0, 1, 0, 1)])

    def test_consecutive_additions_extend_new_range(self):
        script = [("+", 0, 0), ("+", 0, 1)]
        self.assertEqual(linediff_change_blocks(script), [(0, 0, 0, 2)])
