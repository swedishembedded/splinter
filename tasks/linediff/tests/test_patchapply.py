# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import unittest

from linediff.hunks import LinediffHunk
from linediff.patchapply import (
    LinediffConflict,
    linediff_apply_patch,
    linediff_conflicting_hunks,
    linediff_locate_hunk,
    linediff_place_hunks,
    linediff_reverse_hunks,
    linediff_unapply_patch,
)

TEXT = ["a", "b", "c", "d", "e", "f"]
TAGS = {" ": "=", "-": "-", "+": "+"}


def hunk(a_start, b_start, *specs):
    lines = tuple((TAGS[spec[0]], spec[1:]) for spec in specs)
    return LinediffHunk(
        a_start,
        sum(1 for tag, _ in lines if tag != "+"),
        b_start,
        sum(1 for tag, _ in lines if tag != "-"),
        lines,
    )


def conflict_of(lines, hunks, *shift):
    with unittest.TestCase().assertRaises(LinediffConflict) as caught:
        linediff_apply_patch(lines, hunks, *shift)
    return caught.exception


class ApplyPatchTests(unittest.TestCase):
    def test_no_hunks_return_an_equal_copy(self):
        result = linediff_apply_patch(TEXT, [])
        self.assertEqual(result, TEXT)
        self.assertIsNot(result, TEXT)

    def test_replacement_hunk_swaps_the_line(self):
        result = linediff_apply_patch(TEXT, [hunk(1, 1, "-b", "+B")])
        self.assertEqual(result, ["a", "B", "c", "d", "e", "f"])

    def test_context_lines_stay_in_place(self):
        result = linediff_apply_patch(TEXT, [hunk(0, 0, " a", "-b", " c")])
        self.assertEqual(result, ["a", "c", "d", "e", "f"])

    def test_insertion_at_top_is_prepended(self):
        self.assertEqual(
            linediff_apply_patch(TEXT, [hunk(0, 0, "+z")]), ["z"] + TEXT
        )

    def test_insertion_after_last_line_is_appended(self):
        self.assertEqual(
            linediff_apply_patch(TEXT, [hunk(6, 6, "+z")]), TEXT + ["z"]
        )

    def test_deleting_the_last_line_works(self):
        self.assertEqual(
            linediff_apply_patch(TEXT, [hunk(5, 5, "-f")]), TEXT[:5]
        )

    def test_later_hunks_use_original_positions(self):
        hunks = [hunk(1, 1, "-b"), hunk(4, 3, "-e", "+E")]
        self.assertEqual(linediff_apply_patch(TEXT, hunks), ["a", "c", "d", "E", "f"])

    def test_input_lines_are_not_modified(self):
        original = list(TEXT)
        linediff_apply_patch(TEXT, [hunk(1, 1, "-b", "+B")])
        self.assertEqual(TEXT, original)

    def test_mismatching_context_reports_first_hunk(self):
        error = conflict_of(TEXT, [hunk(1, 1, "-x", "+B")])
        self.assertEqual(error.hunk_index, 0)
        self.assertEqual(str(error), "hunk 1 does not match the text near line 2")

    def test_second_hunk_conflict_names_its_index(self):
        error = conflict_of(TEXT, [hunk(0, 0, "-a"), hunk(3, 2, "-q")])
        self.assertEqual(error.hunk_index, 1)
        self.assertEqual(str(error), "hunk 2 does not match the text near line 4")

    def test_hunk_past_end_of_text_conflicts(self):
        self.assertEqual(conflict_of(TEXT, [hunk(10, 10, "-f")]).hunk_index, 0)

    def test_insertion_beyond_end_of_text_conflicts(self):
        self.assertEqual(conflict_of(TEXT, [hunk(7, 7, "+z")]).hunk_index, 0)

    def test_hunk_one_line_late_conflicts_without_shift(self):
        self.assertEqual(conflict_of(TEXT, [hunk(2, 2, "-b")]).hunk_index, 0)

    def test_shift_lets_hunk_float_forward(self):
        result = linediff_apply_patch(TEXT, [hunk(3, 3, "-e")], max_shift=1)
        self.assertEqual(result, ["a", "b", "c", "d", "f"])

    def test_shift_lets_hunk_float_backward(self):
        result = linediff_apply_patch(TEXT, [hunk(5, 5, "-e")], max_shift=1)
        self.assertEqual(result, ["a", "b", "c", "d", "f"])

    def test_shift_beyond_limit_still_conflicts(self):
        self.assertEqual(conflict_of(TEXT, [hunk(2, 2, "-e")], 1).hunk_index, 0)

    def test_forward_candidate_wins_over_backward(self):
        result = linediff_apply_patch(["x", "a", "x"], [hunk(1, 1, "-x")], max_shift=1)
        self.assertEqual(result, ["x", "a"])

    def test_found_offset_carries_to_next_hunk(self):
        text = ["z", "a", "b", "c", "d", "y", "e", "f", "g", "h"]
        hunks = [hunk(1, 1, "-b"), hunk(6, 5, "-g")]
        self.assertEqual(
            linediff_apply_patch(text, hunks, max_shift=1),
            ["z", "a", "c", "d", "y", "e", "f", "h"],
        )

    def test_hunk_cannot_land_before_previous_hunk_end(self):
        hunks = [hunk(1, 1, "-x"), hunk(0, 0, "-x")]
        self.assertEqual(conflict_of(["x", "x"], hunks).hunk_index, 1)


class LocateHunkTests(unittest.TestCase):
    def test_expected_position_is_returned_when_matching(self):
        self.assertEqual(linediff_locate_hunk(TEXT, hunk(2, 2, " c"), 2), 2)

    def test_mismatching_text_gives_none(self):
        self.assertIsNone(linediff_locate_hunk(TEXT, hunk(2, 2, " q"), 2, 5))

    def test_search_defaults_to_exact_position_only(self):
        self.assertIsNone(linediff_locate_hunk(TEXT, hunk(2, 2, " b"), 2))

    def test_default_floor_allows_the_first_line(self):
        self.assertEqual(linediff_locate_hunk(TEXT, hunk(0, 0, " a"), 0), 0)

    def test_shift_of_two_reaches_backward(self):
        self.assertEqual(linediff_locate_hunk(TEXT, hunk(4, 4, " b"), 3, 2), 1)

    def test_shift_of_two_reaches_forward(self):
        self.assertEqual(linediff_locate_hunk(TEXT, hunk(0, 0, " e"), 2, 2), 4)

    def test_floor_excludes_earlier_starts(self):
        self.assertIsNone(linediff_locate_hunk(TEXT, hunk(1, 1, " b"), 1, 0, floor=2))

    def test_start_equal_to_floor_is_allowed(self):
        self.assertEqual(linediff_locate_hunk(TEXT, hunk(1, 1, " b"), 1, 0, floor=1), 1)

    def test_match_may_end_exactly_at_text_end(self):
        self.assertEqual(linediff_locate_hunk(TEXT, hunk(4, 4, " e", " f"), 4), 4)

    def test_match_may_not_run_past_text_end(self):
        self.assertIsNone(linediff_locate_hunk(TEXT, hunk(5, 5, " f", " g"), 5))

    def test_negative_expected_start_is_never_used(self):
        self.assertIsNone(linediff_locate_hunk(TEXT, hunk(0, 0, " f"), -1))


class PlaceHunksTests(unittest.TestCase):
    def test_positions_follow_hunk_order(self):
        hunks = [hunk(1, 1, "-b"), hunk(4, 4, "-q"), hunk(5, 5, "-f")]
        self.assertEqual(linediff_place_hunks(TEXT, hunks), [1, None, 5])

    def test_placement_defaults_to_no_shift(self):
        self.assertEqual(linediff_place_hunks(TEXT, [hunk(2, 2, "-b")]), [None])

    def test_conflicting_hunks_lists_misfits(self):
        hunks = [hunk(0, 0, "-q"), hunk(1, 1, "-b"), hunk(2, 2, "-q")]
        self.assertEqual(linediff_conflicting_hunks(TEXT, hunks), [0, 2])

    def test_clean_patch_has_no_conflicts(self):
        self.assertEqual(linediff_conflicting_hunks(TEXT, [hunk(0, 0, "-a")]), [])

    def test_shift_allowance_removes_conflicts(self):
        hunks = [hunk(2, 2, "-b")]
        self.assertEqual(linediff_conflicting_hunks(TEXT, hunks), [0])
        self.assertEqual(linediff_conflicting_hunks(TEXT, hunks, 1), [])


class ReverseHunksTests(unittest.TestCase):
    def test_reverse_swaps_tags_and_starts(self):
        reversed_hunk = linediff_reverse_hunks([hunk(1, 4, "-b", "+B", " c")])[0]
        self.assertEqual(
            reversed_hunk,
            LinediffHunk(4, 2, 1, 2, (("+", "b"), ("-", "B"), ("=", "c"))),
        )

    def test_reversing_twice_restores_hunks(self):
        hunks = [hunk(1, 1, "-b", "+B"), hunk(4, 4, "+z", " e")]
        self.assertEqual(linediff_reverse_hunks(linediff_reverse_hunks(hunks)), hunks)

    def test_reverse_of_nothing_is_nothing(self):
        self.assertEqual(linediff_reverse_hunks([]), [])


class UnapplyPatchTests(unittest.TestCase):
    def test_unapply_restores_the_original_text(self):
        hunks = [hunk(1, 1, "-b", "+B"), hunk(4, 4, "-e", "+E", "+F")]
        patched = linediff_apply_patch(TEXT, hunks)
        self.assertEqual(linediff_unapply_patch(patched, hunks), TEXT)

    def test_unapply_on_unpatched_text_conflicts(self):
        with self.assertRaises(LinediffConflict):
            linediff_unapply_patch(TEXT, [hunk(1, 1, "-b", "+B")])

    def test_unapply_defaults_to_exact_position_only(self):
        with self.assertRaises(LinediffConflict):
            linediff_unapply_patch(["z", "a", "B", "c"], [hunk(1, 1, "-b", "+B")])

    def test_unapply_honours_shift_allowance(self):
        hunks = [hunk(1, 1, "-b", "+B")]
        shifted = ["z", "a", "B", "c"]
        self.assertEqual(
            linediff_unapply_patch(shifted, hunks, max_shift=1), ["z", "a", "b", "c"]
        )
