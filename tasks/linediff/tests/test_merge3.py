# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import unittest

from linediff.merge3 import (
    linediff_changed_spans,
    linediff_has_conflicts,
    linediff_merge_chunks,
    linediff_merge_three,
    linediff_render_merge,
)


def chunks(base, ours, theirs):
    return linediff_merge_chunks(list(base), list(ours), list(theirs))


def merged(base, ours, theirs):
    lines, count = linediff_merge_three(list(base), list(ours), list(theirs))
    return "".join(lines) if count == 0 else (lines, count)


class TA(unittest.TestCase):
    def test_identical_text_has_no_spans(self):
        self.assertEqual(linediff_changed_spans(list("abc"), list("abc")), [])

    def test_replacement_span_holds_new_lines(self):
        self.assertEqual(
            linediff_changed_spans(list("abc"), list("aXc")), [(1, 2, ["X"])]
        )

    def test_insertion_span_is_empty_in_base(self):
        self.assertEqual(
            linediff_changed_spans(list("ac"), list("abc")), [(1, 1, ["b"])]
        )

    def test_deletion_span_has_no_replacement(self):
        self.assertEqual(
            linediff_changed_spans(list("abc"), list("ac")), [(1, 2, [])]
        )

    def test_two_edits_make_two_spans(self):
        self.assertEqual(
            linediff_changed_spans(list("abcde"), list("aXcYe")),
            [(1, 2, ["X"]), (3, 4, ["Y"])],
        )

    def test_appended_lines_attach_at_the_end(self):
        self.assertEqual(
            linediff_changed_spans(list("abc"), list("abcd")), [(3, 3, ["d"])]
        )


class TB(unittest.TestCase):
    def test_only_our_edit_is_taken(self):
        self.assertEqual(
            chunks("abc", "aXc", "abc"),
            [("clean", ["a"]), ("clean", ["X"]), ("clean", ["c"])],
        )

    def test_only_their_edit_is_taken(self):
        self.assertEqual(
            chunks("abc", "abc", "abY"),
            [("clean", ["a", "b"]), ("clean", ["Y"])],
        )

    def test_identical_edits_on_both_sides_agree(self):
        self.assertEqual(
            chunks("abc", "aXc", "aXc"),
            [("clean", ["a"]), ("clean", ["X"]), ("clean", ["c"])],
        )

    def test_first_line_edit_lacks_leading_chunk(self):
        self.assertEqual(
            chunks("abc", "Xbc", "abc"), [("clean", ["X"]), ("clean", ["b", "c"])]
        )

    def test_separate_edits_from_both_sides_combine(self):
        self.assertEqual(
            chunks("abcde", "aXcde", "abcdY"),
            [("clean", ["a"]), ("clean", ["X"]), ("clean", ["c", "d"]), ("clean", ["Y"])],
        )

    def test_same_line_changed_differently_conflicts(self):
        self.assertEqual(
            chunks("abc", "aXc", "aYc"),
            [("clean", ["a"]), ("conflict", ["X"], ["Y"]), ("clean", ["c"])],
        )

    def test_touching_neighbour_edits_conflict(self):
        self.assertEqual(
            chunks("abcd", "aXcd", "abYd"),
            [("clean", ["a"]), ("conflict", ["X", "c"], ["b", "Y"]), ("clean", ["d"])],
        )

    def test_edits_apart_by_kept_line_agree(self):
        self.assertEqual(
            chunks("abcde", "aXcde", "abcYe"),
            [("clean", ["a"]), ("clean", ["X"]), ("clean", ["c"]), ("clean", ["Y"]), ("clean", ["e"])],
        )

    def test_different_insertions_same_point_conflict(self):
        self.assertEqual(
            chunks("ac", "aXc", "aYc"),
            [("clean", ["a"]), ("conflict", ["X"], ["Y"]), ("clean", ["c"])],
        )

    def test_insertion_touching_a_change_conflicts(self):
        self.assertEqual(
            chunks("abc", "abXc", "aYc"),
            [("clean", ["a"]), ("conflict", ["b", "X"], ["Y"]), ("clean", ["c"])],
        )

    def test_delete_against_change_conflicts(self):
        self.assertEqual(
            chunks("abc", "ac", "aXc"),
            [("clean", ["a"]), ("conflict", [], ["X"]), ("clean", ["c"])],
        )

    def test_chained_edits_form_one_cluster(self):
        self.assertEqual(
            chunks("abcde", "aXcYe", "abZde"),
            [("clean", ["a"]), ("conflict", ["X", "c", "Y"], ["b", "Z", "d"]), ("clean", ["e"])],
        )

    def test_wide_edit_swallows_inner_edits(self):
        self.assertEqual(
            chunks("abcdefg", "aXfg", "abYdZfg"),
            [("clean", ["a"]), ("conflict", ["X"], ["b", "Y", "d", "Z"]), ("clean", ["f", "g"])],
        )

    def test_last_line_edit_lacks_trailing_chunk(self):
        self.assertEqual(
            chunks("abc", "abc", "abZ"), [("clean", ["a", "b"]), ("clean", ["Z"])]
        )

    def test_appends_on_both_sides_conflict(self):
        self.assertEqual(
            chunks("ab", "abX", "abY"), [("clean", ["a", "b"]), ("conflict", ["X"], ["Y"])]
        )


class TC(unittest.TestCase):
    def test_later_cluster_grows_alone(self):
        self.assertEqual(
            chunks("abcdefg", "XbcdYfg", "abcdeZg"),
            [
                ("clean", ["X"]),
                ("clean", ["b", "c", "d"]),
                ("conflict", ["Y", "f"], ["e", "Z"]),
                ("clean", ["g"]),
            ],
        )


class TD(unittest.TestCase):
    def test_conflict_chunk_is_detected(self):
        self.assertTrue(linediff_has_conflicts([("clean", ["a"]), ("conflict", [], [])]))
        self.assertFalse(linediff_has_conflicts([("clean", ["a"])]))


class TE(unittest.TestCase):
    def test_clean_chunks_are_concatenated(self):
        self.assertEqual(
            linediff_render_merge([("clean", ["a"]), ("clean", ["b"])]), ["a", "b"]
        )

    def test_conflict_is_wrapped_in_markers(self):
        self.assertEqual(
            linediff_render_merge([("conflict", ["x"], ["y"])]),
            ["<<<<<<< ours", "x", "=======", "y", ">>>>>>> theirs"],
        )

    def test_custom_labels_follow_markers(self):
        self.assertEqual(
            linediff_render_merge([("conflict", [], [])], "mine", "other"),
            ["<<<<<<< mine", "=======", ">>>>>>> other"],
        )


class TF(unittest.TestCase):
    def test_clean_merge_returns_zero_conflicts(self):
        self.assertEqual(merged("abcde", "aXcde", "abcdY"), "aXcdY")

    def test_conflicting_merge_embeds_markers(self):
        self.assertEqual(
            merged("abc", "aXc", "aYc"),
            (["a", "<<<<<<< ours", "X", "=======", "Y", ">>>>>>> theirs", "c"], 1),
        )

    def test_conflict_count_covers_every_cluster(self):
        _, count = linediff_merge_three(
            list("abcde"), list("XbcdE"), list("YbcdF")
        )
        self.assertEqual(count, 2)

    def test_labels_are_passed_to_markers(self):
        lines, _ = linediff_merge_three(
            list("abc"), list("aXc"), list("aYc"), "left", "right"
        )
        self.assertEqual((lines[1], lines[5]), ("<<<<<<< left", ">>>>>>> right"))
