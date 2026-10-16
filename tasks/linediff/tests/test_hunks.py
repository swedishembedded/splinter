# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import unittest
from dataclasses import FrozenInstanceError

from linediff.hunks import (
    LinediffHunk,
    linediff_group_hunks,
    linediff_hunk_covering,
    linediff_hunk_stats,
    linediff_trim_hunk_context,
)


def world(spec):
    """Build (script, a, b) from a spec such as '==-+=': keep, drop, add."""
    script, a, b = [], [], []
    for tag in spec:
        i, j = len(a), len(b)
        script.append((tag, i, j))
        if tag != "+":
            a.append("L%d" % i)
        if tag == "=":
            b.append("L%d" % i)
        elif tag == "+":
            b.append("N%d" % j)
    return script, a, b


def hunks_of(spec, context):
    script, a, b = world(spec)
    return linediff_group_hunks(script, a, b, context)


class TA(unittest.TestCase):
    def test_unchanged_text_yields_no_hunks(self):
        self.assertEqual(hunks_of("=" * 10, 3), [])

    def test_hunk_old_start_counts_leading_context(self):
        self.assertEqual(hunks_of("====-===", 1)[0].a_start, 3)

    def test_hunk_old_count_includes_context_and_drops(self):
        self.assertEqual(hunks_of("====-===", 1)[0].a_count, 3)

    def test_hunk_new_count_excludes_dropped_lines(self):
        self.assertEqual(hunks_of("====-===", 1)[0].b_count, 2)

    def test_zero_context_keeps_only_changed_lines(self):
        hunk = hunks_of("====-===", 0)[0]
        self.assertEqual((hunk.a_start, hunk.a_count, hunk.lines), (4, 1, (("-", "L4"),)))

    def test_context_is_clamped_at_file_start(self):
        hunk = hunks_of("=-====", 5)[0]
        self.assertEqual((hunk.a_start, hunk.a_count), (0, 6))

    def test_context_is_clamped_at_file_end(self):
        hunk = hunks_of("====-=", 5)[0]
        self.assertEqual((hunk.a_start, hunk.a_count), (0, 6))

    def test_default_context_is_three_lines(self):
        script, a, b = world("=" * 10 + "-" + "=" * 10)
        hunk = linediff_group_hunks(script, a, b)[0]
        self.assertEqual((hunk.a_start, hunk.a_count), (7, 7))

    def test_distant_changes_make_separate_hunks(self):
        hunks = hunks_of("=-====+=", 1)
        self.assertEqual([h.a_start for h in hunks], [0, 5])

    def test_later_changes_join_the_latest_hunk(self):
        hunks = hunks_of("-===++", 0)
        self.assertEqual([len(h.lines) for h in hunks], [1, 2])

    def test_gap_of_twice_context_merges_hunks(self):
        self.assertEqual(len(hunks_of("=-==-=", 1)), 1)

    def test_gap_beyond_twice_context_splits_hunks(self):
        self.assertEqual(len(hunks_of("=-===-=", 1)), 2)

    def test_adjacent_changes_merge_without_context(self):
        self.assertEqual(len(hunks_of("=-+=", 0)), 1)

    def test_single_kept_line_splits_without_context(self):
        self.assertEqual(len(hunks_of("-=-", 0)), 2)

    def test_hunk_lines_carry_tags_and_text(self):
        hunk = hunks_of("=-+=", 1)[0]
        self.assertEqual(
            hunk.lines, (("=", "L0"), ("-", "L1"), ("+", "N1"), ("=", "L2"))
        )

    def test_pure_insertion_has_empty_old_range(self):
        hunk = hunks_of("==+==", 0)[0]
        self.assertEqual(
            (hunk.a_start, hunk.a_count, hunk.b_start, hunk.b_count), (2, 0, 2, 1)
        )

    def test_leading_drop_reports_new_side_position(self):
        hunk = hunks_of("+=-", 0)[0]
        self.assertEqual((hunk.b_start, hunk.b_count), (0, 1))

    def test_negative_context_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            hunks_of("-", -1)
        self.assertEqual(str(caught.exception), "context must not be negative")


class TB(unittest.TestCase):
    def test_old_lines_skip_additions(self):
        hunk = hunks_of("=-+=", 1)[0]
        self.assertEqual(hunk.old_lines(), ["L0", "L1", "L2"])

    def test_new_lines_skip_drops(self):
        hunk = hunks_of("=-+=", 1)[0]
        self.assertEqual(hunk.new_lines(), ["L0", "N1", "L2"])


class TC(unittest.TestCase):
    def test_stats_count_additions_and_drops_apart(self):
        self.assertEqual(linediff_hunk_stats(hunks_of("--+=", 1)), (1, 2))

    def test_stats_sum_over_all_hunks(self):
        self.assertEqual(linediff_hunk_stats(hunks_of("-===+++=-", 0)), (3, 2))


class TD(unittest.TestCase):
    def setUp(self):
        self.hunks = hunks_of("==-===-==", 0)

    def test_first_old_line_is_inside_hunk(self):
        self.assertEqual(linediff_hunk_covering(self.hunks, 2), 0)

    def test_line_before_hunk_is_outside(self):
        self.assertIsNone(linediff_hunk_covering(self.hunks, 1))

    def test_line_after_hunk_end_is_outside(self):
        self.assertIsNone(linediff_hunk_covering(self.hunks, 3))

    def test_insertion_hunk_covers_no_old_line(self):
        self.assertIsNone(linediff_hunk_covering(hunks_of("==+==", 0), 2))


class TE(unittest.TestCase):
    def setUp(self):
        self.hunk = hunks_of("====-=-====", 3)[0]

    def test_trimming_shortens_leading_context(self):
        trimmed = linediff_trim_hunk_context(self.hunk, 1)
        self.assertEqual(trimmed.lines[0], ("=", "L3"))

    def test_trimming_moves_starts_and_counts(self):
        trimmed = linediff_trim_hunk_context(self.hunk, 1)
        self.assertEqual(
            (trimmed.a_start, trimmed.a_count, trimmed.b_start, trimmed.b_count),
            (3, 5, 3, 3),
        )

    def test_trimming_keeps_interior_kept_lines(self):
        trimmed = linediff_trim_hunk_context(self.hunk, 0)
        self.assertEqual([tag for tag, _ in trimmed.lines], ["-", "=", "-"])

    def test_context_equal_to_available_keeps_lines(self):
        hunk = hunks_of("=-", 3)[0]
        self.assertEqual(len(linediff_trim_hunk_context(hunk, 1).lines), 2)

    def test_all_kept_hunk_keeps_leading_context(self):
        hunk = LinediffHunk(4, 3, 4, 3, (("=", "x"), ("=", "y"), ("=", "z")))
        trimmed = linediff_trim_hunk_context(hunk, 1)
        self.assertEqual((trimmed.a_start, trimmed.lines), (6, (("=", "z"),)))

    def test_all_kept_hunk_loses_everything_at_zero(self):
        hunk = LinediffHunk(4, 3, 4, 3, (("=", "x"), ("=", "y"), ("=", "z")))
        trimmed = linediff_trim_hunk_context(hunk, 0)
        self.assertEqual((trimmed.a_count, trimmed.b_count, trimmed.lines), (0, 0, ()))

    def test_hunks_are_immutable_values(self):
        hunk = LinediffHunk(0, 0, 0, 0, ())
        with self.assertRaises(FrozenInstanceError):
            hunk.a_start = 1

    def test_change_first_hunk_keeps_its_start(self):
        hunk = LinediffHunk(2, 2, 2, 1, (("-", "x"), ("=", "y")))
        self.assertEqual(linediff_trim_hunk_context(hunk, 0).a_start, 2)

    def test_trimming_rejects_negative_context(self):
        with self.assertRaises(ValueError) as caught:
            linediff_trim_hunk_context(self.hunk, -1)
        self.assertEqual(str(caught.exception), "context must not be negative")
