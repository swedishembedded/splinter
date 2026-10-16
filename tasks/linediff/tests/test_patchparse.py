# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import unittest

from linediff.hunks import LinediffHunk
from linediff.patchparse import (
    LinediffPatchError,
    linediff_parse_hunk_header,
    linediff_parse_patch,
    linediff_parse_patch_set,
    linediff_parse_range,
)

SIMPLE = "--- a.txt\n+++ b.txt\n@@ -2,3 +2,3 @@\n keep\n-old\n+new\n tail\n"


def message_of(call, *args):
    with unittest.TestCase().assertRaises(LinediffPatchError) as caught:
        call(*args)
    return str(caught.exception)


class TA(unittest.TestCase):
    def test_bare_number_means_single_line(self):
        self.assertEqual(linediff_parse_range("4"), (3, 1))

    def test_number_pair_gives_start_and_count(self):
        self.assertEqual(linediff_parse_range("4,2"), (3, 2))

    def test_zero_count_keeps_start_unshifted(self):
        self.assertEqual(linediff_parse_range("3,0"), (3, 0))

    def test_empty_range_at_top_is_allowed(self):
        self.assertEqual(linediff_parse_range("0,0"), (0, 0))

    def test_nonempty_range_at_line_zero_is_rejected(self):
        self.assertEqual(
            message_of(linediff_parse_range, "0,3"),
            "range '0,3' starts at line 0 but is not empty",
        )

    def test_letters_are_rejected_as_range(self):
        self.assertEqual(message_of(linediff_parse_range, "x"), "bad range 'x'")


class TB(unittest.TestCase):
    def test_header_gives_four_numbers(self):
        self.assertEqual(linediff_parse_hunk_header("@@ -3,3 +3,3 @@"), (2, 3, 2, 3))

    def test_header_with_empty_new_range(self):
        self.assertEqual(linediff_parse_hunk_header("@@ -1 +0,0 @@"), (0, 1, 0, 0))

    def test_header_with_bad_range_reports_range(self):
        self.assertEqual(
            message_of(linediff_parse_hunk_header, "@@ -a +1 @@"), "bad range 'a'"
        )


class TC(unittest.TestCase):
    def test_file_names_are_taken_after_markers(self):
        old, new, _ = linediff_parse_patch(SIMPLE)
        self.assertEqual((old, new), ("a.txt", "b.txt"))

    def test_hunk_fields_come_from_header(self):
        hunk = linediff_parse_patch(SIMPLE)[2][0]
        self.assertEqual((hunk.a_start, hunk.a_count, hunk.b_start, hunk.b_count), (1, 3, 1, 3))

    def test_body_lines_become_tagged_pairs(self):
        hunk = linediff_parse_patch(SIMPLE)[2][0]
        self.assertEqual(
            hunk.lines, (("=", "keep"), ("-", "old"), ("+", "new"), ("=", "tail"))
        )

    def test_missing_final_newline_is_accepted(self):
        self.assertEqual(linediff_parse_patch(SIMPLE[:-1]), linediff_parse_patch(SIMPLE))

    def test_headers_alone_give_no_hunks(self):
        self.assertEqual(linediff_parse_patch("--- a\n+++ b\n"), ("a", "b", []))

    def test_blank_context_line_keeps_empty_text(self):
        text = "--- a\n+++ b\n@@ -1,2 +1,2 @@\n \n-x\n+y\n"
        self.assertEqual(linediff_parse_patch(text)[2][0].lines[0], ("=", ""))

    def test_removed_dashed_line_is_body(self):
        text = "--- a\n+++ b\n@@ -1 +0,0 @@\n--- x\n"
        self.assertEqual(linediff_parse_patch(text)[2][0].lines, (("-", "-- x"),))

    def test_pure_insertion_hunk_is_parsed(self):
        text = "--- a\n+++ b\n@@ -3,0 +4,2 @@\n+p\n+q\n"
        hunk = linediff_parse_patch(text)[2][0]
        self.assertEqual((hunk.a_start, hunk.a_count, hunk.b_start, hunk.b_count), (3, 0, 3, 2))

    def test_adjacent_hunks_are_accepted(self):
        text = "--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n@@ -2 +2 @@\n-p\n+q\n"
        self.assertEqual(len(linediff_parse_patch(text)[2]), 2)

    def test_empty_text_holds_no_patch(self):
        self.assertEqual(
            message_of(linediff_parse_patch, ""), "expected one file patch, found 0"
        )

    def test_two_file_patches_are_refused_as_one(self):
        text = SIMPLE + "--- c\n+++ d\n"
        self.assertEqual(
            message_of(linediff_parse_patch, text), "expected one file patch, found 2"
        )

    def test_text_without_file_header_is_rejected(self):
        self.assertEqual(
            message_of(linediff_parse_patch, "hello\n"),
            "line 1: expected '--- ' file header",
        )

    def test_missing_new_file_header_is_rejected(self):
        self.assertEqual(
            message_of(linediff_parse_patch, "--- a\n"),
            "line 2: expected '+++ ' file header",
        )

    def test_truncated_hunk_is_rejected(self):
        self.assertEqual(
            message_of(linediff_parse_patch, "--- a\n+++ b\n@@ -1,2 +1,2 @@\n x\n"),
            "hunk at line 3 is truncated",
        )

    def test_unknown_body_prefix_is_rejected(self):
        self.assertEqual(
            message_of(linediff_parse_patch, "--- a\n+++ b\n@@ -1,2 +1,2 @@\n x\n?y\n"),
            "line 5: unexpected line '?y'",
        )

    def test_empty_body_line_is_rejected(self):
        self.assertEqual(
            message_of(linediff_parse_patch, "--- a\n+++ b\n@@ -1,2 +1,2 @@\n\n x\n"),
            "line 4: unexpected line ''",
        )

    def test_extra_body_line_breaks_header_counts(self):
        self.assertEqual(
            message_of(linediff_parse_patch, "--- a\n+++ b\n@@ -1 +0,0 @@\n x\n"),
            "hunk at line 3 does not match its header counts",
        )

    def test_overlapping_hunks_are_rejected(self):
        text = "--- a\n+++ b\n@@ -1,2 +1,2 @@\n-x\n-y\n+p\n+q\n@@ -2 +2 @@\n-y\n+z\n"
        self.assertEqual(
            message_of(linediff_parse_patch, text),
            "line 8: hunk overlaps the previous hunk",
        )


class TD(unittest.TestCase):
    def test_set_collects_every_file_patch(self):
        patches = linediff_parse_patch_set(SIMPLE + "--- c\n+++ d\n@@ -1 +0,0 @@\n-z\n")
        self.assertEqual([(p[0], p[1], len(p[2])) for p in patches], [("a.txt", "b.txt", 1), ("c", "d", 1)])
