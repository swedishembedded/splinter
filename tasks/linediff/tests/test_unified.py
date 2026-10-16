# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import unittest

from linediff.hunks import LinediffHunk
from linediff.unified import (
    linediff_format_range,
    linediff_render_hunk,
    linediff_render_hunk_header,
    linediff_render_numbered_hunk,
    linediff_render_patch,
    linediff_render_stat_line,
)

REPLACE = LinediffHunk(
    2, 3, 2, 3, (("=", "keep"), ("-", "old"), ("+", "new"), ("=", "tail"))
)
INSERT = LinediffHunk(5, 0, 6, 2, (("+", "one"), ("+", "two")))
REMOVE = LinediffHunk(0, 1, 0, 0, (("-", "gone"),))


class FormatRangeTests(unittest.TestCase):
    def test_empty_range_prints_zero_count(self):
        self.assertEqual(linediff_format_range(3, 0), "3,0")

    def test_empty_range_at_top_prints_zero_zero(self):
        self.assertEqual(linediff_format_range(0, 0), "0,0")

    def test_single_line_range_omits_count(self):
        self.assertEqual(linediff_format_range(3, 1), "4")

    def test_two_line_range_prints_count(self):
        self.assertEqual(linediff_format_range(3, 2), "4,2")

    def test_long_range_starts_at_one(self):
        self.assertEqual(linediff_format_range(0, 5), "1,5")


class FormatRangeEdgeTests(unittest.TestCase):
    def test_first_line_single_range_is_one(self):
        self.assertEqual(linediff_format_range(0, 1), "1")

    def test_late_two_line_range_counts_from_start(self):
        self.assertEqual(linediff_format_range(99, 2), "100,2")


class HunkHeaderTests(unittest.TestCase):
    def test_header_shows_both_ranges(self):
        self.assertEqual(linediff_render_hunk_header(REPLACE), "@@ -3,3 +3,3 @@")

    def test_header_uses_short_form_for_single_line(self):
        self.assertEqual(linediff_render_hunk_header(REMOVE), "@@ -1 +0,0 @@")

    def test_header_of_insertion_has_empty_old_range(self):
        self.assertEqual(linediff_render_hunk_header(INSERT), "@@ -5,0 +7,2 @@")


class RenderHunkTests(unittest.TestCase):
    def test_body_lines_get_tag_prefixes(self):
        self.assertEqual(
            linediff_render_hunk(REPLACE),
            ["@@ -3,3 +3,3 @@", " keep", "-old", "+new", " tail"],
        )

    def test_added_lines_start_with_plus(self):
        self.assertEqual(linediff_render_hunk(INSERT)[1:], ["+one", "+two"])

    def test_text_with_leading_space_is_preserved(self):
        hunk = LinediffHunk(0, 1, 0, 1, (("=", "  indented"),))
        self.assertEqual(linediff_render_hunk(hunk)[1], "   indented")


class RenderPatchTests(unittest.TestCase):
    def test_no_hunks_render_as_empty_text(self):
        self.assertEqual(linediff_render_patch([], "a.txt", "b.txt"), "")

    def test_file_headers_come_first(self):
        text = linediff_render_patch([REMOVE], "a/x.txt", "b/x.txt")
        self.assertEqual(text.split("\n")[:2], ["--- a/x.txt", "+++ b/x.txt"])

    def test_patch_text_ends_with_newline(self):
        self.assertTrue(linediff_render_patch([REMOVE], "a", "b").endswith("-gone\n"))

    def test_full_patch_text_is_exact(self):
        self.assertEqual(
            linediff_render_patch([REMOVE, INSERT], "old", "new"),
            "--- old\n+++ new\n@@ -1 +0,0 @@\n-gone\n@@ -5,0 +7,2 @@\n+one\n+two\n",
        )

    def test_single_hunk_patch_is_exact(self):
        self.assertEqual(
            linediff_render_patch([REPLACE], "p", "q"),
            "--- p\n+++ q\n@@ -3,3 +3,3 @@\n keep\n-old\n+new\n tail\n",
        )

    def test_names_with_spaces_are_kept_whole(self):
        text = linediff_render_patch([REMOVE], "my file.txt", "my file.txt")
        self.assertTrue(text.startswith("--- my file.txt\n+++ my file.txt\n"))


class StatLineTests(unittest.TestCase):
    def test_plural_counts_use_plural_nouns(self):
        self.assertEqual(
            linediff_render_stat_line([INSERT, REPLACE]),
            "2 hunks, 3 insertions(+), 1 deletion(-)",
        )

    def test_singular_counts_use_singular_nouns(self):
        self.assertEqual(
            linediff_render_stat_line([REMOVE]), "1 hunk, 0 insertions(+), 1 deletion(-)"
        )

    def test_empty_patch_reports_zeros(self):
        self.assertEqual(
            linediff_render_stat_line([]), "0 hunks, 0 insertions(+), 0 deletions(-)"
        )


class NumberedHunkTests(unittest.TestCase):
    def test_numbers_start_after_hunk_offsets(self):
        self.assertEqual(
            linediff_render_numbered_hunk(REPLACE),
            [
                "   3    3  keep",
                "   4      -old",
                "        4 +new",
                "   5    5  tail",
            ],
        )

    def test_width_controls_column_size(self):
        self.assertEqual(linediff_render_numbered_hunk(REMOVE, 2), [" 1    -gone"])

    def test_insertion_counts_from_new_start(self):
        self.assertEqual(
            linediff_render_numbered_hunk(INSERT, 3),
            ["      7 +one", "      8 +two"],
        )
