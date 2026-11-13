# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

import unittest

from textkit.strings import is_palindrome, slugify, title_case, truncate


class SlugifyTests(unittest.TestCase):
    def test_words_are_joined_by_dashes(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_edges_carry_no_dash(self):
        self.assertEqual(slugify("  Hello, World! "), "hello-world")

    def test_runs_of_symbols_make_one_dash(self):
        self.assertEqual(slugify("a --- b"), "a-b")


class TruncateTests(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(truncate("abc", 10), "abc")

    def test_text_exactly_at_the_limit_is_unchanged(self):
        self.assertEqual(truncate("abc", 3), "abc")

    def test_long_text_is_cut_to_the_limit_with_an_ellipsis(self):
        self.assertEqual(truncate("abcdefghij", 8), "abcde...")
        self.assertEqual(len(truncate("abcdefghij", 8)), 8)

    def test_tiny_limit_is_refused(self):
        with self.assertRaises(ValueError):
            truncate("abc", 2)


class PalindromeTests(unittest.TestCase):
    def test_case_is_ignored(self):
        self.assertTrue(is_palindrome("Racecar"))

    def test_punctuation_is_ignored(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))

    def test_other_text_is_not(self):
        self.assertFalse(is_palindrome("abca"))


class TitleCaseTests(unittest.TestCase):
    def test_each_word_is_capitalised(self):
        self.assertEqual(title_case("hello big world"), "Hello Big World")

    def test_the_rest_of_each_word_is_lowered(self):
        self.assertEqual(title_case("hELLO wORLD"), "Hello World")
