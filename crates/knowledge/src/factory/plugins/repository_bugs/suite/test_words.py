# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

import unittest

from textkit.words import count_words, longest_word, top_words


class CountTests(unittest.TestCase):
    def test_case_is_ignored(self):
        self.assertEqual(count_words("The the THE cat"), {"the": 3, "cat": 1})

    def test_punctuation_is_stripped(self):
        self.assertEqual(count_words("end. end, end!"), {"end": 3})


class TopTests(unittest.TestCase):
    def test_most_frequent_first(self):
        self.assertEqual(top_words("a b b c c c", 2), ["c", "b"])

    def test_ties_break_alphabetically(self):
        self.assertEqual(top_words("b a", 1), ["a"])

    def test_n_larger_than_the_vocabulary(self):
        self.assertEqual(top_words("x y", 5), ["x", "y"])


class LongestTests(unittest.TestCase):
    def test_longest_word(self):
        self.assertEqual(longest_word("a abc ab"), "abc")

    def test_first_of_equal_length_wins(self):
        self.assertEqual(longest_word("ab cd"), "ab")

    def test_empty_text(self):
        self.assertEqual(longest_word(""), "")
