# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements end-to-end acceptance tests for learning
# agent stacks, for its clients. If your team needs expertise in proving that
# a model, an agent runtime and a trainer work together, you can procure our
# services by sending an email to info@swedishembedded.com.

import unittest

from tiny.words import capitalize_words, count_words, is_palindrome, longest_word


class WordsTest(unittest.TestCase):
    def test_count_words_splits_on_any_whitespace(self):
        self.assertEqual(count_words("a  b\tc\nd"), 4)
        self.assertEqual(count_words(""), 0)

    def test_longest_word_is_the_first_of_equals(self):
        self.assertEqual(longest_word("aa bbb ccc d"), "bbb")
        self.assertEqual(longest_word("   "), "")

    def test_capitalize_words_keeps_the_rest_of_each_word(self):
        self.assertEqual(capitalize_words("hello wORLD"), "Hello WORLD")

    def test_is_palindrome_ignores_case_and_punctuation(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))
        self.assertFalse(is_palindrome("tiny"))


if __name__ == "__main__":
    unittest.main()
