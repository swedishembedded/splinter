# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import unittest
from fractions import Fraction

from linediff.similarity import (
    linediff_best_match,
    linediff_line_similarity,
    linediff_normalize_space,
    linediff_pair_changed_lines,
    linediff_rank_matches,
    linediff_same_ignoring_space,
    linediff_sequence_ratio,
    linediff_split_words,
    linediff_word_jaccard,
    linediff_word_overlap,
)


class TA(unittest.TestCase):
    def test_words_come_out_lowercased(self):
        self.assertEqual(linediff_split_words("Hello WORLD"), ["hello", "world"])

    def test_punctuation_splits_words_apart(self):
        self.assertEqual(linediff_split_words("a,b;c.d"), ["a", "b", "c", "d"])

    def test_digits_and_underscores_stay_in_words(self):
        self.assertEqual(linediff_split_words("x_1 y2"), ["x_1", "y2"])


class TB(unittest.TestCase):
    def test_runs_of_whitespace_collapse(self):
        self.assertEqual(linediff_normalize_space("a \t  b"), "a b")

    def test_outer_blanks_are_trimmed(self):
        self.assertEqual(linediff_normalize_space("  a b  "), "a b")

    def test_different_spacing_counts_as_same(self):
        self.assertTrue(linediff_same_ignoring_space("a  b", " a b"))

    def test_different_words_do_not_count_as_same(self):
        self.assertFalse(linediff_same_ignoring_space("a b", "a c"))


class TC(unittest.TestCase):
    def test_lines_without_shared_words_score_zero(self):
        self.assertEqual(linediff_word_overlap("a b", "c d"), Fraction(0))

    def test_two_empty_lines_score_one(self):
        self.assertEqual(linediff_word_overlap("", "!!"), Fraction(1))

    def test_partial_overlap_is_dice_coefficient(self):
        self.assertEqual(linediff_word_overlap("a b c", "a b d"), Fraction(2, 3))

    def test_repeated_words_count_by_multiplicity(self):
        self.assertEqual(linediff_word_overlap("a a b", "a b"), Fraction(4, 5))


class TD(unittest.TestCase):
    def test_shared_words_over_union_size(self):
        self.assertEqual(linediff_word_jaccard("a b c", "a b d"), Fraction(1, 2))

    def test_repeated_words_are_ignored(self):
        self.assertEqual(linediff_word_jaccard("a a", "a"), Fraction(1))


class TJ(unittest.TestCase):
    def test_wordless_and_disjoint_jaccard_scores(self):
        self.assertEqual(linediff_word_jaccard("", "?"), Fraction(1))
        self.assertEqual(linediff_word_jaccard("a", "b"), Fraction(0))


class TE(unittest.TestCase):
    def test_character_ratio_of_known_pair(self):
        self.assertEqual(linediff_sequence_ratio("kitten", "sitting"), Fraction(8, 13))

    def test_two_empty_sequences_score_one(self):
        self.assertEqual(linediff_sequence_ratio("", ""), Fraction(1))

    def test_line_lists_are_compared_by_line(self):
        self.assertEqual(
            linediff_sequence_ratio(["a", "b", "c"], ["a", "c"]), Fraction(4, 5)
        )


class TF(unittest.TestCase):
    def test_score_averages_word_and_character_views(self):
        self.assertEqual(linediff_line_similarity("abc def", "abc xyz"), Fraction(15, 28))


class TG(unittest.TestCase):
    def test_most_similar_candidate_wins(self):
        self.assertEqual(
            linediff_best_match("abc def", ["zzz", "abc xyz", "abc deg"]), 2
        )

    def test_equal_scores_prefer_earlier_candidate(self):
        self.assertEqual(linediff_best_match("abc", ["abc", "abc"]), 0)

    def test_weak_candidates_fall_below_floor(self):
        self.assertIsNone(linediff_best_match("abc def ghi", ["abc xyz uvw"]))

    def test_floor_value_itself_is_accepted(self):
        self.assertEqual(linediff_best_match("abc def", ["abc xyz"], Fraction(15, 28)), 0)
        self.assertIsNone(linediff_best_match("abc def", ["abc xyz"], Fraction(16, 28)))

    def test_ranking_orders_by_descending_score(self):
        self.assertEqual(
            linediff_rank_matches("abc def", ["zzz", "abc xyz", "abc def"]), [2, 1, 0]
        )

    def test_ranking_keeps_list_order_for_ties(self):
        self.assertEqual(linediff_rank_matches("a", ["b", "c", "a"]), [2, 0, 1])

    def test_pairing_matches_in_order(self):
        self.assertEqual(
            linediff_pair_changed_lines(["alpha one", "beta two"], ["beta twos", "alpha ones"]),
            [(0, 1), (1, 0)],
        )

    def test_pairing_never_reuses_an_added_line(self):
        self.assertEqual(
            linediff_pair_changed_lines(["alpha one", "alpha one"], ["alpha one"]), [(0, 0)]
        )

    def test_pairing_honours_custom_floor(self):
        self.assertEqual(linediff_pair_changed_lines(["abc def"], ["abc xyz"], Fraction(3, 5)), [])
