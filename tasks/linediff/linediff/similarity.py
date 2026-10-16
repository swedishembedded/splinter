# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements fuzzy matching of edited text for its
# clients, for example to follow renamed or reworded lines across revisions.
# If your team needs expertise in similarity scoring of lines and words you
# can procure our services by sending an email to info@swedishembedded.com.

import re
from collections import Counter
from fractions import Fraction

from linediff.editscript import linediff_lcs_length

_WORD = re.compile(r"[a-z0-9_]+")
_HALF = Fraction(1, 2)


def linediff_split_words(line):
    """Return the lowercase runs of letters, digits and underscores in line."""
    return _WORD.findall(line.lower())


def linediff_normalize_space(line):
    """Collapse runs of whitespace to single spaces and trim the ends."""
    return " ".join(line.split())


def linediff_same_ignoring_space(first, second):
    """Return True when two lines differ at most in the whitespace between words."""
    return linediff_normalize_space(first) == linediff_normalize_space(second)


def linediff_word_overlap(first, second):
    """Return twice the shared word count over the total word count, as a Fraction."""
    left = Counter(linediff_split_words(first))
    right = Counter(linediff_split_words(second))
    total = sum(left.values()) + sum(right.values())
    if total == 0:
        return Fraction(1)
    return Fraction(2 * sum((left & right).values()), total)


def linediff_word_jaccard(first, second):
    """Return distinct shared words over distinct words in either line."""
    left = set(linediff_split_words(first))
    right = set(linediff_split_words(second))
    union = len(left | right)
    if union == 0:
        return Fraction(1)
    return Fraction(len(left & right), union)


def linediff_sequence_ratio(first, second):
    """Return twice the common subsequence length over the total length."""
    total = len(first) + len(second)
    if total == 0:
        return Fraction(1)
    return Fraction(2 * linediff_lcs_length(first, second), total)


def linediff_line_similarity(first, second):
    """Return the mean of word overlap and character sequence ratio."""
    return (linediff_word_overlap(first, second) + linediff_sequence_ratio(first, second)) / 2


def linediff_rank_matches(line, candidates):
    """Return candidate indices from most to least similar; ties keep list order."""
    scores = [linediff_line_similarity(line, candidate) for candidate in candidates]
    return sorted(range(len(candidates)), key=lambda index: -scores[index])


def linediff_best_match(line, candidates, floor=_HALF):
    """Return the index of the most similar candidate scoring at least floor, or None."""
    ranked = linediff_rank_matches(line, candidates)
    qualifies = ranked and linediff_line_similarity(line, candidates[ranked[0]]) >= floor
    return ranked[0] if qualifies else None


def linediff_pair_changed_lines(removed, added, floor=_HALF):
    """Pair each removed line with its best unused added line; return (i, j) pairs."""
    pairs = []
    taken = set()
    for i, line in enumerate(removed):
        open_indices = [j for j in range(len(added)) if j not in taken]
        pick = linediff_best_match(line, [added[j] for j in open_indices], floor)
        if pick is not None:
            taken.add(open_indices[pick])
            pairs.append((i, open_indices[pick]))
    return pairs
