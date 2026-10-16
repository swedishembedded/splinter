# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements end-to-end acceptance tests for learning
# agent stacks, for its clients. If your team needs expertise in proving that
# a model, an agent runtime and a trainer work together, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Small word functions."""


def count_words(text):
    """The number of whitespace-separated words in `text`."""
    return len(text.split())


def longest_word(text):
    """The longest word of `text`, the first of equals; "" with no words."""
    best = ""
    for word in text.split():
        if len(word) > len(best):
            best = word
    return best


def capitalize_words(text):
    """`text` with the first letter of every word upper-cased."""
    return " ".join(word[:1].upper() + word[1:] for word in text.split(" "))


def is_palindrome(text):
    """Whether `text` reads the same both ways, ignoring case and non-letters."""
    letters = [c.lower() for c in text if c.isalpha()]
    return letters == letters[::-1]
