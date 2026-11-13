# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

def count_words(text):
    counts = {}
    for word in text.lower().split():
        word = word.strip(".,;:!?")
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts


def top_words(text, n):
    counts = count_words(text)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:n]]


def longest_word(text):
    words = [w.strip(".,;:!?") for w in text.split()]
    if not words:
        return ""
    best = words[0]
    for word in words[1:]:
        if len(word) > len(best):
            best = word
    return best
