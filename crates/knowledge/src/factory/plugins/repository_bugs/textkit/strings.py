# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

def slugify(text):
    out = []
    previous_dash = True
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            previous_dash = False
        elif not previous_dash:
            out.append("-")
            previous_dash = True
    return "".join(out).strip("-")


def truncate(text, limit):
    if limit < 3:
        raise ValueError("limit must be at least 3")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def is_palindrome(text):
    letters = [c.lower() for c in text if c.isalnum()]
    return letters == letters[::-1]


def title_case(text):
    return " ".join(w[:1].upper() + w[1:].lower() for w in text.split(" "))
