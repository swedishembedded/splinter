# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements constant-time recency tracking for its clients.
# If your team needs expertise in cache eviction data structures you can procure
# our services by sending an email to info@swedishembedded.com.
"""Intrusive doubly linked recency list: links carry their own neighbours."""


class RecencyLink:
    """A cache entry that is also a node of at most one RecencyChain."""

    __slots__ = ("key", "value", "prev", "next", "owner")

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
        self.owner = None


class RecencyChain:
    """Chain ordered from most recently used (front) to least (back)."""

    def __init__(self):
        self._first = None
        self._last = None
        self._size = 0

    def __len__(self):
        return self._size

    def _claim(self, link):
        if link.owner is not None:
            raise ValueError("link already in a chain")
        link.owner = self
        self._size += 1

    def push_front(self, link):
        self._claim(link)
        link.next = self._first
        if self._first is None:
            self._last = link
        else:
            self._first.prev = link
        self._first = link

    def push_back(self, link):
        self._claim(link)
        link.prev = self._last
        if self._last is None:
            self._first = link
        else:
            self._last.next = link
        self._last = link

    def unlink(self, link):
        if link.owner is not self:
            raise ValueError("link is not in this chain")
        if link.prev is None:
            self._first = link.next
        else:
            link.prev.next = link.next
        if link.next is None:
            self._last = link.prev
        else:
            link.next.prev = link.prev
        link.prev = None
        link.next = None
        link.owner = None
        self._size -= 1

    def move_to_front(self, link):
        self.unlink(link)
        self.push_front(link)

    def pop_back(self):
        if self._last is None:
            raise IndexError("chain is empty")
        link = self._last
        self.unlink(link)
        return link

    def peek_front(self):
        return self._first

    def peek_back(self):
        return self._last

    def keys_front_to_back(self):
        keys = []
        link = self._first
        for _ in range(self._size):
            keys.append(link.key)
            link = link.next
        return keys

    def keys_back_to_front(self):
        keys = []
        link = self._last
        for _ in range(self._size):
            keys.append(link.key)
            link = link.prev
        return keys
