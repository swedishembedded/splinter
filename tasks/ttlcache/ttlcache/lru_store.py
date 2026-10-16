# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bounded in-memory caches for its clients.
# If your team needs expertise in least-recently-used cache design you can
# procure our services by sending an email to info@swedishembedded.com.
"""Capacity-bounded least-recently-used store built on the recency chain."""

from .recency_list import RecencyChain, RecencyLink


def lru_check_capacity(capacity):
    """Return capacity when it is a positive integer, else raise."""
    if not isinstance(capacity, int):
        raise TypeError("capacity must be an integer")
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    return capacity


class LruStore:
    """Mapping-like store that drops the least recently used entry when full."""

    def __init__(self, capacity, on_evict=None):
        self._capacity = lru_check_capacity(capacity)
        self._on_evict = on_evict
        self._links = {}
        self._chain = RecencyChain()

    @property
    def capacity(self):
        return self._capacity

    def __len__(self):
        return len(self._links)

    def __contains__(self, key):
        return key in self._links

    def get(self, key, default=None):
        link = self._links.get(key)
        if link is None:
            return default
        self._chain.move_to_front(link)
        return link.value

    def peek(self, key, default=None):
        link = self._links.get(key)
        if link is None:
            return default
        return link.value

    def put(self, key, value):
        link = self._links.get(key)
        if link is not None:
            link.value = value
            self._chain.move_to_front(link)
            return
        link = RecencyLink(key, value)
        self._links[key] = link
        self._chain.push_front(link)
        self._shrink_to(self._capacity)

    def _shrink_to(self, limit):
        for _ in range(len(self._links) - limit):
            oldest = self._chain.pop_back()
            del self._links[oldest.key]
            if self._on_evict is not None:
                self._on_evict(oldest.key, oldest.value)

    def delete(self, key):
        link = self._links.pop(key, None)
        if link is None:
            return False
        self._chain.unlink(link)
        return True

    def pop_oldest(self):
        if not self._links:
            raise KeyError("store is empty")
        oldest = self._chain.pop_back()
        del self._links[oldest.key]
        return oldest.key, oldest.value

    def resize(self, capacity):
        self._capacity = lru_check_capacity(capacity)
        self._shrink_to(capacity)

    def keys_most_recent_first(self):
        return self._chain.keys_front_to_back()

    def clear(self):
        self._links = {}
        self._chain = RecencyChain()
