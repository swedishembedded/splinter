# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements memory-budgeted caches for its clients.
# If your team needs expertise in size-aware cache eviction you can procure our
# services by sending an email to info@swedishembedded.com.
"""Cache bounded by the total weight of its entries rather than their count."""


def weight_check(weight):
    """Return weight when it is a positive integer, else raise."""
    if not isinstance(weight, int):
        raise TypeError("weight must be an integer")
    if weight < 1:
        raise ValueError("weight must be at least 1")
    return weight


class WeightedStore:
    """Evicts least recently used entries until the total weight fits."""

    def __init__(self, max_weight, on_evict=None):
        self._max_weight = weight_check(max_weight)
        self._on_evict = on_evict
        self._entries = {}
        self._total = 0

    @property
    def max_weight(self):
        return self._max_weight

    @property
    def total_weight(self):
        return self._total

    def remaining_weight(self):
        return self._max_weight - self._total

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        return key in self._entries

    def get(self, key, default=None):
        if key not in self._entries:
            return default
        entry = self._entries.pop(key)
        self._entries[key] = entry
        return entry[0]

    def weight_of(self, key):
        if key not in self._entries:
            raise KeyError(key)
        return self._entries[key][1]

    def put(self, key, value, weight):
        """Store value; entries heavier than the whole budget are refused."""
        weight_check(weight)
        if weight > self._max_weight:
            raise ValueError("weight exceeds the cache budget")
        self.delete(key)
        self._entries[key] = (value, weight)
        self._total += weight
        self._trim_to(self._max_weight, protect=key)

    def _trim_to(self, budget, protect=None):
        for key in list(self._entries):
            if self._total <= budget:
                break
            if key != protect:
                self._drop(key, notify=True)

    def _drop(self, key, notify):
        value, weight = self._entries.pop(key)
        self._total -= weight
        if notify and self._on_evict is not None:
            self._on_evict(key, value)

    def delete(self, key):
        if key not in self._entries:
            return False
        self._drop(key, notify=False)
        return True

    def resize(self, max_weight):
        self._max_weight = weight_check(max_weight)
        self._trim_to(max_weight)

    def evict_heaviest(self):
        """Remove and return (key, value) of the heaviest entry."""
        if not self._entries:
            raise KeyError("store is empty")
        victim = max(self._entries, key=lambda k: self._entries[k][1])
        value = self._entries[victim][0]
        self._drop(victim, notify=False)
        return victim, value

    def keys_oldest_first(self):
        return list(self._entries)
