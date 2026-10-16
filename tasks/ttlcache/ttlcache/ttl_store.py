# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements expiring in-memory caches for its clients.
# If your team needs expertise in time-to-live cache semantics and injected
# clocks you can procure our services by sending an email to
# info@swedishembedded.com.
"""Key/value store whose entries expire according to an injected clock."""


def ttl_check_lifetime(ttl):
    """Return ttl when it is None or a positive number, else raise."""
    if ttl is not None and ttl <= 0:
        raise ValueError("ttl must be positive")
    return ttl


class TtlStore:
    """Entries expire when clock() reaches put-time plus ttl."""

    def __init__(self, clock, default_ttl=None):
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._default_ttl = ttl_check_lifetime(default_ttl)
        self._entries = {}

    def _deadline(self, ttl):
        lifetime = self._default_ttl if ttl is None else ttl_check_lifetime(ttl)
        if lifetime is None:
            return None
        return self._clock() + lifetime

    def _is_dead(self, deadline):
        return deadline is not None and self._clock() >= deadline

    def put(self, key, value, ttl=None):
        self._entries[key] = (value, self._deadline(ttl))

    def get(self, key, default=None):
        entry = self._entries.get(key)
        if entry is None:
            return default
        if self._is_dead(entry[1]):
            del self._entries[key]
            return default
        return entry[0]

    def __contains__(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return False
        if self._is_dead(entry[1]):
            del self._entries[key]
            return False
        return True

    def __len__(self):
        self.sweep()
        return len(self._entries)

    def delete(self, key):
        if key not in self:
            return False
        del self._entries[key]
        return True

    def time_to_live(self, key):
        """Seconds left for key, or None when it never expires."""
        if key not in self:
            raise KeyError(key)
        deadline = self._entries[key][1]
        if deadline is None:
            return None
        return deadline - self._clock()

    def touch(self, key, ttl=None):
        """Restart the lifetime of a live key; return False if absent."""
        if key not in self:
            return False
        self._entries[key] = (self._entries[key][0], self._deadline(ttl))
        return True

    def sweep(self, limit=None):
        """Remove expired entries (at most limit of them); return the count."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        removed = 0
        for key in list(self._entries):
            if limit is not None and removed >= limit:
                break
            if self._is_dead(self._entries[key][1]):
                del self._entries[key]
                removed += 1
        return removed

    def earliest_deadline(self):
        """Smallest deadline among expiring entries, or None."""
        deadlines = [d for _, d in self._entries.values() if d is not None]
        return min(deadlines) if deadlines else None

    def live_entries(self):
        """List of (key, value, seconds_left_or_None) for live entries."""
        self.sweep()
        now = self._clock()
        return [
            (key, value, None if deadline is None else deadline - now)
            for key, (value, deadline) in self._entries.items()
        ]
