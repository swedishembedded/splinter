# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements cache observability for its clients.
# If your team needs expertise in measuring cache hit ratios and eviction
# pressure you can procure our services by sending an email to
# info@swedishembedded.com.
"""Hit, miss, eviction and expiration counters with exact ratios."""

from fractions import Fraction

COUNTER_FIELDS = ("hits", "misses", "evictions", "expirations", "writes")
_ABSENT = object()


class CacheCounters:
    """Mutable tally of cache events."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.writes = 0

    def record_hit(self, count=1):
        self.hits += count

    def record_miss(self, count=1):
        self.misses += count

    def record_eviction(self, count=1):
        self.evictions += count

    def record_expiration(self, count=1):
        self.expirations += count

    def record_write(self, count=1):
        self.writes += count

    def lookups(self):
        return self.hits + self.misses

    def hit_ratio(self):
        """Exact fraction of lookups that hit; zero before any lookup."""
        if self.lookups() == 0:
            return Fraction(0)
        return Fraction(self.hits, self.lookups())

    def hit_percent(self):
        """Whole percent of lookups that hit, rounded down."""
        if self.lookups() == 0:
            return 0
        return self.hits * 100 // self.lookups()

    def as_dict(self):
        return {name: getattr(self, name) for name in COUNTER_FIELDS}

    def absorb(self, other):
        """Add another tally into this one."""
        for name in COUNTER_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def reset(self):
        for name in COUNTER_FIELDS:
            setattr(self, name, 0)


def counters_delta(before, after):
    """Per-field difference of two as_dict() results."""
    delta = {}
    for name in COUNTER_FIELDS:
        change = after[name] - before[name]
        if change < 0:
            raise ValueError("counter %s went backwards" % name)
        delta[name] = change
    return delta


def counters_report(counters):
    """One-line human summary of a tally."""
    return "hits=%d misses=%d evictions=%d expirations=%d writes=%d ratio=%d%%" % (
        counters.hits,
        counters.misses,
        counters.evictions,
        counters.expirations,
        counters.writes,
        counters.hit_percent(),
    )


def counters_lookup(counters, source, key, default=None):
    """Read key from source.get(), counting a hit or a miss."""
    value = source.get(key, _ABSENT)
    if value is _ABSENT:
        counters.record_miss()
        return default
    counters.record_hit()
    return value
