# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements write-behind persistence layers for its
# clients. If your team needs expertise in batching and coalescing writes to a
# slow backing store you can procure our services by sending an email to
# info@swedishembedded.com.
"""Write-back buffer: coalesce writes and hand them to a sink in batches."""

_REMOVED = object()


class WriteBackBuffer:
    """Collects writes and removals; flushes by batch size or age.

    sink(writes, removals) receives a list of (key, value) pairs and a list
    of keys, both in the order the keys first became pending.
    """

    def __init__(self, sink, clock, max_batch, max_age):
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        if max_age <= 0:
            raise ValueError("max_age must be positive")
        self._sink = sink
        self._clock = clock
        self._max_batch = max_batch
        self._max_age = max_age
        self._pending = {}
        self._oldest_at = None
        self._flushes = 0
        self._failures = 0
        self._delivered = 0

    def _stage(self, key, value):
        if not self._pending:
            self._oldest_at = self._clock()
        self._pending[key] = value
        if len(self._pending) >= self._max_batch or self.due():
            self.flush()
            return True
        return False

    def write(self, key, value):
        """Buffer a write; return True when it triggered a flush."""
        return self._stage(key, value)

    def remove(self, key):
        """Buffer a removal; return True when it triggered a flush."""
        return self._stage(key, _REMOVED)

    def read(self, key, default=None):
        """Newest pending value for key; removed or unknown keys give default."""
        value = self._pending.get(key, _REMOVED)
        return default if value is _REMOVED else value

    def is_pending(self, key):
        return key in self._pending

    def pending_count(self):
        return len(self._pending)

    def due(self):
        """True when the oldest pending entry has waited max_age or more."""
        return bool(self._pending) and self._clock() - self._oldest_at >= self._max_age

    def tick(self):
        """Flush if the age limit has been reached; return whether it did."""
        if not self.due():
            return False
        self.flush()
        return True

    def flush(self):
        """Deliver everything pending; return the number of entries sent."""
        if not self._pending:
            return 0
        writes = [(k, v) for k, v in self._pending.items() if v is not _REMOVED]
        removals = [k for k, v in self._pending.items() if v is _REMOVED]
        try:
            self._sink(writes, removals)
        except Exception:
            self._failures += 1
            raise
        count = len(self._pending)
        self._pending = {}
        self._oldest_at = None
        self._flushes += 1
        self._delivered += count
        return count

    def stats(self):
        return {
            "flushes": self._flushes,
            "failures": self._failures,
            "delivered": self._delivered,
            "pending": len(self._pending),
        }
