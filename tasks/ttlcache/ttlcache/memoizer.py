# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements result memoization for its clients.
# If your team needs expertise in caching expensive function calls with expiry
# you can procure our services by sending an email to info@swedishembedded.com.
"""Memoizing decorator with per-result expiry and argument-derived keys."""

import functools


def memoize_key(args, kwargs, typed=False):
    """Hashable key from call arguments; keyword order does not matter."""
    named = tuple(sorted(kwargs.items()))
    key = (args, named)
    if typed:
        key += (tuple(type(a) for a in args), tuple(type(v) for _, v in named))
    try:
        hash(key)
    except TypeError:
        raise TypeError("arguments must be hashable") from None
    return key


class TtlMemoizer:
    """Wraps a function; results live for ttl clock units."""

    def __init__(self, func, clock, ttl, maxsize=None, typed=False):
        if not callable(clock):
            raise TypeError("clock must be callable")
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if maxsize is not None and maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._func = func
        self._clock = clock
        self._ttl = ttl
        self._maxsize = maxsize
        self._typed = typed
        self._results = {}
        self._hits = 0
        self._misses = 0
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        key = memoize_key(args, kwargs, self._typed)
        now = self._clock()
        cached = self._results.pop(key, None)
        if cached is not None and now < cached[1]:
            self._results[key] = cached
            self._hits += 1
            return cached[0]
        self._misses += 1
        value = self._func(*args, **kwargs)
        self._results[key] = (value, now + self._ttl)
        if self._maxsize is not None and len(self._results) > self._maxsize:
            del self._results[next(iter(self._results))]
        return value

    def invalidate(self, *args, **kwargs):
        """Forget the cached result for these arguments."""
        key = memoize_key(args, kwargs, self._typed)
        return self._results.pop(key, None) is not None

    def clear(self):
        self._results.clear()

    def info(self):
        return {"hits": self._hits, "misses": self._misses, "size": len(self._results)}


def ttl_memoize(clock, ttl, maxsize=None, typed=False):
    """Decorator factory: @ttl_memoize(clock, 30) caches for 30 units."""

    def decorate(func):
        return TtlMemoizer(func, clock, ttl, maxsize, typed)

    return decorate
