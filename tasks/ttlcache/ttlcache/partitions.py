# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements tenant-isolated caching for its clients.
# If your team needs expertise in namespaced and partitioned cache layouts you
# can procure our services by sending an email to info@swedishembedded.com.
"""Namespaced caches: each namespace has its own entries and capacity."""

SEPARATOR = ":"
ESCAPE = "\\"
_MISSING = object()


def partition_check_name(namespace):
    """Return namespace when it is a non-empty string, else raise."""
    if not isinstance(namespace, str) or namespace == "":
        raise ValueError("namespace must be a non-empty string")
    return namespace


def partition_join(namespace, key):
    """Flatten (namespace, key) into one string, escaping separators."""
    escaped = namespace.replace(ESCAPE, ESCAPE + ESCAPE).replace(SEPARATOR, ESCAPE + SEPARATOR)
    return escaped + SEPARATOR + key


def partition_split(text):
    """Inverse of partition_join; raises ValueError for malformed text."""
    namespace = []
    chars = iter(text)
    for char in chars:
        if char == ESCAPE:
            escaped = next(chars, None)
            if escaped is None:
                raise ValueError("dangling escape in %r" % text)
            namespace.append(escaped)
        elif char == SEPARATOR:
            return "".join(namespace), "".join(chars)
        else:
            namespace.append(char)
    raise ValueError("missing separator in %r" % text)


class PartitionedCaches:
    """Independent least-recently-used caches keyed by namespace."""

    def __init__(self, default_capacity):
        if default_capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._default_capacity = default_capacity
        self._capacities = {}
        self._spaces = {}

    def capacity_of(self, namespace):
        return self._capacities.get(namespace, self._default_capacity)

    def set_capacity(self, namespace, capacity):
        partition_check_name(namespace)
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacities[namespace] = capacity
        self._trim(namespace)

    def _trim(self, namespace):
        space = self._spaces.get(namespace, {})
        excess = len(space) - self.capacity_of(namespace)
        if excess > 0:
            for key in list(space)[:excess]:
                del space[key]

    def put(self, namespace, key, value):
        partition_check_name(namespace)
        space = self._spaces.setdefault(namespace, {})
        space.pop(key, None)
        space[key] = value
        self._trim(namespace)

    def get(self, namespace, key, default=None):
        space = self._spaces.get(namespace, {})
        if key not in space:
            return default
        value = space.pop(key)
        space[key] = value
        return value

    def delete(self, namespace, key):
        return self._spaces.get(namespace, {}).pop(key, _MISSING) is not _MISSING

    def size(self, namespace):
        return len(self._spaces.get(namespace, ()))

    def total_size(self):
        return sum(len(space) for space in self._spaces.values())

    def namespaces(self):
        """Names of namespaces that currently hold entries, sorted."""
        return sorted(name for name, space in self._spaces.items() if space)

    def keys(self, namespace):
        """Keys of one namespace, least recently used first."""
        return list(self._spaces.get(namespace, ()))

    def clear_namespace(self, namespace):
        """Empty one namespace and return how many entries it held."""
        return len(self._spaces.pop(namespace, ()))
