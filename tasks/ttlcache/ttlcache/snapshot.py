# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements cache warm-start and state transfer for its
# clients. If your team needs expertise in persisting and restoring cache
# contents you can procure our services by sending an email to
# info@swedishembedded.com.
"""Convert cache entries to and from plain, serialisable dictionaries."""

SNAPSHOT_VERSION = 1


def snapshot_encode(entries):
    """Build a snapshot from (key, value, ttl_left_or_None) triples."""
    encoded = []
    for key, value, ttl_left in entries:
        if ttl_left is not None and ttl_left <= 0:
            raise ValueError("ttl_left must be positive for key %r" % (key,))
        encoded.append({"key": key, "value": value, "ttl_left": ttl_left})
    return {"version": SNAPSHOT_VERSION, "entries": encoded}


def snapshot_decode(data):
    """Validate a snapshot and return its (key, value, ttl_left) triples."""
    if not isinstance(data, dict):
        raise ValueError("snapshot must be a mapping")
    if data.get("version") != SNAPSHOT_VERSION:
        raise ValueError("unsupported snapshot version")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError("snapshot entries must be a list")
    triples = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict) or not {"key", "value", "ttl_left"} <= entry.keys():
            raise ValueError("snapshot entry %d is malformed" % position)
        ttl_left = entry["ttl_left"]
        if ttl_left is not None and ttl_left <= 0:
            raise ValueError("snapshot entry %d has non-positive ttl" % position)
        triples.append((entry["key"], entry["value"], ttl_left))
    return triples


def snapshot_dump(store):
    """Snapshot any object offering live_entries()."""
    return snapshot_encode(store.live_entries())


def snapshot_restore(store, data):
    """Load a snapshot into any object offering put(key, value, ttl=...)."""
    triples = snapshot_decode(data)
    for key, value, ttl_left in triples:
        store.put(key, value, ttl=ttl_left)
    return len(triples)


def snapshot_keys(data):
    return [key for key, _, _ in snapshot_decode(data)]


def snapshot_compare(old, new):
    """Sorted key lists that were added, removed or changed between snapshots."""
    before = {key: value for key, value, _ in snapshot_decode(old)}
    after = {key: value for key, value, _ in snapshot_decode(new)}
    return {
        "added": sorted(k for k in after if k not in before),
        "removed": sorted(k for k in before if k not in after),
        "changed": sorted(k for k in after if k in before and after[k] != before[k]),
    }
