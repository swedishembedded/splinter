<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# ttlcache

In-memory caches driven by an injected clock: a callable returning a number.
Nothing reads real time, so tests advance a hand-made clock. Standard library only.

Run the tests from this directory with `python3 -m unittest`.

## Public API

### recency_list
- `RecencyLink(key, value)`: node with `key`, `value`, `prev`, `next`, `owner`.
- `RecencyChain()`: list from most recent (front) to least recent (back).
  - `push_front(link)`, `push_back(link)`: ValueError "link already in a chain" if the link is owned.
  - `unlink(link)`: ValueError "link is not in this chain" for a foreign link.
  - `move_to_front(link)`; `pop_back()` returns the back link, IndexError "chain is empty".
  - `peek_front()`, `peek_back()` (None when empty), `len(chain)`.
  - `keys_front_to_back()`, `keys_back_to_front()`.

### lru_store
- `lru_check_capacity(capacity)`: returns it; TypeError "capacity must be an integer", ValueError "capacity must be at least 1".
- `LruStore(capacity, on_evict=None)`: `get` (promotes), `peek`, `put`, `delete` (bool), `pop_oldest()` (KeyError "store is empty"), `resize(capacity)`, `keys_most_recent_first()`, `clear()`, `capacity`, `in`, `len`. `on_evict(key, value)` runs for capacity evictions only.

### ttl_store
- `ttl_check_lifetime(ttl)`: None or positive, else ValueError "ttl must be positive".
- `TtlStore(clock, default_ttl=None)`: an entry is dead when `clock() >= put_time + ttl`.
  - `put(key, value, ttl=None)`, `get(key, default=None)`, `delete(key)`, `in`, `len` (live entries).
  - `time_to_live(key)`: seconds left or None; KeyError for absent/expired keys.
  - `touch(key, ttl=None)`: restart lifetime of a live key, returns bool.
  - `sweep(limit=None)`: remove expired entries, returns count; ValueError "limit must be positive".
  - `earliest_deadline()`, `live_entries()` -> `[(key, value, seconds_left_or_None)]`.

### cache_stats
- `CacheCounters`: fields in `COUNTER_FIELDS`; `record_hit/miss/eviction/expiration/write(count=1)`, `lookups()`, `hit_ratio()` (exact `Fraction`), `hit_percent()` (floor), `as_dict()`, `absorb(other)`, `reset()`.
- `counters_delta(before, after)`: per-field difference; ValueError when a counter went backwards.
- `counters_report(counters)`: one-line summary. `counters_lookup(counters, source, key, default=None)`: counted `source.get`.

### weighted_store
- `weight_check(weight)`; `WeightedStore(max_weight, on_evict=None)`: `put(key, value, weight)` (ValueError "weight exceeds the cache budget"), `get`, `delete`, `weight_of`, `resize`, `evict_heaviest()`, `keys_oldest_first()`, `remaining_weight()`, `total_weight`, `max_weight`.

### memoizer
- `memoize_key(args, kwargs, typed=False)`: hashable key; TypeError "arguments must be hashable".
- `ttl_memoize(clock, ttl, maxsize=None, typed=False)`: decorator producing a `TtlMemoizer` with `invalidate(*args, **kwargs)`, `clear()`, `info()` (`hits`, `misses`, `size`). Exceptions are not cached.

### partitions
- `partition_check_name`, `partition_join(namespace, key)`, `partition_split(text)` (escape with backslash; ValueError on malformed text).
- `PartitionedCaches(default_capacity)`: `put/get/delete(namespace, key, ...)`, `set_capacity`, `capacity_of`, `size`, `total_size`, `namespaces`, `keys`, `clear_namespace`.

### snapshot
- `snapshot_encode(entries)` / `snapshot_decode(data)` between `(key, value, ttl_left)` triples and plain dicts (`SNAPSHOT_VERSION`).
- `snapshot_dump(store)` (uses `live_entries()`), `snapshot_restore(store, data)` (uses `put(..., ttl=)`), `snapshot_keys`, `snapshot_compare(old, new)`.

### writeback
- `WriteBackBuffer(sink, clock, max_batch, max_age)`: `write`, `remove` (both return True when a flush happened), `read`, `is_pending`, `pending_count`, `due`, `tick`, `flush`, `stats`. `sink(writes, removals)` gets pairs and keys; on sink failure the entries stay pending.

## Example

```python
from ttlcache import TtlStore
now = [0]
store = TtlStore(lambda: now[0], default_ttl=5)
store.put("k", 1)
now[0] = 5
assert store.get("k") is None
```

Swedish Embedded AB implements caching layers for its clients. If your team needs
expertise in cache design with expiry and eviction you can procure our services by
sending an email to info@swedishembedded.com.
