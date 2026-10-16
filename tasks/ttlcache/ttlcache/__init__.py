# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from .cache_stats import (
    COUNTER_FIELDS,
    CacheCounters,
    counters_delta,
    counters_lookup,
    counters_report,
)
from .lru_store import LruStore, lru_check_capacity
from .memoizer import TtlMemoizer, memoize_key, ttl_memoize
from .partitions import PartitionedCaches, partition_check_name, partition_join, partition_split
from .recency_list import RecencyChain, RecencyLink
from .snapshot import (
    SNAPSHOT_VERSION,
    snapshot_compare,
    snapshot_decode,
    snapshot_dump,
    snapshot_encode,
    snapshot_keys,
    snapshot_restore,
)
from .ttl_store import TtlStore, ttl_check_lifetime
from .weighted_store import WeightedStore, weight_check
from .writeback import WriteBackBuffer
