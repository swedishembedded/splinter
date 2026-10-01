// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Every tunable of the database, passed in as a value. Nothing here reads
//! the environment.

use std::time::Duration;

/// Smallest chunk a large object is split into.
pub const DEFAULT_CHUNK_MIN: u32 = 16 * 1024;
/// Target chunk size for content-defined chunking.
pub const DEFAULT_CHUNK_AVG: u32 = 64 * 1024;
/// Largest chunk a large object is split into.
pub const DEFAULT_CHUNK_MAX: u32 = 256 * 1024;
/// A blob pack is sealed once it holds this many bytes.
pub const DEFAULT_PACK_TARGET_BYTES: usize = 64 * 1024 * 1024;
/// Records per column block of a segment.
pub const DEFAULT_BLOCK_RECORDS: usize = 1024;
/// A writer seals its segment when this many records are buffered.
pub const DEFAULT_MAX_BUFFERED_RECORDS: usize = 65_536;
/// Payloads up to this size are stored inline in the record's block.
pub const DEFAULT_INLINE_PAYLOAD_LIMIT: usize = 4096;
/// A context log writes a full checkpoint after this many deltas.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 32;
/// Compaction stops merging a tier once a segment reaches this size.
pub const DEFAULT_COMPACT_TARGET_BYTES: usize = 1024 * 1024 * 1024;
/// An unpublished file younger than this is never collected.
pub const DEFAULT_ORPHAN_GRACE: Duration = Duration::from_secs(24 * 60 * 60);

/// How record blocks are compressed on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// Stored as is.
    None,
    /// Zstandard at the given level.
    Zstd(i32),
}

/// The database's settings.
#[derive(Debug, Clone)]
pub struct Config {
    /// Minimum chunk size for large objects.
    pub chunk_min: u32,
    /// Average chunk size for large objects.
    pub chunk_avg: u32,
    /// Maximum chunk size for large objects.
    pub chunk_max: u32,
    /// Bytes at which an open blob pack is sealed.
    pub pack_target_bytes: usize,
    /// Records per column block in a segment.
    pub block_records: usize,
    /// Records a writer buffers before sealing a segment; bounds its memory.
    pub max_buffered_records: usize,
    /// Payloads at most this large live inside the record block.
    pub inline_payload_limit: usize,
    /// Deltas between full checkpoints of a context log.
    pub checkpoint_every: usize,
    /// Block compression.
    pub compression: Compression,
    /// An unpublished file younger than this is never collected, because its
    /// writer may be about to publish it.
    pub orphan_grace: Duration,
    /// Compaction stops merging a tier once a segment reaches this size.
    pub compact_target_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_min: DEFAULT_CHUNK_MIN,
            chunk_avg: DEFAULT_CHUNK_AVG,
            chunk_max: DEFAULT_CHUNK_MAX,
            pack_target_bytes: DEFAULT_PACK_TARGET_BYTES,
            block_records: DEFAULT_BLOCK_RECORDS,
            max_buffered_records: DEFAULT_MAX_BUFFERED_RECORDS,
            inline_payload_limit: DEFAULT_INLINE_PAYLOAD_LIMIT,
            checkpoint_every: DEFAULT_CHECKPOINT_EVERY,
            compression: Compression::Zstd(3),
            orphan_grace: DEFAULT_ORPHAN_GRACE,
            compact_target_bytes: DEFAULT_COMPACT_TARGET_BYTES,
        }
    }
}
