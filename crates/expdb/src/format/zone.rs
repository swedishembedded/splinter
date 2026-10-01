// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Zone maps: what a block can possibly hold, so a scan reads only blocks
//! that might match.

use serde::{Deserialize, Serialize};

use crate::id::{ContentId, RecordId};
use crate::model::{Record, RecordKind};

/// Bits in a block's Bloom filter, as 64-bit words.
const BLOOM_WORDS: usize = 128;
const BLOOM_PROBES: u32 = 3;

const TAG_ID: u8 = 1;
const TAG_ATTEMPT: u8 = 2;
const TAG_FAMILY: u8 = 3;
const TAG_INSTANCE: u8 = 4;

fn mix(tag: u8, value: u64) -> u64 {
    let mut x = value ^ u64::from(tag).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Summary of one block: the kinds and times in it and a Bloom filter over
/// the ids, attempts, families and task instances of its records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Zone {
    /// Bit `k` is set if the block holds a record of kind code `k`.
    pub kind_mask: u64,
    /// The earliest timestamp in the block.
    pub ts_min: u64,
    /// The latest timestamp in the block.
    pub ts_max: u64,
    bloom: Vec<u64>,
}

impl Zone {
    pub(crate) fn of(records: &[Record]) -> Self {
        let mut zone = Zone {
            kind_mask: 0,
            ts_min: u64::MAX,
            ts_max: 0,
            bloom: vec![0; BLOOM_WORDS],
        };
        for record in records {
            zone.kind_mask |= 1 << record.kind().code();
            zone.ts_min = zone.ts_min.min(record.timestamp_ns);
            zone.ts_max = zone.ts_max.max(record.timestamp_ns);
            zone.insert(TAG_ID, record.id.hash64());
            if let Some(attempt) = record.attempt {
                zone.insert(TAG_ATTEMPT, attempt.hash64());
            }
            if let Some(family) = record.family {
                zone.insert(TAG_FAMILY, family.prefix_u64());
            }
            if let Some(instance) = record.task_instance {
                zone.insert(TAG_INSTANCE, instance.prefix_u64());
            }
        }
        zone
    }

    fn positions(tag: u8, value: u64) -> impl Iterator<Item = (usize, u64)> {
        let h = mix(tag, value);
        (0..BLOOM_PROBES).map(move |probe| {
            let bit = h.rotate_left(probe * 21) % (BLOOM_WORDS as u64 * 64);
            ((bit / 64) as usize, 1u64 << (bit % 64))
        })
    }

    fn insert(&mut self, tag: u8, value: u64) {
        for (word, mask) in Self::positions(tag, value) {
            self.bloom[word] |= mask;
        }
    }

    fn may_contain(&self, tag: u8, value: u64) -> bool {
        Self::positions(tag, value)
            .all(|(word, mask)| self.bloom.get(word).is_some_and(|w| w & mask != 0))
    }

    /// Whether the block may hold a record of `kind`; exact.
    pub fn has_kind(&self, kind: RecordKind) -> bool {
        self.kind_mask & (1 << kind.code()) != 0
    }

    /// Whether the block may overlap the half-open time range.
    pub fn overlaps_time(&self, from: u64, to: u64) -> bool {
        self.ts_min < to && self.ts_max >= from
    }

    /// Whether the block may hold the record. A `false` is certain.
    pub fn may_contain_id(&self, id: RecordId) -> bool {
        self.may_contain(TAG_ID, id.hash64())
    }

    /// Whether the block may hold a record of the attempt. A `false` is certain.
    pub fn may_contain_attempt(&self, attempt: RecordId) -> bool {
        self.may_contain(TAG_ATTEMPT, attempt.hash64())
    }

    /// Whether the block may hold a record of the family. A `false` is certain.
    pub fn may_contain_family(&self, family: &ContentId) -> bool {
        self.may_contain(TAG_FAMILY, family.prefix_u64())
    }

    /// Whether the block may hold a record of the task instance. A `false` is certain.
    pub fn may_contain_task_instance(&self, instance: &ContentId) -> bool {
        self.may_contain(TAG_INSTANCE, instance.prefix_u64())
    }
}
