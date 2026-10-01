// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Planning and running a query.

use std::collections::HashSet;

use super::logical::{Plan, Query, QueryResult};
use crate::error::Result;
use crate::format::Block;
use crate::id::RecordId;
use crate::index::SnapshotCache;
use crate::manifest::Snapshot;
use crate::model::Record;

impl Query {
    fn kind_mask(&self) -> u64 {
        self.kinds
            .iter()
            .fold(0, |mask, kind| mask | 1 << kind.code())
    }

    /// The envelope predicates, which need no body.
    fn envelope_matches(&self, block: &Block, row: usize) -> Result<bool> {
        let (from, to) = self.time.unwrap_or((0, u64::MAX));
        let ts = block.timestamp_ns(row);
        Ok(
            (self.kinds.is_empty() || self.kinds.contains(&block.kind(row)?))
                && self.family.is_none_or(|f| block.family(row) == Some(f))
                && self.attempt.is_none_or(|a| block.attempt(row) == Some(a))
                && self
                    .task_instance
                    .is_none_or(|t| block.task_instance(row) == Some(t))
                && ts >= from
                && ts < to,
        )
    }

    /// Whether an index can narrow the query at all; time and epistemic
    /// predicates alone cannot use one.
    fn is_indexable(&self) -> bool {
        !self.kinds.is_empty()
            || self.family.is_some()
            || self.attempt.is_some()
            || self.task_instance.is_some()
    }

    fn record_matches(&self, record: &Record) -> bool {
        self.epistemic.is_none_or(|e| record.body.epistemic() == e)
    }

    fn finish(&self, mut records: Vec<Record>) -> Vec<Record> {
        records.sort_by_key(|r| r.id);
        records.dedup_by_key(|r| r.id);
        if let Some(limit) = self.limit {
            records.truncate(limit);
        }
        records
    }
}

impl Snapshot {
    /// Whether an index already answers for every segment of the snapshot:
    /// built in memory, or persisted runs that cover it.
    fn index_covers_everything(&self) -> Result<bool> {
        let cache: &SnapshotCache = self.cache();
        if cache.has_index() {
            return Ok(true);
        }
        let mut covered = HashSet::new();
        for object in self.index_runs() {
            covered.extend(crate::index::run_covers(
                self.database().backend(),
                &object.id,
            )?);
        }
        Ok(self.segments().iter().all(|s| covered.contains(&s.id)))
    }

    fn total_blocks(&self) -> Result<u64> {
        let mut total = 0;
        for object in self.segments() {
            total += self.segment(object.id)?.info().blocks.len() as u64;
        }
        Ok(total)
    }

    /// Answers a query, from a persisted index if one covers the snapshot
    /// and otherwise by scanning with zone maps to skip blocks.
    pub fn query(&self, query: &Query) -> Result<QueryResult> {
        let blocks_total = self.total_blocks()?;
        let before = self.stats().blocks_read;
        let (plan, records) = if query.is_indexable() && self.index_covers_everything()? {
            (Plan::Index, self.query_by_index(query)?)
        } else {
            (Plan::Scan, self.query_by_scan(query)?)
        };
        let blocks_read = self.stats().blocks_read - before;
        Ok(QueryResult {
            records: query.finish(records),
            plan,
            blocks_total,
            blocks_read,
        })
    }

    fn query_by_scan(&self, query: &Query) -> Result<Vec<Record>> {
        let mask = query.kind_mask();
        let mut found = Vec::new();
        for object in self.segments() {
            let segment = self.segment(object.id)?;
            for (i, info) in segment.info().blocks.iter().enumerate() {
                let zone = &info.zone;
                let (from, to) = query.time.unwrap_or((0, u64::MAX));
                let may_match = (mask == 0 || zone.kind_mask & mask != 0)
                    && zone.overlaps_time(from, to)
                    && query.family.is_none_or(|f| zone.may_contain_family(&f))
                    && query.attempt.is_none_or(|a| zone.may_contain_attempt(a))
                    && query
                        .task_instance
                        .is_none_or(|t| zone.may_contain_task_instance(&t));
                if !may_match {
                    continue;
                }
                let block = segment.read_block(i)?;
                for row in 0..block.len() {
                    if query.envelope_matches(&block, row)? {
                        let record = block.record(row)?;
                        if query.record_matches(&record) {
                            found.push(record);
                        }
                    }
                }
            }
        }
        Ok(found)
    }

    fn query_by_index(&self, query: &Query) -> Result<Vec<Record>> {
        let index = self.index()?;
        let mut sets: Vec<Vec<RecordId>> = Vec::new();
        if !query.kinds.is_empty() {
            let mut union: Vec<RecordId> =
                query.kinds.iter().flat_map(|k| index.by_kind(*k)).collect();
            union.sort_unstable();
            union.dedup();
            sets.push(union);
        }
        if let Some(f) = query.family {
            sets.push(index.by_family(&f));
        }
        if let Some(a) = query.attempt {
            sets.push(index.by_attempt(a));
        }
        if let Some(t) = query.task_instance {
            sets.push(index.by_task_instance(&t));
        }
        let Some(smallest) = sets.iter().min_by_key(|s| s.len()) else {
            return self.query_by_scan(query);
        };
        let others: Vec<HashSet<RecordId>> =
            sets.iter().map(|s| s.iter().copied().collect()).collect();
        let candidates: Vec<RecordId> = smallest
            .iter()
            .copied()
            .filter(|id| others.iter().all(|o| o.contains(id)))
            .collect();
        let (from, to) = query.time.unwrap_or((0, u64::MAX));
        let mut found = Vec::new();
        for id in candidates {
            if let Some(record) = self.get(id)? {
                if record.timestamp_ns >= from
                    && record.timestamp_ns < to
                    && query.record_matches(&record)
                {
                    found.push(record);
                }
            }
        }
        Ok(found)
    }
}
