// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Repacking small blob packs.

use super::{group_by_size, CompactReport};
use crate::blob::BlobStore;
use crate::database::Database;
use crate::error::Result;
use crate::manifest::ObjectRef;

impl Database {
    /// Merges the snapshot's blob packs smaller than the pack target into
    /// larger ones. A chunk held by several packs is kept once.
    pub fn compact_blobs(&self) -> Result<CompactReport> {
        let snapshot = self.snapshot()?;
        let sized: Vec<(u64, ObjectRef)> = snapshot
            .blob_packs()
            .into_iter()
            .map(|p| (p.bytes, p))
            .collect();
        let mut report = CompactReport::default();
        for group in group_by_size(&sized, self.config().pack_target_bytes as u64) {
            let names: Vec<_> = group.iter().map(|p| p.id).collect();
            let merged = BlobStore::repack(self, &names)?;
            report.outputs.push(merged.id);
            let removed = group
                .iter()
                .filter(|o| o.id != merged.id)
                .cloned()
                .collect();
            self.publish_once("repacker", vec![merged], removed)?;
            report.groups += 1;
            report.inputs += group.len();
        }
        Ok(report)
    }
}
