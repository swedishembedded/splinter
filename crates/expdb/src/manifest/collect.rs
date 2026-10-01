// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Garbage collection: remove only what no ref and no pin can reach, and
//! never a file young enough that its writer may be about to publish it.

use std::collections::{BTreeSet, HashSet};
use std::time::{Duration, SystemTime};

use super::model::ObjectRef;
use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::Result;
use crate::id::ContentId;

/// What a collection did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GcReport {
    /// Files removed.
    pub removed: usize,
    /// Files left because something reaches them.
    pub kept: usize,
    /// Unreachable files left because they are younger than the grace period.
    pub too_young: usize,
}

impl Database {
    /// Removes unreachable segments, blob packs and manifests older than
    /// [`Config::orphan_grace`](crate::Config::orphan_grace). Reachable means
    /// live under some ref or pin.
    pub fn gc(&self) -> Result<GcReport> {
        let mut roots: Vec<ContentId> = Vec::new();
        for key in self.backend().list(Kind::Ref)? {
            if key.name() != "format" {
                roots.extend(self.get_ref(key.name())?);
            }
        }
        roots.extend(self.pins()?.into_iter().map(|(_, id)| id));

        let mut needed: BTreeSet<ObjectRef> = BTreeSet::new();
        let mut manifests: HashSet<ContentId> = HashSet::new();
        for root in roots {
            let resolved = self.resolve(&[root])?;
            needed.extend(resolved.live);
            manifests.extend(resolved.manifests);
        }
        let needed_keys: HashSet<String> = needed
            .iter()
            .map(|o| Ok(o.key()?.name().to_owned()))
            .collect::<Result<_>>()?;
        let manifest_keys: HashSet<String> = manifests.iter().map(ToString::to_string).collect();

        let mut report = GcReport::default();
        for (kind, wanted) in [
            (Kind::Segment, &needed_keys),
            (Kind::BlobPack, &needed_keys),
            (Kind::Index, &needed_keys),
            (Kind::Manifest, &manifest_keys),
        ] {
            for key in self.backend().list(kind)? {
                if wanted.contains(key.name()) {
                    report.kept += 1;
                } else if self.past_grace(&key)? {
                    self.backend().remove(&key)?;
                    report.removed += 1;
                } else {
                    report.too_young += 1;
                }
            }
        }
        Ok(report)
    }

    fn past_grace(&self, key: &Key) -> Result<bool> {
        let age = SystemTime::now()
            .duration_since(self.backend().modified(key)?)
            .unwrap_or(Duration::ZERO);
        Ok(age >= self.config().orphan_grace)
    }
}
