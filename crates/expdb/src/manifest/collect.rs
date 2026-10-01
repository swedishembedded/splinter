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
        // The database as it stands: every ref read together, so a removal
        // published by one process applies to what another still names.
        let mut heads: Vec<ContentId> = Vec::new();
        for key in self.backend().list(Kind::Ref)? {
            if key.name() != "format" {
                heads.extend(self.get_ref(key.name())?);
            }
        }
        let mut needed: BTreeSet<ObjectRef> = BTreeSet::new();
        let mut manifests: HashSet<ContentId> = HashSet::new();
        let current = self.resolve(&heads)?;
        needed.extend(current.live);
        manifests.extend(current.manifests);
        // A pin keeps its own snapshot whole, whatever came after.
        for (_, pinned) in self.pins()? {
            let resolved = self.resolve(&[pinned])?;
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

/// What absorbing idle job refs did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AbsorbReport {
    /// Job refs removed because the catalog already holds their history.
    pub pruned: usize,
}

impl Database {
    /// Folds every job's history into the catalog and removes the refs of
    /// jobs that have been idle for the orphan grace period, so ten thousand
    /// finished writers do not leave ten thousand refs behind.
    ///
    /// A ref is removed only if the catalog holds its manifest and it has not
    /// been written for the grace period. A writer that publishes again after
    /// that simply starts a new chain; what it published before is in the
    /// catalog. The grace period is what makes removing a ref safe against a
    /// writer publishing at the same moment, so it must be longer than any
    /// writer's pause between publishes.
    pub fn absorb_jobs(&self) -> Result<AbsorbReport> {
        let catalog = self.merge_catalog()?;
        let absorbed = self.resolve(&[catalog])?.manifests;
        let mut report = AbsorbReport::default();
        for key in self.backend().list(Kind::Ref)? {
            if !key.name().starts_with("jobs/") {
                continue;
            }
            let Some(head) = self.get_ref(key.name())? else {
                continue;
            };
            if absorbed.contains(&head) && self.past_grace(&key)? {
                self.backend().remove(&key)?;
                report.pruned += 1;
            }
        }
        Ok(report)
    }
}
